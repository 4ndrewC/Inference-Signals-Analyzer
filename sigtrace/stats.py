"""Tensor summary statistics.

Everything here turns a (possibly huge, possibly GPU-resident) tensor into a
small JSON-serialisable dict. Reductions are stacked and moved to the host in a
single transfer so that tracing a CUDA model costs one sync per tensor rather
than one per statistic.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)
NAN_CODE = 255  # reserved byte in quantised heatmaps for non-finite values


@dataclass
class StatsConfig:
    hist_bins: int = 64
    heatmap: bool = True
    heatmap_size: tuple[int, int] = (64, 128)  # max (rows, cols)
    mosaic_channels: int = 16  # conv feature maps tiled into the heatmap
    channel_bins: int = 256  # max length of the per-channel profile
    sample_size: int = 65536  # elements used for quantiles


def _num(v: float) -> float | None:
    """JSON-safe float with ~5 significant digits (NaN/Inf -> None)."""
    if v is None or not math.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.5g}")


def _kind(t: torch.Tensor) -> str:
    if t.dtype == torch.bool:
        return "bool"
    if t.is_complex():
        return "complex"
    if t.is_floating_point():
        return "float"
    return "int"


def _quantise(img: torch.Tensor) -> dict[str, Any]:
    """Encode a 2-D float tensor as base64 uint8 (0..254 linear, 255 = non-finite)."""
    finite = torch.isfinite(img)
    if finite.any():
        vals = img[finite]
        lo, hi = vals.min().item(), vals.max().item()
    else:
        lo, hi = 0.0, 0.0
    span = hi - lo if hi > lo else 1.0
    q = ((img - lo) / span * 254).round().clamp(0, 254)
    q = torch.where(finite, q, torch.full_like(q, NAN_CODE)).to(torch.uint8)
    return {
        "rows": int(img.shape[0]),
        "cols": int(img.shape[1]),
        "lo": _num(lo),
        "hi": _num(hi),
        "q": base64.b64encode(q.cpu().numpy().tobytes()).decode("ascii"),
    }


def _pool2d(img: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Average-pool a 2-D tensor down to at most (rows, cols)."""
    r, c = img.shape
    if r <= rows and c <= cols:
        return img
    out = F.adaptive_avg_pool2d(img[None, None], (min(r, rows), min(c, cols)))
    return out[0, 0]


def _heatmap(x: torch.Tensor, cfg: StatsConfig) -> dict[str, Any] | None:
    """Reduce an activation to a 2-D image that keeps its natural layout.

    * 1-D  [N]            -> 1 x N
    * 2-D  [B, F]         -> B x F
    * 3-D  [B, S, D]      -> S x D of the first batch element
    * 4-D  [B, C, H, W]   -> mosaic of the first `mosaic_channels` feature maps
    * N-D                 -> first batch element flattened to (-1, last)
    """
    max_r, max_c = cfg.heatmap_size
    if x.ndim == 0:
        return None
    if x.ndim == 1:
        img, layout = x[None, :], "vector"
    elif x.ndim == 2:
        img, layout = x, "matrix"
    elif x.ndim == 3:
        img, layout = x[0], "sequence"
    elif x.ndim == 4 and x.shape[2] > 1 and x.shape[3] > 1:
        fmap = x[0]  # [C, H, W]
        k = min(fmap.shape[0], cfg.mosaic_channels)
        ncols = math.ceil(math.sqrt(k))
        nrows = math.ceil(k / ncols)
        side = max(1, min(max_r // nrows, max_c // ncols))
        th, tw = min(fmap.shape[1], side), min(fmap.shape[2], side)
        tiles = F.adaptive_avg_pool2d(fmap[:k][None], (th, tw))[0]  # [k, th, tw]
        canvas = torch.full((nrows * th, ncols * tw), float("nan"), device=x.device)
        for i in range(k):
            r, c = divmod(i, ncols)
            canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tiles[i]
        out = _quantise(canvas)
        out.update(layout="mosaic", tile=[th, tw], grid=[nrows, ncols], channels=k)
        return out
    else:
        img, layout = x[0].reshape(-1, x.shape[-1]), "flattened"
    # Pool with non-finite values zeroed so a single NaN doesn't wipe a region,
    # then re-mark pooled cells that contained any non-finite value.
    finite = torch.isfinite(img)
    pooled = _pool2d(torch.where(finite, img, torch.zeros_like(img)), max_r, max_c)
    if not finite.all():
        bad = _pool2d((~finite).float(), max_r, max_c) > 0
        pooled = torch.where(bad, torch.full_like(pooled, float("nan")), pooled)
    out = _quantise(pooled)
    out.update(layout=layout, src=[int(img.shape[0]), int(img.shape[1])])
    return out


def _channel_profile(x: torch.Tensor, cfg: StatsConfig) -> dict[str, Any] | None:
    """Mean |x| per channel, where the channel axis is dim 1 for conv-shaped
    tensors ([B, C, H, W]) and the last dim otherwise (features / hidden size).

    Long profiles are max-pooled so outlier channels survive downsampling.
    """
    if x.ndim < 2:
        return None
    if x.ndim == 4:
        prof = x.abs().mean(dim=(0, 2, 3))
        axis = 1
    else:
        prof = x.reshape(-1, x.shape[-1]).abs().mean(dim=0)
        axis = x.ndim - 1
    n = prof.numel()
    if n < 2:
        return None
    prof = torch.nan_to_num(prof, nan=0.0, posinf=float(torch.finfo(prof.dtype).max))
    med = prof.median().item()
    top = torch.topk(prof, k=min(5, n))
    pooled = prof
    if n > cfg.channel_bins:
        pooled = F.adaptive_max_pool1d(prof[None, None], cfg.channel_bins)[0, 0]
    return {
        "axis": axis,
        "n": n,
        "values": [_num(v) for v in pooled.tolist()],
        "median": _num(med),
        "top": [[int(i), _num(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
    }


@torch.no_grad()
def tensor_stats(t: torch.Tensor, cfg: StatsConfig, *, detail: bool = True) -> dict[str, Any]:
    """Summarise a tensor. `detail=False` skips heatmap and channel profile."""
    kind = _kind(t)
    info: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "kind": kind,
        "numel": t.numel(),
    }
    if t.numel() == 0:
        return info
    t = t.detach()
    if kind == "complex":
        t = t.abs()
    x = t.float()
    flat = x.reshape(-1)

    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    finite = ~(nan_mask | inf_mask)
    n_finite_t = finite.sum()
    xf = torch.where(finite, flat, torch.zeros_like(flat))
    denom = n_finite_t.clamp(min=1).float()
    mean = xf.sum() / denom
    sq_mean = (xf * xf).sum() / denom
    big = torch.finfo(torch.float32).max
    vmin = torch.where(finite, flat, torch.full_like(flat, big)).min()
    vmax = torch.where(finite, flat, torch.full_like(flat, -big)).max()

    packed = torch.stack(
        [
            nan_mask.sum().float(),
            inf_mask.sum().float(),
            n_finite_t.float(),
            mean,
            sq_mean,
            xf.abs().sum() / denom,
            vmin,
            vmax,
            ((flat == 0) & finite).sum().float(),
        ]
    ).tolist()  # single device->host transfer
    n_nan, n_inf, n_finite, mean_v, sq_v, absmean_v, min_v, max_v, n_zero = packed

    info["nan"] = int(n_nan)
    info["inf"] = int(n_inf)
    if n_finite == 0:
        return info

    var = max(sq_v - mean_v * mean_v, 0.0)
    info.update(
        mean=_num(mean_v),
        std=_num(math.sqrt(var)),
        rms=_num(math.sqrt(max(sq_v, 0.0))),
        abs_mean=_num(absmean_v),
        min=_num(min_v),
        max=_num(max_v),
        zero_frac=_num(n_zero / n_finite),
    )

    # Quantiles on a strided sample of finite values.
    fin_vals = flat[finite] if n_finite < flat.numel() else flat
    step = max(1, fin_vals.numel() // cfg.sample_size)
    sample = fin_vals[::step]
    qs = torch.quantile(sample, torch.tensor(QUANTILES, device=sample.device)).tolist()
    info["q"] = [_num(v) for v in qs]

    # Histogram over the finite range.
    lo, hi = min_v, max_v
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    if kind in ("int", "bool") and hi - lo + 1 <= cfg.hist_bins:
        lo, hi = lo - 0.5, hi + 0.5  # integer-aligned bins
        bins = int(round(hi - lo))
    else:
        bins = cfg.hist_bins
    counts = torch.histc(fin_vals, bins=bins, min=lo, max=hi)
    info["hist"] = {"lo": _num(lo), "hi": _num(hi), "counts": [int(c) for c in counts.tolist()]}

    if detail and kind in ("float", "complex"):
        if cfg.heatmap:
            hm = _heatmap(x, cfg)
            if hm is not None:
                info["heatmap"] = hm
        prof = _channel_profile(x, cfg)
        if prof is not None:
            info["channels"] = prof
    return info
