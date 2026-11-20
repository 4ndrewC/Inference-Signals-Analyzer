"""JEPA / world-model analysis on top of the signal tracer.

A JEPA has a context encoder, a target encoder (often an EMA copy) and a
predictor that maps context embeddings (plus actions / mask tokens) to target
embeddings. ``analyze_jepa`` runs one analysis batch through user code and
records, in addition to the usual per-module signal trace:

* embedding health per role: per-dimension std, variance spectrum, effective
  rank, low-variance dimensions, correlation, plus a collapse verdict;
* prediction quality: R², cosine, MSE, a copy-context baseline, and
  error by rollout horizon;
* module influence: mean-ablation of each module (Δloss and the drift of
  predictions / embeddings) and gradient × activation attribution of the loss;
* input influence: the same measures when each input (e.g. the action) is
  shuffled across the batch;
* a PCA "landscape" of context / target / predicted embeddings.
"""

from __future__ import annotations

import contextlib
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .stats import StatsConfig, _num, _quantise, tensor_stats
from .tracer import SignalTracer

ROLE_LABELS = {"context": "Context encoder", "target": "Target encoder", "predictor": "Predictor"}


# --------------------------------------------------------------------------- user-facing types
@dataclass
class JEPAOutput:
    """What a JEPA step function returns.

    ``target`` and ``prediction`` are aligned tensors (``[B, D]`` or
    ``[B, N, D]``), or equal-length lists of them, one per rollout horizon.
    ``loss`` defaults to the mean MSE between prediction and target.
    """

    context: torch.Tensor
    target: torch.Tensor | Sequence[torch.Tensor]
    prediction: torch.Tensor | Sequence[torch.Tensor]
    loss: torch.Tensor | None = None


StepFn = Callable[[nn.Module, Any], Any]


def _as_output(out: Any) -> JEPAOutput:
    if isinstance(out, JEPAOutput):
        return out
    if isinstance(out, Mapping):
        pred = out.get("prediction", out.get("predictions", out.get("pred")))
        tgt = out.get("target", out.get("targets"))
        return JEPAOutput(context=out["context"], target=tgt, prediction=pred, loss=out.get("loss"))
    if isinstance(out, (tuple, list)) and len(out) in (3, 4):
        return JEPAOutput(*out)
    raise TypeError("step_fn must return JEPAOutput, a dict with context/target/prediction, or a (context, target, prediction[, loss]) tuple")


def _as_list(x: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [x] if isinstance(x, torch.Tensor) else list(x)


def default_loss(out: JEPAOutput) -> torch.Tensor:
    preds, tgts = _as_list(out.prediction), _as_list(out.target)
    return torch.stack([F.mse_loss(p.float(), t.float()) for p, t in zip(preds, tgts)]).mean()


def _loss_of(out: JEPAOutput) -> torch.Tensor:
    return out.loss if out.loss is not None else default_loss(out)


# --------------------------------------------------------------------------- helpers
def _rows(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Flatten an embedding tensor to [rows, D] plus (batch index, token index) per row."""
    t = t.detach().float().cpu()
    if t.ndim == 1:
        t = t[None]
    d = t.shape[-1]
    n = t.reshape(-1, d).shape[0]
    idx = torch.arange(n)
    if t.ndim == 2:
        return t, idx, None
    per = math.prod(t.shape[1:-1])
    return t.reshape(-1, d), idx // per, idx % per


@contextlib.contextmanager
def _seeded(seed: int | None):
    """Run a block with fixed RNG state (so random masks match across runs) without
    disturbing the caller's RNG."""
    if seed is None:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() and torch.cuda.is_initialized() else []
    py_state = random.getstate()
    try:
        import numpy as np

        np_state = np.random.get_state()
    except ImportError:  # pragma: no cover
        np, np_state = None, None
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        random.seed(seed)
        if np is not None:
            np.random.seed(seed % 2**32)
        try:
            yield
        finally:
            random.setstate(py_state)
            if np is not None:
                np.random.set_state(np_state)


def _map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_map_tensors(v, fn) for v in obj))
    if isinstance(obj, (tuple, list)):
        return type(obj)(_map_tensors(v, fn) for v in obj)
    if type(obj) is dict:
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, dict):  # dict subclasses such as HF ModelOutput: update in place
        for k in list(obj.keys()):
            obj[k] = _map_tensors(obj[k], fn)
        return obj
    return obj


def _named_inputs(batch: Any) -> list[tuple[str, Any]]:
    if isinstance(batch, Mapping):
        return [(str(k), k) for k in batch.keys()]
    if isinstance(batch, (tuple, list)):
        return [(f"arg{i}", i) for i in range(len(batch))]
    if isinstance(batch, torch.Tensor):
        return [("input", None)]
    return []


def _replace_input(batch: Any, key: Any, value: torch.Tensor) -> Any:
    if key is None:
        return value
    if isinstance(batch, Mapping):
        new = dict(batch)
        new[key] = value
        return new
    new = list(batch)
    new[key] = value
    return type(batch)(new) if isinstance(batch, tuple) else new


def _get_input(batch: Any, key: Any) -> Any:
    return batch if key is None else batch[key]


def _drift(new: torch.Tensor, ref: torch.Tensor) -> float | None:
    """||new - ref|| relative to the spread of ref around its mean (1.0 = sample information destroyed)."""
    if new.shape != ref.shape:
        return None
    spread = (ref - ref.mean(0, keepdim=True)).norm()
    if spread <= 0:
        spread = ref.norm()
    if spread <= 0:
        return None
    return (new - ref).norm().item() / spread.item()


def _cat_rows(ts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([_rows(t)[0] for t in ts], 0)


# --------------------------------------------------------------------------- embedding health
@torch.no_grad()
def embedding_stats(
    z: torch.Tensor, *, low_var_rel: float = 0.01, max_rows: int = 8192, max_dims: int = 4096, seed: int = 0
) -> dict[str, Any]:
    """Collapse-oriented statistics of a [rows, D] embedding matrix."""
    z = z.detach().float().cpu()
    n, d = z.shape
    if n > max_rows:
        g = torch.Generator().manual_seed(seed)
        z = z[torch.randperm(n, generator=g)[:max_rows]]
    m = z.shape[0]
    mu = z.mean(0)
    zc = z - mu
    std = zc.pow(2).mean(0).sqrt()
    rms = z.pow(2).mean().sqrt().item()
    max_std = std.max().item()
    thresh = max(low_var_rel * max_std, 1e-8)
    low = int((std < thresh).sum())

    lam = torch.linalg.svdvals(zc).pow(2) / max(1, m - 1) if m > 1 else torch.zeros(1)
    tot = lam.sum().item()
    if tot > 0:
        p = lam / lam.sum()
        p = p[p > 0]
        erank = math.exp(-(p * p.log()).sum().item())
        pr = tot**2 / lam.pow(2).sum().item()
        top1 = lam[0].item() / tot
    else:
        erank, pr, top1 = 0.0, 0.0, 1.0
    max_rank = max(1, min(d, m - 1))

    # Pairwise cosine similarity (raw, so a shared offset also shows up as "everything similar").
    k = min(m, 512)
    zn = F.normalize(z[:k], dim=1)
    cos = (zn @ zn.T)
    cos_mean = ((cos.sum() - cos.diagonal().sum()) / max(1, k * (k - 1))).item() if k > 1 else None

    corr_info: dict[str, Any] = {}
    live = std > 1e-12
    if m > 2 and live.sum() > 1:
        zs = zc[:, live] / std[live]
        corr = (zs.T @ zs) / m
        dl = corr.shape[0]
        off = (corr.abs().sum() - corr.diagonal().abs().sum()) / max(1, dl * (dl - 1))
        corr_info["corr_abs_mean"] = _num(off.item())
        full = torch.zeros(d, d)
        idx = live.nonzero().squeeze(1)
        full[idx[:, None], idx[None, :]] = corr.abs()
        side = min(d, 64)
        pooled = F.adaptive_avg_pool2d(full[None, None], (side, side))[0, 0] if d > side else full
        corr_info["corr"] = _quantise(pooled)

    # Share of embedding energy that is sample-specific: ~1 healthy, ->0 when every
    # sample maps to (nearly) the same vector, however many directions the residue spans.
    energy = z.pow(2).sum(1).mean().item()
    var_frac = tot / energy if energy > 0 else 0.0
    if d >= 2 and (var_frac < 1e-3 or max_std <= 1e-8 or (tot > 0 and erank < 1.05)):
        status, why = "collapsed", f"only {var_frac:.2%} of embedding energy varies across samples"
    elif d >= 8 and (erank < 0.1 * max_rank or low > 0.5 * d):
        status, why = "dimensional", f"variance lives in ~{erank:.1f} of {max_rank} possible directions"
    elif var_frac < 0.02:
        status, why = "offset", f"a shared offset dominates: only {var_frac:.1%} of energy varies across samples"
    else:
        status, why = "healthy", f"{var_frac:.0%} of energy varies across samples, spread over ~{erank:.1f} of {max_rank} directions"

    std_list = std[:max_dims].tolist()
    return {
        "rows": n,
        "dim": d,
        "max_rank": max_rank,
        "std": [_num(v) for v in std_list],
        "mean_std": _num(std.mean().item()),
        "min_std": _num(std.min().item()),
        "max_std": _num(max_std),
        "low_var_dims": low,
        "low_var_thresh": _num(thresh),
        "total_var": _num(tot),
        "var_frac": _num(var_frac),
        "erank": _num(erank),
        "participation": _num(pr),
        "top1_frac": _num(top1),
        "spectrum": [_num(v) for v in (lam / tot if tot > 0 else lam)[:256].tolist()],
        "rms": _num(rms),
        "mean_norm": _num(z.norm(dim=1).mean().item()),
        "offset_norm": _num(mu.norm().item()),
        "cos_mean": _num(cos_mean) if cos_mean is not None else None,
        "status": status,
        "status_reason": why,
        **corr_info,
    }


# --------------------------------------------------------------------------- prediction quality
@torch.no_grad()
def prediction_stats(pred: torch.Tensor, tgt: torch.Tensor, ctx: torch.Tensor | None = None) -> dict[str, Any]:
    p, t = _rows(pred)[0], _rows(tgt)[0]
    if p.shape != t.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(tgt.shape)} must have the same shape")
    sse = (p - t).pow(2).sum().item()
    sst = (t - t.mean(0)).pow(2).sum().item()
    # R² is undefined when the target has no variance; judge "no variance" relative to the
    # target's energy so float round-off on a constant target doesn't give R² = -1e13.
    has_var = sst > 1e-9 * t.pow(2).sum().item() and sst > 1e-30
    out: dict[str, Any] = {
        "mse": _num((p - t).pow(2).mean().item()),
        "r2": _num(1 - sse / sst) if has_var else None,
        "cos": _num(F.cosine_similarity(p, t, dim=1).mean().item()),
        "target_var": _num(sst / max(1, t.shape[0])),
        "pred_var": _num((p - p.mean(0)).pow(2).sum().item() / max(1, p.shape[0])),
    }
    if ctx is not None:
        c = _rows(ctx)[0]
        if c.shape == t.shape:
            out["copy_r2"] = _num(1 - (c - t).pow(2).sum().item() / sst) if has_var else None
            out["copy_cos"] = _num(F.cosine_similarity(c, t, dim=1).mean().item())
            out["copy_mse"] = _num((c - t).pow(2).mean().item())
    return out


# --------------------------------------------------------------------------- landscape
@torch.no_grad()
def landscape(
    ctx: torch.Tensor, tgts: list[torch.Tensor], preds: list[torch.Tensor], *, max_points: int = 900, n_pcs: int = 6, seed: int = 0
) -> dict[str, Any]:
    """Project sampled context / target / prediction rows onto a shared PCA basis."""
    g = torch.Generator().manual_seed(seed)
    h = len(tgts)
    roles: dict[str, dict[str, Any]] = {}

    def pick(n: int, k: int) -> torch.Tensor:
        return torch.randperm(n, generator=g)[:k].sort().values if n > k else torch.arange(n)

    c_rows, c_b, c_t = _rows(ctx)
    ci = pick(c_rows.shape[0], max_points)
    sel = {"context": [(c_rows[ci], c_b[ci], c_t[ci] if c_t is not None else None, None, None)]}
    per_h = max(1, max_points // h)
    sel["target"], sel["prediction"] = [], []
    for hi, (tg, pr) in enumerate(zip(tgts, preds)):
        t_rows, t_b, t_t = _rows(tg)
        p_rows = _rows(pr)[0]
        ii = pick(t_rows.shape[0], per_h)
        err = (p_rows[ii] - t_rows[ii]).pow(2).mean(1)
        tok = t_t[ii] if t_t is not None else None
        sel["target"].append((t_rows[ii], t_b[ii], tok, hi, err))
        sel["prediction"].append((p_rows[ii], t_b[ii], tok, hi, err))

    d = sel["target"][0][0].shape[1]
    fit_roles = [r for r in ("context", "target", "prediction") if sel[r][0][0].shape[1] == d]
    fit = torch.cat([x[0] for r in fit_roles for x in sel[r]], 0)
    mean = fit.mean(0)
    _, s, vh = torch.linalg.svd(fit - mean, full_matrices=False)
    k = min(n_pcs, vh.shape[0])
    comps = vh[:k]
    var = s.pow(2)
    explained = (var / var.sum()).tolist() if var.sum() > 0 else [0.0] * len(var)

    for role, parts in sel.items():
        rows = torch.cat([x[0] for x in parts], 0)
        if rows.shape[1] == d:
            proj, basis = (rows - mean) @ comps.T, "joint"
        else:  # context with a different width: its own PCA
            m2 = rows.mean(0)
            _, _, v2 = torch.linalg.svd(rows - m2, full_matrices=False)
            proj, basis = (rows - m2) @ v2[:k].T, "own"
            if proj.shape[1] < k:
                proj = F.pad(proj, (0, k - proj.shape[1]))
        rec: dict[str, Any] = {
            "n": int(rows.shape[0]),
            "basis": basis,
            "xy": [_num(v) for v in proj.reshape(-1).tolist()],
            "b": torch.cat([x[1] for x in parts]).tolist(),
        }
        if parts[0][2] is not None:
            rec["tok"] = torch.cat([x[2] for x in parts]).tolist()
        if parts[0][3] is not None:
            rec["h"] = [x[3] for x in parts for _ in range(x[0].shape[0])]
        if parts[0][4] is not None:
            rec["err"] = [_num(v) for v in torch.cat([x[4] for x in parts]).tolist()]
        roles[role] = rec
    return {"pcs": k, "explained": [_num(v) for v in explained[:k]], "dim": d, "roles": roles, "paired": True}


# --------------------------------------------------------------------------- influence
def _ablate_tensor(t: torch.Tensor, mode: str) -> torch.Tensor:
    if not t.is_floating_point() or t.ndim == 0:
        return t
    if mode == "zero":
        return torch.zeros_like(t)
    if t.shape[0] > 1:
        return t.mean(0, keepdim=True).expand_as(t).clone()
    # batch of one: fall back to the mean over every axis but the feature axis
    dims = tuple(range(t.ndim - 1))
    return t.mean(dim=dims, keepdim=True).expand_as(t).clone() if dims else torch.zeros_like(t)


@dataclass
class _Run:
    loss: float
    ctx: torch.Tensor
    tgt: torch.Tensor
    pred: torch.Tensor


def _run(model: nn.Module, step_fn: StepFn, batch: Any, seed: int | None) -> _Run:
    with torch.no_grad(), _seeded(seed):
        out = _as_output(step_fn(model, batch))
        loss = _loss_of(out).float().item()
    return _Run(loss, _rows(out.context)[0], _cat_rows(_as_list(out.target)), _cat_rows(_as_list(out.prediction)))


def _effect(run: _Run, clean: _Run) -> dict[str, Any]:
    rel = (run.loss - clean.loss) / clean.loss if clean.loss > 1e-12 else None
    return {
        "loss": _num(run.loss),
        "d_loss": _num(rel) if rel is not None else None,
        "d_loss_abs": _num(run.loss - clean.loss),
        "pred_drift": _num(_drift(run.pred, clean.pred)),
        "ctx_drift": _num(_drift(run.ctx, clean.ctx)),
        "tgt_drift": _num(_drift(run.tgt, clean.tgt)),
    }


def _attribution(
    model: nn.Module, step_fn: StepFn, batch: Any, module_ids: dict[int, int], seed: int | None
) -> tuple[dict[int, dict[str, float]], dict[str, float], str | None]:
    """Gradient x activation of the JEPA loss for every module output and float input."""
    params = [(p, p.requires_grad) for p in model.parameters()]
    store: dict[int, list[torch.Tensor]] = {}
    handles = []

    def track(lst: list[torch.Tensor]):
        def fn(t: torch.Tensor) -> torch.Tensor:
            # Outputs produced under no_grad (e.g. a stop-grad target branch) have no
            # gradient path to the loss; leave them alone.
            if not t.is_floating_point() or not torch.is_grad_enabled():
                return t
            if not t.requires_grad:  # no graph upstream: start one here
                t = t.detach().requires_grad_().clone()
            t.retain_grad()
            lst.append(t)
            return t

        return fn

    for module in model.modules():
        mid = module_ids.get(id(module))
        if mid is None or mid == 0:
            continue
        lst = store.setdefault(mid, [])
        handles.append(module.register_forward_hook(lambda m, a, o, _f=track(lst): _map_tensors(o, _f)))

    inputs: dict[str, torch.Tensor] = {}
    gbatch = batch
    for name, key in _named_inputs(batch):
        v = _get_input(batch, key)
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            leaf = v.detach().requires_grad_()
            inputs[name] = leaf
            gbatch = _replace_input(gbatch, key, leaf)
    error = None
    try:
        for p, _ in params:
            p.requires_grad_(False)
        with torch.enable_grad(), _seeded(seed):
            out = _as_output(step_fn(model, gbatch))
            loss = _loss_of(out)
            if loss.requires_grad:
                loss.backward()
            else:
                error = "loss has no gradient path (computed entirely under no_grad?)"
    except RuntimeError as exc:  # e.g. in-place ops on tracked tensors
        error = f"attribution failed: {exc}"
    finally:
        for h in handles:
            h.remove()
        for p, req in params:
            p.requires_grad_(req)

    mods: dict[int, dict[str, float]] = {}
    for mid, lst in store.items():
        grads = [(t, t.grad) for t in lst if t.grad is not None]
        if not grads:
            continue
        attr = sum((g * t.detach()).abs().sum().item() for t, g in grads)
        gn = math.sqrt(sum(g.pow(2).sum().item() for _, g in grads))
        mods[mid] = {"attr": attr, "grad_norm": gn}
    ins = {n: (t.grad * t.detach()).abs().sum().item() for n, t in inputs.items() if t.grad is not None}
    return mods, ins, error


# --------------------------------------------------------------------------- roles
def _guess_roles(model: nn.Module) -> dict[str, nn.Module]:
    roles: dict[str, nn.Module] = {}
    for name, child in model.named_children():
        low = name.lower()
        if "target" in low or "ema" in low or "teacher" in low:
            roles.setdefault("target", child)
        elif "pred" in low:
            roles.setdefault("predictor", child)
        elif "context" in low or "online" in low or "student" in low or "encoder" in low:
            roles.setdefault("context", child)
    return roles


# --------------------------------------------------------------------------- report
@dataclass
class JEPAReport:
    trace: dict[str, Any]
    tracer: SignalTracer = field(repr=False)

    @property
    def jepa(self) -> dict[str, Any]:
        return self.trace["jepa"]

    def save(self, path: str | Path) -> Path:
        import gzip
        import json

        path = Path(path)
        data = json.dumps(self.trace, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
        return path

    def save_html(self, path: str | Path) -> Path:
        from .viewer import render_html

        path = Path(path)
        path.write_text(render_html(self.trace), encoding="utf-8")
        return path

    def summary(self) -> str:
        j = self.jepa
        lines = [f"JEPA analysis of {self.trace['meta']['name']}"]
        for role, st in j["embeddings"].items():
            lines.append(
                f"  {role:<10} dim={st['dim']:<5} eff.rank={st['erank']:<7} low-var dims={st['low_var_dims']:<4} {st['status'].upper()}: {st['status_reason']}"
            )
        ov = j["prediction"]["overall"]
        base = f"  (copy-context baseline R²={ov['copy_r2']})" if ov.get("copy_r2") is not None else ""
        lines.append(f"  prediction R²={ov['r2']} cos={ov['cos']} mse={ov['mse']}{base}")
        inf = j.get("influence") or {}
        mods = sorted((m for m in inf.get("modules", []) if m.get("d_loss") is not None), key=lambda m: -abs(m["d_loss"]))[:5]
        if mods:
            names = {m["id"]: m["path"] for m in self.trace["modules"]}
            lines.append("  most influential modules (Δloss when mean-ablated):")
            lines += [f"    {names[m['module']] or '<root>':<40} {m['d_loss']:+.3g}" for m in mods]
        for i in inf.get("inputs", []):
            if i.get("d_loss") is not None:
                lines.append(f"  input '{i['name']}' shuffled: Δloss {i['d_loss']:+.3g}, prediction drift {i['pred_drift']}")
        return "\n".join(lines)


def analyze_jepa(
    model: nn.Module,
    step_fn: StepFn,
    batch: Any,
    *,
    roles: Mapping[str, nn.Module] | None = None,
    ablate: bool = True,
    ablation_mode: str = "mean",
    max_ablations: int = 400,
    ablate_filter: Callable[[str, nn.Module], bool] | None = None,
    attribution: bool = True,
    input_influence: bool = True,
    max_points: int = 900,
    n_pcs: int = 6,
    low_var_rel: float = 0.01,
    seed: int | None = 0,
    name: str | None = None,
    **tracer_kwargs: Any,
) -> JEPAReport:
    """Analyse a JEPA / world model on one batch.

    Args:
        model: module containing the encoders and predictor (used for tracing,
            role assignment and ablations).
        step_fn: ``step_fn(model, batch)`` runs one JEPA forward and returns a
            :class:`JEPAOutput` (or dict / tuple with context, target, prediction).
        batch: analysis batch; mean-ablation needs a batch size > 1.
        roles: ``{"context": m, "target": m, "predictor": m}``; guessed from child
            names (``context*``/``encoder``, ``target*``/``ema*``, ``pred*``) if omitted.
        ablation_mode: ``"mean"`` (replace with batch mean) or ``"zero"``.
        ablate_filter: ``(path, module) -> bool`` restricting which modules are ablated.
        seed: RNG seed applied before every run so random masking is identical.
        **tracer_kwargs: forwarded to :class:`SignalTracer` (include/exclude, ...).
    """
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        tracer = SignalTracer(model, name=name, **tracer_kwargs)
        with tracer, torch.no_grad(), _seeded(seed):
            tracer.step("analysis batch")
            out = _as_output(step_fn(model, batch))
            clean_loss = _loss_of(out).float().item()
        trace = tracer.trace

        # ---- roles
        role_mods = dict(roles) if roles is not None else _guess_roles(model)
        module_ids = tracer._module_ids
        role_of: dict[int, str] = {}
        for role in ("predictor", "target", "context"):  # later roles win for shared modules
            rm = role_mods.get(role)
            if rm is None:
                continue
            for sub in rm.modules():
                mid = module_ids.get(id(sub))
                if mid is not None:
                    role_of[mid] = f"{role_of[mid]}+{role}" if mid in role_of and role_of[mid] != role else role
        for m in trace["modules"]:
            if m["id"] in role_of:
                m["role"] = role_of[m["id"]]

        # ---- embeddings & prediction quality
        tgts, preds = _as_list(out.target), _as_list(out.prediction)
        if len(tgts) != len(preds):
            raise ValueError("target and prediction lists must have the same length (one per horizon)")
        embeds = {
            "context": embedding_stats(_rows(out.context)[0], low_var_rel=low_var_rel),
            "target": embedding_stats(_cat_rows(tgts), low_var_rel=low_var_rel),
            "prediction": embedding_stats(_cat_rows(preds), low_var_rel=low_var_rel),
        }
        horizons = [prediction_stats(p, t, out.context) for p, t in zip(preds, tgts)]
        p_all, t_all, c_rows = _cat_rows(preds), _cat_rows(tgts), _rows(out.context)[0]
        c_all = torch.cat([c_rows] * len(tgts)) if c_rows.shape == _rows(tgts[0])[0].shape else None
        overall = prediction_stats(p_all, t_all, c_all)
        row_err = (p_all - t_all).pow(2).mean(1)
        err_hist = tensor_stats(row_err, StatsConfig(hist_bins=40, heatmap=False), detail=False)

        jepa: dict[str, Any] = {
            "roles": {r: {"label": ROLE_LABELS.get(r, r), "module": module_ids.get(id(m))} for r, m in role_mods.items()},
            "loss": _num(clean_loss),
            "embeddings": embeds,
            "prediction": {"overall": overall, "horizons": horizons, "err_hist": err_hist},
            "landscape": landscape(out.context, tgts, preds, max_points=max_points, n_pcs=n_pcs, seed=seed or 0),
            "config": {"ablation_mode": ablation_mode, "low_var_rel": low_var_rel, "seed": seed},
        }

        # ---- influence
        influence: dict[str, Any] = {"mode": ablation_mode, "loss_clean": _num(clean_loss), "modules": [], "inputs": []}
        attr_mods, attr_inputs, attr_err = ({}, {}, None)
        if attribution:
            attr_mods, attr_inputs, attr_err = _attribution(model, step_fn, batch, module_ids, seed)
            if attr_err:
                influence["attribution_error"] = attr_err
        if ablate or attribution:
            clean = _run(model, step_fn, batch, seed)
            called = {e["module"] for e in trace["events"]}
            by_id = {mid: mod for mod in model.modules() if (mid := module_ids.get(id(mod))) is not None}
            cands = [m for m in trace["modules"] if m["id"] != 0 and m["id"] in called]
            if ablate_filter is not None:
                cands = [m for m in cands if ablate_filter(m["path"], by_id[m["id"]])]
            cands.sort(key=lambda m: (m["depth"], m["id"]))
            ablated = {m["id"] for m in cands[:max_ablations]} if ablate else set()
            for m in sorted(cands, key=lambda m: m["id"]):
                rec: dict[str, Any] = {"module": m["id"]}
                if m["id"] in ablated:
                    h = by_id[m["id"]].register_forward_hook(lambda mod, a, o: _map_tensors(o, lambda t: _ablate_tensor(t, ablation_mode)))
                    try:
                        rec.update(_effect(_run(model, step_fn, batch, seed), clean))
                    except Exception as exc:  # an ablation can break shape assumptions downstream
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    finally:
                        h.remove()
                if m["id"] in attr_mods:
                    rec["attr"] = _num(attr_mods[m["id"]]["attr"])
                    rec["grad_norm"] = _num(attr_mods[m["id"]]["grad_norm"])
                if len(rec) > 1:
                    influence["modules"].append(rec)
            if input_influence:
                g = torch.Generator().manual_seed(seed or 0)
                for nm, key in _named_inputs(batch):
                    v = _get_input(batch, key)
                    if not isinstance(v, torch.Tensor) or v.ndim == 0 or v.shape[0] < 2:
                        continue
                    perm = torch.randperm(v.shape[0], generator=g)
                    if torch.equal(perm, torch.arange(v.shape[0])):
                        perm = perm.roll(1)
                    rec = {"name": nm, "shape": list(v.shape)}
                    try:
                        rec.update(_effect(_run(model, step_fn, _replace_input(batch, key, v[perm.to(v.device)]), seed), clean))
                    except Exception as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    if nm in attr_inputs:
                        rec["attr"] = _num(attr_inputs[nm])
                    influence["inputs"].append(rec)
        jepa["influence"] = influence
        jepa["analysis_s"] = round(time.perf_counter() - t0, 3)
        trace["jepa"] = jepa
        return JEPAReport(trace=trace, tracer=tracer)
    finally:
        model.train(was_training)
