"""Forward-hook based signal tracer for torch.nn.Module inference."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import fnmatch
import gzip
import inspect
import json
import platform
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .stats import StatsConfig, tensor_stats

TRACE_VERSION = 1

ModuleFilter = Callable[[str, nn.Module], bool] | str | Iterable[str] | None


def _flatten(obj: Any, prefix: str, out: list[tuple[str, torch.Tensor]], limit: int, depth: int = 0) -> int:
    """Collect tensors from nested containers as (path, tensor). Returns how many were dropped."""
    if isinstance(obj, torch.Tensor):
        if len(out) < limit:
            out.append((prefix, obj))
            return 0
        return 1
    if depth > 4 or obj is None:
        return 0
    dropped = 0
    if isinstance(obj, Mapping):
        items = [(f"{prefix}.{k}", v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple)):
        items = [(f"{prefix}[{i}]", v) for i, v in enumerate(obj)]
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [(f"{prefix}.{f.name}", getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    else:
        return 0
    for name, v in items:
        dropped += _flatten(v, name, out, limit, depth + 1)
    return dropped


def _compile_filter(spec: ModuleFilter) -> Callable[[str, nn.Module], bool] | None:
    """Patterns match the dotted module path *or* the class name, fnmatch-style."""
    if spec is None:
        return None
    if callable(spec):
        return spec
    patterns = [spec] if isinstance(spec, str) else list(spec)

    def match(path: str, module: nn.Module) -> bool:
        tname = type(module).__name__
        return any(fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(tname, p) for p in patterns)

    return match


class SignalTracer:
    """Record activation statistics for every module call during inference.

    Usage::

        with SignalTracer(model) as tracer:
            model(x)
        tracer.save_html("trace.html")

    Each call to the root module starts a new *step* (e.g. one token of
    autoregressive generation). Call :meth:`step` yourself to group calls
    differently; after the first manual call automatic stepping is disabled.

    Args:
        model: module to trace.
        include / exclude: fnmatch patterns on module path or class name (or a
            ``(path, module) -> bool`` callable) selecting which modules get hooks.
        leaves_only: only hook modules without children (plus the root).
        max_depth: only hook modules at most this deep (root = 0).
        capture_inputs: record stats for module inputs as well as outputs.
        capture_params: record per-parameter stats once at attach time.
        input_detail: also compute heatmaps / channel profiles for inputs.
        max_tensors: max tensors recorded per input/output structure.
        max_steps: stop recording after this many steps.
        sync_cuda: synchronise CUDA around each module for accurate timings.
        hist_bins, heatmap, heatmap_size, channel_bins, sample_size: see StatsConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        include: ModuleFilter = None,
        exclude: ModuleFilter = None,
        leaves_only: bool = False,
        max_depth: int | None = None,
        capture_inputs: bool = True,
        capture_params: bool = True,
        input_detail: bool = False,
        max_tensors: int = 8,
        max_steps: int | None = None,
        sync_cuda: bool = False,
        name: str | None = None,
        hist_bins: int = 64,
        heatmap: bool = True,
        heatmap_size: tuple[int, int] = (64, 128),
        channel_bins: int = 256,
        sample_size: int = 65536,
    ) -> None:
        self.model = model
        self.name = name or type(model).__name__
        self.include = _compile_filter(include)
        self.exclude = _compile_filter(exclude)
        self.leaves_only = leaves_only
        self.max_depth = max_depth
        self.capture_inputs = capture_inputs
        self.capture_params = capture_params
        self.input_detail = input_detail
        self.max_tensors = max_tensors
        self.max_steps = max_steps
        self.sync_cuda = sync_cuda
        self.cfg = StatsConfig(
            hist_bins=hist_bins,
            heatmap=heatmap,
            heatmap_size=tuple(heatmap_size),
            channel_bins=channel_bins,
            sample_size=sample_size,
        )

        self._handles: list[Any] = []
        self._module_ids: dict[int, int] = {}
        self.modules: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self._stack: list[tuple[int, float, float]] = []  # (event index, t0, overhead0)
        self._overhead = 0.0
        self._auto_steps = True
        self._call_counts: dict[int, int] = {}
        self._root_depth = 0
        self._order = 0
        self._started = time.time()
        self._build_module_table()

    # ------------------------------------------------------------------ setup

    def _build_module_table(self) -> None:
        path_to_id: dict[str, int] = {}
        for path, module in self.model.named_modules():
            if id(module) in self._module_ids:  # shared module registered under first path
                continue
            depth = 0 if path == "" else path.count(".") + 1
            parent_path = path.rsplit(".", 1)[0] if "." in path else ("" if path else None)
            is_leaf = next(iter(module.children()), None) is None
            own = sum(p.numel() for p in module.parameters(recurse=False))
            total = sum(p.numel() for p in module.parameters())
            try:
                extra = module.extra_repr()
            except Exception:  # pragma: no cover - user modules can do anything
                extra = ""
            mid = len(self.modules)
            rec: dict[str, Any] = {
                "id": mid,
                "path": path,
                "name": path.rsplit(".", 1)[-1] if path else self.name,
                "type": type(module).__name__,
                "parent": path_to_id.get(parent_path) if parent_path is not None else None,
                "depth": depth,
                "leaf": is_leaf,
                "extra": extra[:300],
                "n_params": own,
                "n_params_total": total,
                "traced": self._should_trace(path, module, depth, is_leaf),
            }
            if self.capture_params and own:
                rec["params"] = [
                    {"name": pname, **tensor_stats(p, self.cfg, detail=False)}
                    for pname, p in module.named_parameters(recurse=False)
                ]
                bufs = list(module.named_buffers(recurse=False))
                if bufs:
                    rec["buffers"] = [{"name": b, **tensor_stats(t, self.cfg, detail=False)} for b, t in bufs]
            self.modules.append(rec)
            self._module_ids[id(module)] = mid
            path_to_id[path] = mid

    def _should_trace(self, path: str, module: nn.Module, depth: int, is_leaf: bool) -> bool:
        if path == "":
            return True  # the root always defines steps
        if self.max_depth is not None and depth > self.max_depth:
            return False
        if self.leaves_only and not is_leaf:
            return False
        if self.include is not None and not self.include(path, module):
            return False
        if self.exclude is not None and self.exclude(path, module):
            return False
        return True

    def attach(self) -> SignalTracer:
        if self._handles:
            return self
        for module in self.model.modules():
            mid = self._module_ids.get(id(module))
            if mid is None or not self.modules[mid]["traced"]:
                continue
            pre = module.register_forward_pre_hook(self._make_pre_hook(mid), with_kwargs=True)
            try:
                post = module.register_forward_hook(self._make_post_hook(mid), with_kwargs=True, always_call=True)
            except TypeError:  # torch < 2.1
                post = module.register_forward_hook(self._make_post_hook(mid), with_kwargs=True)
            self._handles += [pre, post]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._stack.clear()

    def __enter__(self) -> SignalTracer:
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    # ------------------------------------------------------------------ steps

    def step(self, label: str | None = None, **info: Any) -> None:
        """Start a new step manually (disables automatic per-root-call steps)."""
        self._auto_steps = False
        self._new_step(label, info)

    def _new_step(self, label: str | None, info: dict[str, Any] | None = None) -> None:
        idx = len(self.steps)
        self.steps.append({"index": idx, "label": label or f"step {idx}", **(info or {})})
        self._call_counts = {}

    def _recording(self) -> bool:
        return self.max_steps is None or len(self.steps) <= self.max_steps

    # ------------------------------------------------------------------ hooks

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _summarise(self, obj: Any, prefix: str, detail: bool) -> tuple[list[dict[str, Any]], int]:
        found: list[tuple[str, torch.Tensor]] = []
        dropped = _flatten(obj, prefix, found, self.max_tensors)
        return [{"name": n, **tensor_stats(t, self.cfg, detail=detail)} for n, t in found], dropped

    def _make_pre_hook(self, mid: int):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
            h0 = time.perf_counter()
            if mid == 0 and self._auto_steps and not self._stack:
                self._new_step(None)
            elif not self.steps:
                self._new_step(None)
            if not self._recording():
                self._stack.append((-1, 0.0, 0.0))
                self._overhead += time.perf_counter() - h0
                return
            call = self._call_counts.get(mid, 0)
            self._call_counts[mid] = call + 1
            ev: dict[str, Any] = {
                "id": len(self.events),
                "module": mid,
                "step": len(self.steps) - 1,
                "call": call,
                "order": self._order,
                "parent": self._stack[-1][0] if self._stack else None,
                "depth": len(self._stack),
            }
            self._order += 1
            if self.capture_inputs:
                ins, dropped = self._summarise(args, "in", self.input_detail)
                if kwargs:
                    kw, d2 = self._summarise(kwargs, "in", self.input_detail)
                    ins += kw
                    dropped += d2
                ev["inputs"] = ins
                if dropped:
                    ev["inputs_dropped"] = dropped
            self.events.append(ev)
            self._sync()
            self._overhead += time.perf_counter() - h0
            self._stack.append((ev["id"], time.perf_counter(), self._overhead))

        return hook

    def _make_post_hook(self, mid: int):
        def hook(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
            self._sync()
            t1 = time.perf_counter()
            ov1 = self._overhead
            # Pop until we find this module's frame (robust to exceptions in children).
            frame = None
            while self._stack:
                cand = self._stack.pop()
                if cand[0] == -1 or self.events[cand[0]]["module"] == mid:
                    frame = cand
                    break
            if frame is None or frame[0] == -1:
                self._overhead += time.perf_counter() - t1
                return
            ev = self.events[frame[0]]
            ev["ms"] = round((t1 - frame[1] - (ov1 - frame[2])) * 1000, 4)
            if output is None:
                ev["outputs"] = []
            else:
                outs, dropped = self._summarise(output, "out", True)
                ev["outputs"] = outs
                if dropped:
                    ev["outputs_dropped"] = dropped
            self._overhead += time.perf_counter() - t1

        return hook

    # ------------------------------------------------------------------ export

    @property
    def trace(self) -> dict[str, Any]:
        n_total = sum(p.numel() for p in self.model.parameters())
        first_param = next(self.model.parameters(), None)
        return {
            "format": "sigtrace",
            "version": TRACE_VERSION,
            "meta": {
                "name": self.name,
                "model_type": type(self.model).__name__,
                "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "n_params": n_total,
                "device": str(first_param.device) if first_param is not None else None,
                "param_dtype": str(first_param.dtype).replace("torch.", "") if first_param is not None else None,
                "tracer_overhead_s": round(self._overhead, 3),
                "config": {
                    "leaves_only": self.leaves_only,
                    "max_depth": self.max_depth,
                    "capture_inputs": self.capture_inputs,
                    "hist_bins": self.cfg.hist_bins,
                    "heatmap_size": list(self.cfg.heatmap_size),
                },
            },
            "modules": self.modules,
            "steps": self.steps if self.max_steps is None else self.steps[: self.max_steps],
            "events": self.events,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.trace, separators=(",", ":"), allow_nan=False, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Write the trace as JSON (gzip-compressed if the name ends in .gz)."""
        path = Path(path)
        data = self.to_json().encode()
        path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
        return path

    def save_html(self, path: str | Path) -> Path:
        """Write a self-contained HTML viewer with this trace embedded."""
        from .viewer import render_html

        path = Path(path)
        path.write_text(render_html(self.trace), encoding="utf-8")
        return path


def trace(model: nn.Module, *inputs: Any, **kwargs: Any) -> SignalTracer:
    """One-shot helper: run ``model(*inputs)`` under a tracer in eval/no-grad mode."""
    accepted = inspect.signature(SignalTracer.__init__).parameters
    tracer_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in accepted and k != "model"}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), SignalTracer(model, **tracer_kwargs) as tracer:
            model(*inputs, **kwargs)
    finally:
        model.train(was_training)
    return tracer
