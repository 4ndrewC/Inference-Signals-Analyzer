# sigtrace

Trace the signals flowing through a PyTorch model during inference, then explore them in an interactive, self-contained HTML viewer.

`sigtrace` attaches forward hooks to every `nn.Module` and records compact statistics for each call's inputs and outputs: RMS, std, min/max, quantiles, sparsity, NaN/Inf counts, a histogram, a downsampled activation map and a per-channel profile. Raw tensors are never kept, so traces stay small (about 1 MB for a 4-layer transformer over 8 generation steps).

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/sigtrace demo --model transformer --open   # or: mlp, cnn
```

```python
import torch
from sigtrace import SignalTracer

with torch.inference_mode(), SignalTracer(model) as tracer:
    model(inputs)                    # every root call becomes a "step"

tracer.save_html("trace.html")       # standalone viewer, works offline
tracer.save("trace.json.gz")         # raw trace (drop it onto any viewer page)
```

To do it in one call, `sigtrace.trace(model, *inputs, **tracer_kwargs)` runs the model in eval mode under `inference_mode` and returns the tracer.

### Steps

Each call of the root module starts a new step, which suits autoregressive generation (one step per token). To group calls yourself, call `tracer.step("label", **info)`. After the first manual call, automatic stepping is turned off.

### Narrowing a trace (large models)

| Option | Effect |
|---|---|
| `include=` / `exclude=` | fnmatch patterns on module path **or** class name (`"blocks.*.attn"`, `"Dropout"`), or a `(path, module) -> bool` callable |
| `leaves_only=True` | only hook modules without children |
| `max_depth=2` | only hook modules up to this nesting depth |
| `max_steps=16` | stop recording after N steps |
| `capture_inputs=False` | skip input statistics |
| `input_detail=True` | also keep heatmaps and channel profiles for inputs |
| `heatmap=False`, `heatmap_size=(64, 128)` | turn activation maps off, or change their maximum resolution |
| `hist_bins`, `channel_bins`, `sample_size` | change histogram, channel-profile and quantile resolution |
| `sync_cuda=True` | synchronise CUDA around each module so timings are accurate |

## Viewer

- **Module tree**: shows the selected metric for each module at the current step and flags modules with anomalies. You can filter it by name or type.
- **Signal flow**: plots a metric for every module call in execution order. Points are colored by module type, and backgrounds are banded by block (`blocks.0`, `blocks.1`, …). Metrics: RMS, std, mean |x|, max |x|, value range (p1–p99 band), mean, zero fraction, outlier-channel ratio, gain (out RMS ÷ in RMS), NaN/Inf count and time. You can view either the input or the output tensor.
- **Across steps**: a module × step matrix aligned with the flow chart, useful for spotting drift over generated tokens. Click a cell to jump to it.
- **Details**: full stats for each input/output tensor, a histogram, an activation map (sequence × features, or a mosaic of conv feature maps; diverging blue/gray/red around zero), a per-channel profile that shows outlier channels, the metric across steps, and parameter/buffer stats.
- **Anomalies**: NaN/Inf, exploding values (|x| > 1e4), mostly-zero or constant outputs, outlier channels (≥ 20× the median) and extreme gains (> 100× or < 0.01×).

Keys: `←`/`→` change the step, `↑`/`↓` select the previous/next module, `Space` plays through the steps. Light and dark themes follow the OS setting, and the header button switches between them.

## JEPA / world models

`analyze_jepa` runs a JEPA forward step that you supply on one analysis batch. It records the usual module signals and adds JEPA-specific diagnostics. The viewer shows these in a **JEPA** tab next to **Signals**.

```python
from sigtrace import JEPAOutput, analyze_jepa

def step(model, batch):                        # one JEPA forward, your way
    ctx = model.context_encoder(batch["obs"])
    z, preds, tgts = ctx, [], []
    for h in range(batch["actions"].shape[1]):  # optional multi-step rollout
        z = model.predictor(z, batch["actions"][:, h])
        preds.append(z)
        with torch.no_grad():
            tgts.append(model.target_encoder(batch["future_obs"][:, h]))
    return JEPAOutput(context=ctx, target=tgts, prediction=preds)   # tensors or per-horizon lists

report = analyze_jepa(model, step, batch,
                      roles={"context": model.context_encoder, "target": model.target_encoder, "predictor": model.predictor})
print(report.summary())
report.save_html("jepa.html")
```

Embeddings can be pooled (`[B, D]`) or per token (`[B, N, D]`, as in I-JEPA/V-JEPA). Token embeddings are flattened to rows. `step` may also return a dict or a `(context, target, prediction[, loss])` tuple. The loss defaults to MSE(prediction, target). If you leave out `roles`, they are guessed from child names (`context*`/`encoder`, `target*`/`ema*`, `pred*`).

| Diagnostic | What it tells you |
|---|---|
| **Embedding landscape** | Joint PCA of context, target and predicted embeddings, plotted with equal units on both axes. It has 2σ covariance ellipses, density shading, prediction→target error vectors, a horizon filter and coloring by error. By default the plot is framed by the embedding norm, so a collapsed encoder appears as a dot; "Zoom to data" is still available. |
| **Variance structure** | Per role: eigenvalue spectrum, effective rank, participation ratio, sample-specific energy (variance ÷ mean ‖z‖²), per-dimension std with a low-variance threshold, the \|corr\| matrix (redundancy) and mean cosine between samples. Verdicts: *healthy*, *offset-dominated*, *dimensional collapse* or *collapsed*. |
| **Prediction quality** | R², cosine and MSE overall and per rollout horizon, compared with a copy-context baseline, plus a histogram of per-sample error. A tiny loss on a collapsed target is flagged as trivial. |
| **Module influence** | Each module's output is mean-ablated in turn (replaced by its batch mean) and the run repeated, measuring Δloss and how far the predictions, context and target embeddings drift. A clean backward pass also gives gradient × activation of the loss. These measures also appear as metrics in the Signals view. |
| **Input influence** | Each batched input (e.g. `actions`) is shuffled across the batch in turn, giving Δloss and prediction drift. A world model that ignores its actions shows almost no drift for them. |

Options: `ablation_mode="mean"|"zero"`, `max_ablations`, `ablate_filter=(path, module) -> bool`, `attribution=False`, `input_influence=False`, `max_points` / `n_pcs` (landscape) and `low_var_rel`. Every run inside the analysis uses the same RNG seed (`seed=0`), so random masks are identical across ablations, and your own RNG state is left untouched. Mean-ablation needs a batch size greater than 1.

Demos: `sigtrace demo --model jepa` (healthy, VICReg-regularized) and `sigtrace demo --model jepa-collapsed` (trained without regularization or EMA lag). Each trains a small action-conditioned world model for a few seconds.

## CLI

```bash
sigtrace demo  [--model mlp|cnn|transformer|jepa|jepa-collapsed] [--steps N] [--horizon H] [-o out.html] [--open]
sigtrace html  trace.json.gz [-o trace.html] [--open]   # embed a trace into a standalone page
sigtrace serve [trace.json.gz] [--port 8765]            # for remote/GPU boxes: forward the port
```

`python -m sigtrace …` works too. A viewer page without an embedded trace also accepts `?trace=<url>` or drag-and-drop.

## Layout

```
sigtrace/
  stats.py          tensor → JSON summary (one device→host sync per tensor)
  tracer.py         SignalTracer: hooks, steps, timing, export
  viewer/index.html the viewer (vanilla JS, no dependencies)
  jepa.py           analyze_jepa: embedding health, prediction quality, influence, landscape
  cli.py, demo.py, demo_jepa.py
examples/           trace_custom_model.py (NaN bug), trace_generation.py, analyze_world_model.py
tests/
```

## Notes

- Input stats are computed in the forward **pre**-hook, so in-place ops such as `ReLU(inplace=True)` don't corrupt them.
- Timings exclude the tracer's own overhead but still include Python hook dispatch, so treat them as relative.
- Modules called several times in one step (shared weights) are tracked separately, one record per call.
- Heatmaps are average-pooled and channel profiles are max-pooled, so outliers survive downsampling.
