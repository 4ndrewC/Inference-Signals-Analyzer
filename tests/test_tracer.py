import base64
import gzip
import json
import math

import pytest
import torch
from torch import nn

from sigtrace import SignalTracer, StatsConfig, render_html, tensor_stats, trace
from sigtrace.demo import TinyGPT, make_mlp


def _events_for(tr, path):
    ids = {m["id"] for m in tr.modules if m["path"] == path}
    return [e for e in tr.events if e["module"] in ids]


# ----------------------------------------------------------------------- stats
def test_stats_match_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32) * 3 + 1
    st = tensor_stats(x, StatsConfig())
    assert st["shape"] == [4, 8, 32] and st["numel"] == x.numel()
    assert st["mean"] == pytest.approx(x.mean().item(), rel=1e-3)
    assert st["std"] == pytest.approx(x.std(unbiased=False).item(), rel=1e-3)
    assert st["rms"] == pytest.approx(x.pow(2).mean().sqrt().item(), rel=1e-3)
    assert st["min"] == pytest.approx(x.min().item(), rel=1e-4)
    assert st["max"] == pytest.approx(x.max().item(), rel=1e-4)
    assert sum(st["hist"]["counts"]) == x.numel()
    assert st["q"][2] == pytest.approx(x.median().item(), abs=0.05)


def test_stats_nonfinite_are_counted_and_excluded():
    x = torch.tensor([1.0, 2.0, float("nan"), float("inf"), -float("inf"), 3.0])
    st = tensor_stats(x, StatsConfig())
    assert st["nan"] == 1 and st["inf"] == 2
    assert st["mean"] == pytest.approx(2.0)
    assert st["min"] == 1.0 and st["max"] == 3.0
    json.dumps(st, allow_nan=False)  # must be strict-JSON serialisable


def test_stats_all_nan_and_empty():
    st = tensor_stats(torch.full((3, 3), float("nan")), StatsConfig())
    assert st["nan"] == 9 and "mean" not in st
    st = tensor_stats(torch.empty(0, 4), StatsConfig())
    assert st["numel"] == 0
    json.dumps(st, allow_nan=False)


def test_heatmap_layouts_and_quantisation():
    cfg = StatsConfig(heatmap_size=(16, 32))
    seq = tensor_stats(torch.randn(2, 40, 100), cfg)["heatmap"]
    assert seq["layout"] == "sequence" and (seq["rows"], seq["cols"]) == (16, 32) and seq["src"] == [40, 100]
    q = base64.b64decode(seq["q"])
    assert len(q) == 16 * 32 and max(q) <= 254

    conv = tensor_stats(torch.randn(1, 20, 12, 12), cfg)["heatmap"]
    assert conv["layout"] == "mosaic" and conv["channels"] == 16 and conv["grid"] == [4, 4]
    assert len(base64.b64decode(conv["q"])) == conv["rows"] * conv["cols"]

    x = torch.randn(8, 8)
    x[2, 3] = float("nan")
    hm = tensor_stats(x, cfg)["heatmap"]
    assert base64.b64decode(hm["q"])[2 * 8 + 3] == 255


def test_channel_profile_finds_outlier():
    x = torch.randn(2, 10, 512) * 0.1
    x[..., 300] += 50
    ch = tensor_stats(x, StatsConfig(channel_bins=128))["channels"]
    assert ch["n"] == 512 and len(ch["values"]) == 128
    assert ch["top"][0][0] == 300
    assert max(ch["values"]) == pytest.approx(ch["top"][0][1], rel=1e-3)  # max-pool keeps the outlier


def test_int_and_bool_tensors():
    st = tensor_stats(torch.randint(0, 5, (100,)), StatsConfig())
    assert st["kind"] == "int" and "heatmap" not in st and len(st["hist"]["counts"]) == 5
    st = tensor_stats(torch.rand(10) > 0.5, StatsConfig())
    assert st["kind"] == "bool"


# ----------------------------------------------------------------------- tracer
def test_mlp_trace_structure_and_order():
    model = make_mlp(depth=3, width=16).eval()
    with torch.inference_mode(), SignalTracer(model) as tr:
        model(torch.randn(4, 32))
        model(torch.randn(4, 32))
    assert len(tr.steps) == 2
    per_step = [e for e in tr.events if e["step"] == 0]
    assert len(per_step) == 1 + 7  # root + 3x(Linear, Tanh) + head
    assert [tr.modules[e["module"]]["path"] for e in per_step] == ["", "0", "1", "2", "3", "4", "5", "6"]
    assert all(e["parent"] == per_step[0]["id"] for e in per_step[1:])
    # output of layer i is the input of layer i+1
    assert per_step[1]["outputs"][0]["rms"] == pytest.approx(per_step[2]["inputs"][0]["rms"])
    assert all(e["ms"] >= 0 for e in per_step)
    # hooks are removed on exit
    n = len(tr.events)
    model(torch.randn(4, 32))
    assert len(tr.events) == n


def test_inplace_relu_input_captured_before_mutation():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(inplace=True))
    with torch.no_grad(), SignalTracer(model) as tr:
        model(torch.randn(64, 8))
    relu = _events_for(tr, "1")[0]
    assert relu["inputs"][0]["min"] < 0  # pre-hook saw the negative values
    assert relu["outputs"][0]["min"] >= 0


def test_nested_outputs_and_kwargs():
    class M(nn.Module):
        def forward(self, x, *, scale=None):
            return {"a": x * scale, "b": (x + 1, [x - 1, 7])}

    with SignalTracer(M()) as tr:
        x = torch.ones(3)
        tr.model(x, scale=torch.tensor(2.0))
    ev = tr.events[0]
    assert [t["name"] for t in ev["inputs"]] == ["in[0]", "in.scale"]
    assert [t["name"] for t in ev["outputs"]] == ["out.a", "out.b[0]", "out.b[1][0]"]


def test_max_tensors_limit():
    class M(nn.Module):
        def forward(self, x):
            return tuple(x + i for i in range(20))

    with SignalTracer(M(), max_tensors=5) as tr:
        tr.model(torch.zeros(2))
    assert len(tr.events[0]["outputs"]) == 5 and tr.events[0]["outputs_dropped"] == 15


def test_include_exclude_leaves_depth():
    model = TinyGPT(layers=2).eval()
    idx = torch.randint(0, 128, (1, 5))
    with torch.no_grad(), SignalTracer(model, include=["Linear"]) as tr:
        model(idx)
    types = {tr.modules[e["module"]]["type"] for e in tr.events}
    assert types == {"TinyGPT", "Linear"}  # root always traced

    with torch.no_grad(), SignalTracer(model, exclude="blocks.*.attn*") as tr:
        model(idx)
    assert not any(".attn" in tr.modules[e["module"]]["path"] for e in tr.events)

    with torch.no_grad(), SignalTracer(model, leaves_only=True) as tr:
        model(idx)
    assert all(tr.modules[e["module"]]["leaf"] or e["module"] == 0 for e in tr.events)

    with torch.no_grad(), SignalTracer(model, max_depth=1) as tr:
        model(idx)
    assert max(tr.modules[e["module"]]["depth"] for e in tr.events) == 1


def test_shared_module_multiple_calls():
    lin = nn.Linear(4, 4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = lin
            self.b = lin  # same module object under two names

        def forward(self, x):
            return self.b(self.a(x))

    with SignalTracer(M()) as tr:
        tr.model(torch.randn(2, 4))
    calls = [e for e in tr.events if e["module"] != 0]
    assert len(calls) == 2 and calls[0]["module"] == calls[1]["module"]
    assert [c["call"] for c in calls] == [0, 1]


def test_manual_steps_and_max_steps():
    model = nn.Linear(4, 4)
    with SignalTracer(model) as tr:
        tr.step("warmup")
        model(torch.randn(1, 4))
        model(torch.randn(1, 4))
        tr.step("main", temperature=0.7)
        model(torch.randn(1, 4))
    assert [s["label"] for s in tr.steps] == ["warmup", "main"]
    assert tr.steps[1]["temperature"] == 0.7
    assert [e["step"] for e in tr.events] == [0, 0, 1]
    assert [e["call"] for e in tr.events] == [0, 1, 0]

    with SignalTracer(model, max_steps=2) as tr:
        for _ in range(5):
            model(torch.randn(1, 4))
    assert len(tr.trace["steps"]) == 2 and len(tr.events) == 2


def test_exception_in_forward_keeps_tracer_consistent():
    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    model = nn.Sequential(nn.Linear(2, 2), Boom())
    with SignalTracer(model) as tr:
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 2))
        model[0](torch.randn(1, 2))  # tracer still works afterwards
    assert tr.events[-1]["outputs"]


def test_nan_propagation_is_recorded():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    t = trace(model, torch.randn(3, 4))
    lin = _events_for(t, "0")[0]
    assert lin["outputs"][0]["nan"] == 3
    json.loads(t.to_json())


def test_param_stats_and_trace_helper_restores_mode():
    model = make_mlp(depth=2, width=8)
    model.train()
    t = trace(model, torch.randn(2, 32), hist_bins=16)
    assert model.training
    lin = next(m for m in t.modules if m["type"] == "Linear")
    names = [p["name"] for p in lin["params"]]
    assert names == ["weight", "bias"] and len(lin["params"][0]["hist"]["counts"]) == 16


def test_save_json_gz_and_html(tmp_path):
    model = TinyGPT(layers=1).eval()
    with torch.no_grad(), SignalTracer(model, name="</script><b>x") as tr:
        model(torch.randint(0, 128, (1, 4)))
    p = tr.save(tmp_path / "t.json.gz")
    data = json.loads(gzip.decompress(p.read_bytes()))
    assert data["format"] == "sigtrace" and data["meta"]["name"] == "</script><b>x"
    assert json.loads(tr.save(tmp_path / "t.json").read_text())["version"] == 1

    html = tr.save_html(tmp_path / "t.html").read_text()
    assert html.count("</script>") == 2  # data must not close its own <script>
    start = html.index('id="sigtrace-data" type="application/json">') + len('id="sigtrace-data" type="application/json">')
    payload = html[start : html.index("</script>", start)]
    assert json.loads(payload.replace("<\\/", "</"))["meta"]["name"] == "</script><b>x"


def test_render_html_without_trace_keeps_placeholder():
    assert "/*__SIGTRACE_DATA__*/null" in render_html(None)


def test_gpt_generation_steps_and_json_strict():
    model = TinyGPT(layers=2).eval()
    idx = torch.randint(0, 128, (1, 3))
    with torch.inference_mode(), SignalTracer(model) as tr:
        for _ in range(3):
            idx = torch.cat([idx, model(idx)[:, -1].argmax(-1, keepdim=True)], 1)
    assert len(tr.steps) == 3
    sm = [e for e in tr.events if tr.modules[e["module"]]["type"] == "Softmax"]
    assert sm[0]["outputs"][0]["heatmap"]["layout"] == "mosaic"
    assert all(math.isclose(e["outputs"][0]["max"], 1.0) for e in sm[:1])
    json.loads(tr.to_json())
