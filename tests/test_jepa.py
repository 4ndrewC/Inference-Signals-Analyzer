import json

import pytest
import torch
from torch import nn

from sigtrace import JEPAOutput, analyze_jepa, embedding_stats
from sigtrace.jepa import prediction_stats
from sigtrace.viewer import render_html


# ----------------------------------------------------------------------- embedding health
def test_embedding_stats_healthy():
    st = embedding_stats(torch.randn(2000, 32))
    assert st["status"] == "healthy"
    assert st["erank"] > 25 and st["low_var_dims"] == 0
    assert st["var_frac"] > 0.95
    assert sum(st["spectrum"]) == pytest.approx(1, abs=1e-3)
    assert len(st["std"]) == 32 and st["corr"]["rows"] == 32
    json.dumps(st, allow_nan=False)


def test_embedding_stats_full_collapse_despite_rank():
    # Every sample maps to the same vector plus tiny isotropic noise: the residue
    # has high effective rank, but almost no energy varies across samples.
    z = torch.randn(1, 32) + 1e-3 * torch.randn(500, 32)
    st = embedding_stats(z)
    assert st["status"] == "collapsed"
    assert st["erank"] > 10 and st["var_frac"] < 1e-3
    assert st["cos_mean"] > 0.999


def test_embedding_stats_dimensional_collapse():
    z = torch.randn(1000, 2) @ torch.randn(2, 64)  # rank-2 in 64 dims
    st = embedding_stats(z)
    assert st["status"] == "dimensional"
    assert st["erank"] < 2.5


def test_embedding_stats_constant_and_dead_dims():
    st = embedding_stats(torch.ones(100, 16))
    assert st["status"] == "collapsed" and "corr" not in st
    z = torch.randn(400, 16)
    z[:, :4] = 0.5  # four dead dimensions
    st = embedding_stats(z)
    assert st["low_var_dims"] == 4
    json.dumps(st, allow_nan=False)


def test_prediction_stats_reference_points():
    t = torch.randn(256, 8)
    assert prediction_stats(t.clone(), t)["r2"] == pytest.approx(1.0)
    assert prediction_stats(t.mean(0, keepdim=True).expand_as(t), t)["r2"] == pytest.approx(0.0, abs=1e-6)
    ps = prediction_stats(t * 0.9, t, ctx=t + 0.1)
    assert ps["copy_r2"] is not None and ps["copy_cos"] > 0.9
    with pytest.raises(ValueError):
        prediction_stats(torch.randn(4, 8), torch.randn(4, 9))


# ----------------------------------------------------------------------- I-JEPA style token model
class TokenEncoder(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.patch = nn.Linear(4, d)
        self.mix = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())

    def forward(self, x):  # x [B, N, 4]
        return self.mix(self.patch(x))


class TokenPredictor(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.pos = nn.Embedding(8, d)
        self.net = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d))
        self.unused = nn.Linear(d, d)

    def forward(self, ctx, tgt_idx):
        _ = self.unused(ctx)  # called but its output never matters
        pooled = ctx.mean(1, keepdim=True)  # [B, 1, d]
        return self.net(pooled + self.pos(tgt_idx))  # [B, M, d]


class IJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = TokenEncoder()
        self.target_encoder = TokenEncoder().requires_grad_(False)
        self.predictor = TokenPredictor()


def ijepa_step(model, batch):
    x = batch["patches"]  # [B, 8, 4]
    b = x.shape[0]
    tgt_idx = torch.stack([torch.randperm(8)[:3] for _ in range(b)])  # random masks
    ctx = model.context_encoder(x)
    with torch.no_grad():  # typical: stop-grad through the EMA target branch
        tgt_all = model.target_encoder(x)
    tgt = torch.gather(tgt_all, 1, tgt_idx[..., None].expand(-1, -1, tgt_all.shape[-1]))
    pred = model.predictor(ctx, tgt_idx)
    return {"context": ctx, "target": tgt, "prediction": pred}


def test_analyze_ijepa_tokens():
    torch.manual_seed(0)
    model = IJEPA()
    batch = {"patches": torch.randn(32, 8, 4), "meta": torch.randn(32, 3)}
    rng_before = torch.random.get_rng_state()
    rep = analyze_jepa(model, ijepa_step, batch)
    assert torch.equal(rng_before, torch.random.get_rng_state())  # caller RNG untouched

    j = rep.jepa
    paths = {m["id"]: m["path"] for m in rep.trace["modules"]}
    roles = {m["path"]: m.get("role") for m in rep.trace["modules"]}
    assert roles["context_encoder.patch"] == "context" and roles["target_encoder.mix.1"] == "target"
    assert roles["predictor.net.0"] == "predictor" and roles[""] is None

    # token embeddings are flattened to rows: 32 samples x 8 tokens
    assert j["embeddings"]["context"]["rows"] == 256 and j["embeddings"]["target"]["rows"] == 96
    lc = j["landscape"]["roles"]
    assert len(lc["context"]["tok"]) == lc["context"]["n"] and lc["prediction"]["n"] == lc["target"]["n"]
    assert len(lc["target"]["xy"]) == lc["target"]["n"] * j["landscape"]["pcs"]

    inf = {paths[r["module"]]: r for r in j["influence"]["modules"]}
    # seeded runs make a no-op ablation exactly zero even with random masking
    assert inf["predictor.unused"]["d_loss"] == 0 and inf["predictor.unused"]["pred_drift"] == 0
    assert abs(inf["predictor.net"]["d_loss"]) > 0.01
    # target branch ran under no_grad: ablation still measured, no attribution
    assert inf["target_encoder.mix"]["tgt_drift"] > 0 and "attr" not in inf["target_encoder.mix"]
    assert inf["context_encoder.patch"]["attr"] > 0

    ins = {r["name"]: r for r in j["influence"]["inputs"]}
    assert ins["meta"]["d_loss"] == 0 and ins["meta"]["pred_drift"] == 0  # unused input
    assert ins["patches"]["d_loss"] != 0

    # parameters untouched by attribution
    assert all(p.requires_grad for p in model.context_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert model.training  # train/eval mode restored

    json.loads(json.dumps(rep.trace, allow_nan=False))
    assert '"jepa"' in render_html(rep.trace)
    assert "prediction R²" in rep.summary()


def test_world_model_rollout_and_collapse_detection():
    from sigtrace.demo_jepa import WorldModelJEPA, jepa_step, sample_batch, train_world_model

    model = train_world_model(steps=40)
    rep = analyze_jepa(model, jepa_step, sample_batch(64, horizon=3, seed=1), max_ablations=5)
    j = rep.jepa
    assert len(j["prediction"]["horizons"]) == 3
    assert j["prediction"]["horizons"][0]["copy_r2"] is not None
    assert set(j["landscape"]["roles"]["prediction"]["h"]) == {0, 1, 2}
    assert sum("d_loss" in r for r in j["influence"]["modules"]) == 5  # max_ablations respected
    assert {r["name"] for r in j["influence"]["inputs"]} == {"obs", "future_obs", "actions"}

    # force a total collapse: encoders output a constant vector
    collapsed = WorldModelJEPA()
    for enc in (collapsed.context_encoder, collapsed.target_encoder):
        nn.init.zeros_(enc.head.weight)
        nn.init.constant_(enc.head.bias, 0.3)
    rep = analyze_jepa(collapsed, jepa_step, sample_batch(64, horizon=2, seed=2), ablate=False, attribution=False)
    assert rep.jepa["embeddings"]["target"]["status"] == "collapsed"
    assert rep.jepa["prediction"]["overall"]["r2"] is None  # target variance is zero


def test_step_fn_return_forms():
    model = IJEPA()
    batch = {"patches": torch.randn(8, 8, 4)}

    def as_tuple(m, b):
        o = ijepa_step(m, b)
        return o["context"], o["target"], o["prediction"]

    def as_output(m, b):
        o = ijepa_step(m, b)
        return JEPAOutput(o["context"], o["target"], o["prediction"], loss=(o["prediction"] - o["target"]).abs().mean())

    for fn in (as_tuple, as_output):
        rep = analyze_jepa(model, fn, batch, ablate=False, input_influence=False)
        assert rep.jepa["loss"] > 0
    with pytest.raises(TypeError):
        analyze_jepa(model, lambda m, b: 42, batch)
