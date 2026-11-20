"""A tiny action-conditioned JEPA world model for demos and tests.

Environment: a Gaussian blob on a 16x16 image moves by ``0.25 * action`` each
step. The context encoder embeds the first frame, the predictor rolls the
embedding forward with the actions, and an EMA target encoder embeds the future
frames.

``train_world_model(collapse=False)`` trains with a VICReg-style variance /
covariance penalty (healthy embeddings); ``collapse=True`` trains with the bare
prediction loss and no EMA lag, which collapses the representation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .jepa import JEPAOutput, JEPAReport, analyze_jepa

SIZE = 16


def render(pos: torch.Tensor, noise: float = 0.05, g: torch.Generator | None = None) -> torch.Tensor:
    """pos [B, 2] in [-1, 1] -> images [B, 1, SIZE, SIZE]."""
    lin = torch.linspace(-1, 1, SIZE)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    d2 = (xx[None] - pos[:, 0, None, None]) ** 2 + (yy[None] - pos[:, 1, None, None]) ** 2
    img = torch.exp(-d2 / (2 * 0.18**2))
    return (img + noise * torch.randn(img.shape, generator=g))[:, None]


def sample_batch(batch: int = 128, horizon: int = 1, seed: int | None = None) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    pos = torch.rand(batch, 2, generator=g) * 1.6 - 0.8
    actions = torch.rand(batch, horizon, 2, generator=g) * 2 - 1
    frames = [render(pos, g=g)]
    for h in range(horizon):
        pos = (pos + 0.25 * actions[:, h]).clamp(-1, 1)
        frames.append(render(pos, g=g))
    return {"obs": frames[0], "future_obs": torch.stack(frames[1:], 1), "actions": actions}


class ObsEncoder(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU()
        )
        self.proj = nn.Linear(32 * 4 * 4, 64)
        self.act = nn.GELU()
        self.head = nn.Linear(64, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.proj(self.conv(x).flatten(1))))


class ActionPredictor(nn.Module):
    def __init__(self, dim: int = 32, action_dim: int = 2) -> None:
        super().__init__()
        self.action_embed = nn.Linear(action_dim, 32)
        self.fc1 = nn.Linear(dim + 32, 128)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(128, dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.fc2(self.act(self.fc1(torch.cat([z, self.action_embed(a)], -1))))


class WorldModelJEPA(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.context_encoder = ObsEncoder(dim)
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.predictor = ActionPredictor(dim)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.lerp_(pc, 1 - momentum)


def jepa_step(model: WorldModelJEPA, batch: dict[str, torch.Tensor]) -> JEPAOutput:
    """Encode the first frame, roll the predictor forward with the actions, embed the future frames."""
    ctx = model.context_encoder(batch["obs"])
    z, preds, tgts = ctx, [], []
    for h in range(batch["actions"].shape[1]):
        z = model.predictor(z, batch["actions"][:, h])
        preds.append(z)
        tgts.append(model.target_encoder(batch["future_obs"][:, h]))
    return JEPAOutput(context=ctx, target=tgts, prediction=preds)


def _vicreg(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zc = z - z.mean(0)
    std = torch.sqrt(zc.var(0) + 1e-4)
    var_loss = F.relu(1 - std).mean()
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - torch.diag(cov.diagonal())
    return var_loss, off.pow(2).sum() / z.shape[1]


def train_world_model(collapse: bool = False, steps: int = 400, seed: int = 0) -> WorldModelJEPA:
    torch.manual_seed(seed)
    model = WorldModelJEPA()
    opt = torch.optim.Adam([*model.context_encoder.parameters(), *model.predictor.parameters()], lr=2e-3)
    momentum = 0.0 if collapse else 0.98
    for i in range(steps):
        b = sample_batch(128, horizon=1, seed=seed * 100_000 + i)
        out = jepa_step(model, b)
        loss = F.mse_loss(out.prediction[0], out.target[0].detach())
        if not collapse:
            v, c = _vicreg(out.context)
            loss = loss + 1.0 * v + 0.04 * c
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target(momentum)
    return model.eval()


def run_jepa_demo(collapse: bool = False, horizon: int = 4, steps: int = 400) -> JEPAReport:
    model = train_world_model(collapse=collapse, steps=steps)
    batch = sample_batch(256, horizon=horizon, seed=12345)
    return analyze_jepa(
        model, jepa_step, batch, name="WorldModelJEPA" + (" (collapsed)" if collapse else ""),
        roles={"context": model.context_encoder, "target": model.target_encoder, "predictor": model.predictor},
    )
