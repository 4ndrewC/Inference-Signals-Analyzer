"""Small built-in models that exercise the tracer and show typical signal pathologies."""

from __future__ import annotations

import math

import torch
from torch import nn

from .tracer import SignalTracer


# --------------------------------------------------------------------------- MLP
def make_mlp(depth: int = 12, width: int = 256, gain: float = 0.6) -> nn.Sequential:
    """Deep tanh MLP with under-scaled init: the signal shrinks layer by layer."""
    layers: list[nn.Module] = []
    d_in = 32
    for _ in range(depth):
        lin = nn.Linear(d_in, width)
        nn.init.normal_(lin.weight, std=gain / math.sqrt(d_in))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d_in = width
    layers.append(nn.Linear(d_in, 10))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- CNN
class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))


def synthetic_image(seed: int, size: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    cx, cy, r = (torch.rand(3, generator=g) * torch.tensor([1.2, 1.2, 0.4]) - torch.tensor([0.6, 0.6, -0.2])).tolist()
    disc = (((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).float()
    stripes = torch.sin(xx * (6 + seed % 5) + yy * 3)
    img = torch.stack([disc, 0.5 * stripes, xx * yy]) + 0.1 * torch.randn(3, size, size, generator=g)
    return img[None]


# --------------------------------------------------------------------------- tiny GPT
class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.softmax = nn.Softmax(dim=-1)  # a module, so attention probabilities get traced
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(c // self.heads)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        att = self.softmax(scores.masked_fill(mask, float("-inf")))
        y = (att @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d, 4 * d)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 64, heads: int = 4, layers: int = 4, ctx: int = 64) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def run_demo(kind: str = "transformer", steps: int = 8) -> SignalTracer:
    torch.manual_seed(0)
    if kind == "mlp":
        model = make_mlp().eval()
        with torch.inference_mode(), SignalTracer(model, name="DeepTanhMLP") as tracer:
            for i in range(max(1, steps // 2)):
                model(torch.randn(16, 32) * (1 + i))
        return tracer

    if kind == "cnn":
        model = SmallCNN().eval()
        with torch.inference_mode(), SignalTracer(model) as tracer:
            for i in range(max(1, steps // 2)):
                model(synthetic_image(i))
        return tracer

    model = TinyGPT().eval()
    with torch.no_grad():
        # Plant a "massive activation" outlier channel in the residual stream.
        model.blocks[1].mlp.proj.bias[7] = 30.0
    idx = torch.randint(0, 128, (1, 6))
    with torch.inference_mode(), SignalTracer(model) as tracer:
        for _ in range(steps):
            logits = model(idx)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            tracer.steps[-1]["label"] = f"token {idx.shape[1] - 1}"
    return tracer
