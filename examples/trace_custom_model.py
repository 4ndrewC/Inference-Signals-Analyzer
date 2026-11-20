"""Trace a custom model that has a numerical bug, and write a standalone HTML report.

    python examples/trace_custom_model.py && xdg-open custom_trace.html
"""

import torch
from torch import nn

from sigtrace import SignalTracer


class LogFeatures(nn.Module):
    """Buggy: log() of a signed activation produces NaN for negative inputs."""

    def forward(self, x):
        return torch.log(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.LayerNorm(64))
        self.features = LogFeatures()
        self.mixer = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(self.mixer(self.features(h)))


def main():
    torch.manual_seed(0)
    model = Net().eval()
    with torch.inference_mode(), SignalTracer(model) as tracer:
        for i in range(3):
            tracer.step(f"batch {i}")  # manual step labels
            model(torch.randn(32, 16))
    tracer.save("custom_trace.json.gz")
    path = tracer.save_html("custom_trace.html")
    print(f"wrote {path} ({len(tracer.events)} module calls)")


if __name__ == "__main__":
    main()
