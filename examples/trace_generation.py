"""Trace autoregressive generation step by step with the built-in TinyGPT.

The same pattern works for any decoder (e.g. a Hugging Face model): each call
of the root module becomes one step, or label steps yourself with
``tracer.step(...)``. For big models, narrow the trace with ``include=`` /
``exclude=`` / ``max_depth=`` and ``max_steps=`` to keep the file small.
"""

import torch

from sigtrace import SignalTracer
from sigtrace.demo import TinyGPT


def main():
    torch.manual_seed(0)
    model = TinyGPT().eval()
    idx = torch.randint(0, 128, (1, 8))
    with torch.inference_mode(), SignalTracer(model, exclude=["Dropout"], max_steps=16) as tracer:
        for _ in range(16):
            logits = model(idx)
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
            tracer.steps[-1]["label"] = f"token {idx.shape[1] - 1}"
    print(tracer.save_html("generation_trace.html"))


if __name__ == "__main__":
    main()
