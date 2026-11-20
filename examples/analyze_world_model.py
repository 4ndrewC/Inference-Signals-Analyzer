"""Analyse a JEPA world model: embedding health, prediction quality, module/input influence.

Adapt `step` to your model: it runs one JEPA forward on a batch and returns the
context embeddings, the target embeddings and the predictor's outputs (lists for
multi-step rollouts). Everything else is generic.

    python examples/analyze_world_model.py && xdg-open world_model_jepa.html
"""

import torch

from sigtrace import JEPAOutput, analyze_jepa
from sigtrace.demo_jepa import sample_batch, train_world_model


def step(model, batch):
    ctx = model.context_encoder(batch["obs"])
    z, preds, targets = ctx, [], []
    for h in range(batch["actions"].shape[1]):  # roll the predictor forward
        z = model.predictor(z, batch["actions"][:, h])
        preds.append(z)
        with torch.no_grad():  # stop-grad EMA target, as in training
            targets.append(model.target_encoder(batch["future_obs"][:, h]))
    return JEPAOutput(context=ctx, target=targets, prediction=preds)


def main():
    model = train_world_model(steps=400)  # replace with your trained model
    batch = sample_batch(256, horizon=4, seed=0)  # a held-out batch; batch size > 1 for mean-ablation
    report = analyze_jepa(
        model,
        step,
        batch,
        roles={"context": model.context_encoder, "target": model.target_encoder, "predictor": model.predictor},
        # ablate_filter=lambda path, m: m.__class__.__name__ in {"Linear", "Conv2d"},  # narrow big models
    )
    print(report.summary())
    report.save("world_model_jepa.json.gz")
    print(report.save_html("world_model_jepa.html"))


if __name__ == "__main__":
    main()
