"""Small integration fit on an actual compiler trajectory, not a pilot evaluation."""
from dataclasses import fields
import json
import math
from pathlib import Path
import torch
from .schema import Observation


def trajectory(path):
    report = json.loads(Path(path).read_text())
    observations = [Observation(**{f.name: x[f.name] for f in fields(Observation)}) for x in report["observations"]]
    return report, observations


def fit_smoke(model, path, steps=25):
    report, observations = trajectory(path)
    # Behavior cloning checks that target heads learn from actual registered
    # actions. Value/resource labels are measured, but one smoke trace is not
    # a representative training set or a generalization/performance result.
    chosen = {t["source"]: t["action"] for t in report["transitions"] if t["success"]}
    target_value = math.log(report["median_latency_ms"]) / 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses, gradient = [], 0.0
    for step in range(steps):
        state, loss, first = None, torch.tensor(0.0), None
        for observation in observations:
            (scores, value, resources), state = model(observation, state)
            if first is None:
                first = state["h"]
                first.retain_grad()
            action = chosen.get(observation.snapshot, {"kind": "advance", "target": 0, "parameters": {}})
            candidates = observation.graph["candidates"]
            index = next(i for i, c in enumerate(candidates) if all(c[k] == action[k] for k in ("kind", "target", "parameters")))
            loss = loss + torch.nn.functional.cross_entropy(scores[None], torch.tensor([index]))
        terminal = (value - target_value).square()
        terminal = terminal + 0.01 * (resources - torch.tensor([
            math.log1p(report["measurement"]["code_bytes"]) / 10,
            math.log1p(report["measurement"]["costs"]["commands_s"]) / 10])).square().mean()
        # This derivative uses terminal loss only; intermediate action losses
        # cannot falsely demonstrate cross-stage connectivity.
        cross = torch.autograd.grad(terminal, first, retain_graph=True, allow_unused=True)[0]
        gradient = 0.0 if cross is None else cross.abs().sum().item()
        loss = loss + terminal
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    if not losses[-1] < losses[0]:
        raise RuntimeError("integration fit did not reduce loss")
    if model.history and gradient <= 0:
        raise RuntimeError("terminal loss has no gradient to first-stage hidden state")
    return {"kind": "single-smoke-trajectory behavior fit; not N4", "loss": losses,
            "terminal_to_first_hidden_gradient_l1": gradient, "steps": steps}
