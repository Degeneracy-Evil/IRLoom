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


def comparison_groups(paths, config):
    """Prepare immutable labels once; never order latencies across workloads."""
    import statistics
    groups = {}
    for path in paths:
        report, observations = trajectory(path)
        if any(not t["success"] for t in report["transitions"]):
            raise ValueError("failed trajectory cannot receive a success label")
        if not report["measurement"]["correctness"]:
            raise ValueError("missing independent correctness checks")
        key = json.dumps(report.get("comparison_group", {"program": report["program"]}), sort_keys=True)
        rows = groups.setdefault(key, [])
        samples = [x["latency_ms"] for x in report["measurement"]["samples"]]
        median = statistics.median(samples)
        rows.append({"report": report, "observations": observations, "latency": median,
                     "mad": statistics.median(abs(x - median) for x in samples),
                     "ir": Path(report["final_ir"]).read_bytes() if report.get("final_ir") else None,
                     "object": (Path(path).parent / "program.o").read_bytes() if report.get("final_ir") else None})
    prepared = []
    for key, rows in groups.items():
        defaults = [i for i,r in enumerate(rows) if r["report"].get("execution", {}).get("policy") == "default"]
        if not defaults:
            raise ValueError("comparison group has no declared default reference")
        pairs, skipped = [], {"identical_artifact": 0, "near_tie": 0}
        for i in range(len(rows)):
            for j in range(i):
                a,b=rows[i],rows[j]
                if (a["ir"] is not None and a["ir"] == b["ir"]) or (a["object"] is not None and a["object"] == b["object"]):
                    skipped["identical_artifact"] += 1
                    continue
                threshold = max(config.get("near_tie_absolute_ms", 0),
                                config.get("near_tie_relative", 0) * max(a["latency"],b["latency"]),
                                config.get("uncertainty_mad_multiplier", 0) * max(a["mad"],b["mad"]))
                if abs(a["latency"]-b["latency"]) <= threshold:
                    skipped["near_tie"] += 1
                    continue
                pairs.append((i,j,math.copysign(1,a["latency"]-b["latency"])))
        prepared.append({"key": json.loads(key), "rows": rows, "baseline": rows[defaults[0]]["latency"], "pairs": pairs, "skipped": skipped})
    if not prepared:
        raise ValueError("no valid comparison groups")
    return prepared


def coverage(paths, config):
    groups = comparison_groups(paths, config)
    output = []
    for group in groups:
        stages = {}
        for row in group["rows"]:
            selected = {t["source"]: t["action"] for t in row["report"]["transitions"] if t["success"]}
            for obs in row["observations"]:
                stage = stages.setdefault(obs.stage, {"observations": 0, "choice_observations": 0, "legal": {}, "executed": {}})
                stage["observations"] += 1
                stage["choice_observations"] += len(obs.graph["candidates"]) > 1
                for c in obs.graph["candidates"]:
                    stage["legal"][c["kind"]] = stage["legal"].get(c["kind"],0)+1
                if obs.snapshot in selected:
                    kind=selected[obs.snapshot]["kind"]
                    stage["executed"][kind]=stage["executed"].get(kind,0)+1
        output.append({"program": group["key"]["program"], "paths":len(group["rows"]), "stages":stages,
                       "pair_labels":len(group["pairs"]), "skipped_pairs":group["skipped"]})
    return output


def measured_loss(model, groups, resource_weight):
    losses = {}
    for group in groups:
        policy, values, resources = [], [], []
        for row in group["rows"]:
            report, observations = row["report"], row["observations"]
            state, selected_score = None, torch.tensor(0.0)
            choices = {t["source"]: t["action"] for t in report["transitions"]}
            for observation in observations:
                (scores, value, resource), state = model(observation, state)
                chosen = choices.get(observation.snapshot, {"kind": "advance", "target": 0, "parameters": {}})
                index = next(i for i,c in enumerate(observation.graph["candidates"])
                             if all(c[k]==chosen[k] for k in ("kind","target","parameters")))
                selected_score = selected_score + scores.log_softmax(0)[index]
            target = math.log(row["latency"] / group["baseline"])
            compile_s = sum(e["seconds"] for e in report["events"] if e["stage"] in
                            ("import","observe","apply","lower-S0","lower-S1","lower-S2","translate","codegen","link"))
            policy.append(selected_score)
            values.append((value-target).square())
            resources.append((resource-torch.tensor([math.log1p(report["measurement"]["code_bytes"])/10,
                                                    math.log1p(compile_s)/10])).square().mean())
        pairs=[torch.nn.functional.softplus(direction*(policy[i]-policy[j])) for i,j,direction in group["pairs"]]
        ranking=torch.stack(pairs).mean() if pairs else torch.tensor(0.0)
        losses.setdefault(group["key"]["program"], []).append(ranking+torch.stack(values).mean()+resource_weight*torch.stack(resources).mean())
    # Inputs/groups average inside each program, then programs get equal weight.
    return torch.stack([torch.stack(rows).mean() for rows in losses.values()]).mean()


def fit_measured(model, train_paths, validation_paths, config):
    train_groups = comparison_groups(train_paths, config)
    validation_groups = comparison_groups(validation_paths, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    curves, best, best_state = [], math.inf, None
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = measured_loss(model, train_groups, config["resource_weight"])
        loss.backward()
        from .encoder import ACTIONS
        gradients = {k: float(model.heads.actions.weight.grad[i].abs().sum()) for k,i in ACTIONS.items()}
        stage_gradients = {name: float(sum((p.grad.abs().sum() for p in block.parameters() if p.grad is not None), torch.tensor(0.0)))
                           for name,block in (([(f"S{i}",block) for i,block in enumerate(model.blocks)]) if hasattr(model,"blocks") else [("shared",model.block)])}
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(measured_loss(model, validation_groups, config["resource_weight"]))
        curves.append({"epoch": epoch, "train": float(loss.detach()), "validation": validation, "action_gradient_l1": gradients, "block_gradient_l1": stage_gradients})
        if validation < best:
            best, best_state = validation, {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.costs.clear()
    model.load_state_dict(best_state)
    return {"objective": config["objective"], "curves": curves, "selected_validation_loss": best,
            "selection": "minimum validation loss across the prelocked epochs; test family never loaded",
            "train_group_count": len(train_groups), "validation_group_count": len(validation_groups)}
