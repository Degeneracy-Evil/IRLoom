"""Bounded real-program pilot. Families and budgets are fixed in checked-in JSON."""
from dataclasses import asdict
import json
import random
import resource
import time

import torch

from .driver import CACHE, ROOT, Session, save
from .interleaved import Interleaved
from .recurrent import Recurrent
from .schema import Decision
from .programs import RECIPES
from .training import fit_measured

CONFIG = json.loads((ROOT / "experiments/architecture/pilot.json").read_text())


def run_path(name, program, policy="default", model=None, seed=23):
    start = time.perf_counter()
    session = Session(name, program=program)
    rng = random.Random(seed)
    if model:
        model.eval(); model.costs.clear()
    for stage in session.manifest["stages"]:
        observation = session.observe(stage["id"])
        candidates = observation.graph["candidates"]
        if model:
            with torch.no_grad():
                (scores, value, resources), session.state = model(observation, session.state)
            ranking = scores.argsort(descending=True, stable=True).tolist()
            selected = candidates[ranking[0]]
            decision = Decision(observation.snapshot, ranking, selected, scores.tolist(), value.item(), resources.tolist())
        else:
            index = 0 if policy == "default" else min(1, len(candidates) - 1) if policy == "balanced" else rng.randrange(len(candidates))
            selected = candidates[index]
            decision = Decision(observation.snapshot, [index], selected, [])
        session.decisions.append(asdict(decision))
        if model or policy != "default":
            session.apply(selected)
        session.advance(stage)
    result = session.finalize()
    result["execution"] = {"policy": policy, "wall_s": time.perf_counter() - start,
                           "model_costs": list(model.costs) if model else [],
                           "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    save(session.output / "report.json", result)
    return session.output / "report.json"


def run(name):
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    output = CACHE / "pilots" / name
    output.mkdir(parents=True, exist_ok=False)
    save(output / "config.json", CONFIG)
    save(output / "programs.json", RECIPES)
    collected, models, result = {}, {}, {"config": CONFIG, "collection": {}, "training": {}, "single_path": {}}
    start = time.perf_counter()
    for program in CONFIG["programs"]:
        split = RECIPES["programs"][program]["split"]
        policies = CONFIG["collection"]["test_paths_before_fit" if split == "test" else "train_and_validation_paths"]
        collected[program] = []
        for index, policy in enumerate(policies):
            path = run_path(f"{name}-data-{program}-{policy}", program, policy, seed=CONFIG["seed"] + index)
            collected[program].append(path)
            print(json.dumps({"phase": "collect", "program": program, "policy": policy, "report": str(path)}), flush=True)
        result["collection"][program] = [str(p) for p in collected[program]]
        save(output / "progress.json", result)
    result["label_acquisition_s"] = time.perf_counter() - start
    train = next(p for p in CONFIG["programs"] if RECIPES["programs"][p]["split"] == "train")
    validation = next(p for p in CONFIG["programs"] if RECIPES["programs"][p]["split"] == "validation")
    for label, config in CONFIG["models"].items():
        torch.manual_seed(CONFIG["seed"])
        kind = "B" if label.startswith("B") else "A"
        model = (Recurrent if kind == "B" else Interleaved)(**config)
        start = time.perf_counter()
        fitting = fit_measured(model, collected[train], collected[validation], CONFIG["training"])
        fitting.update(seconds=time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters()))
        result["training"][label] = fitting
        models[label] = model
        torch.save({"model": kind, "config": {"history": True, **config}, "state_dict": model.state_dict()}, output / f"{label}.pt")
        print(json.dumps({"phase": "trained", "model": label, "seconds": fitting["seconds"]}), flush=True)
        save(output / "progress.json", result)
    # All weights are now frozen, including capacity and current-only models.
    for variant in CONFIG["evaluation"]["variants"]:
        result["single_path"][variant] = {}
        key = "A" if variant == "A_no_history" else "B" if variant == "B_reset" else variant
        model = None if variant == "default" else models[key]
        if model:
            model.history = variant not in ("A_no_history", "B_reset", "current_only")
        for program in CONFIG["programs"]:
            path = collected[program][0] if variant == "default" else run_path(f"{name}-single-{variant}-{program}", program, variant, model)
            result["single_path"][variant][program] = str(path)
            print(json.dumps({"phase": "single-path", "variant": variant, "program": program}), flush=True)
            save(output / "progress.json", result)
    result["scope"] = "N4 small measured pilot; three complete program families; N5 bounded search still separate"
    save(output / "report.json", result)
    print(json.dumps({"pilot": str(output / "report.json")}), flush=True)
    return result
