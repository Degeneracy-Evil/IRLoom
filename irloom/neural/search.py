"""Shared bounded beam over immutable IR + private neural-state checkpoints."""
from dataclasses import asdict
import json
import math
import resource
import time

import torch

from .driver import CACHE, Session, save
from .schema import Decision
from .pilot import CONFIG
from .interleaved import Interleaved
from .recurrent import Recurrent
from irloom.compiler import CommandError


def beam(name, program, model, budget):
    start = time.perf_counter()
    root = Session(name + "-root", program=program)
    root.deadline = start + budget["optimization_timeout_s"]
    active, sessions = [(0.0, root)], [root]
    attempts, serial = 0, 0
    model.eval(); model.costs.clear()
    for stage in root.manifest["stages"]:
        expanded = []
        for score, parent in active:
            observation = parent.observe(stage["id"])
            with torch.no_grad():
                (scores, value, resources), parent.state = model(observation, parent.state)
            logp = scores.log_softmax(0)
            ranking = scores.argsort(descending=True, stable=True).tolist()
            accepted = 0
            for index in ranking:
                if attempts >= budget["max_action_attempts"] or time.perf_counter() - start >= budget["optimization_timeout_s"]:
                    break
                selected = observation.graph["candidates"][index]
                child = parent.fork(f"{name}-branch-{serial}")
                serial += 1
                sessions.append(child)
                child.decisions.append(asdict(Decision(observation.snapshot, ranking, selected, scores.tolist(), value.item(), resources.tolist())))
                attempts += 1
                transition = child.apply(selected)
                if not transition.success:
                    # A failed mutation cannot enter the candidate ranking.
                    save(child.output / "rejected.json", asdict(transition))
                    continue
                try:
                    child.advance(stage)
                except CommandError as error:
                    save(child.output / "rejected-lowering.json", {"error": str(error), "events": child.events})
                    continue
                expanded.append((score + float(logp[index]), child))
                accepted += 1
                if accepted == budget["beam_width"]:
                    break
        active = sorted(expanded, key=lambda item: item[0], reverse=True)[:budget["beam_width"]]
        if not active:
            raise RuntimeError("common search budget produced no valid continuation")
    terminals = []
    for score, candidate in active[:budget["max_terminal_candidates"]]:
        # Every subprocess timeout is capped by the remaining shared wall budget.
        if time.perf_counter() - start >= budget["optimization_timeout_s"]:
            break
        try:
            report = candidate.finalize()
        except (CommandError, RuntimeError) as error:
            save(candidate.output / "rejected-terminal.json", {"error": str(error), "events": candidate.events})
            continue
        terminals.append({"score": score, "report": str(candidate.output / "report.json"),
                          "latency_ms": report["median_latency_ms"]})
    if not terminals:
        raise RuntimeError("no terminal candidate completed within the admission budget")
    events = {e["log"]: e for session in sessions for e in session.events}
    elapsed = time.perf_counter() - start
    result = {"program": program, "budget": budget, "attempts": attempts, "terminals": terminals,
              "selected": min(terminals, key=lambda x: x["latency_ms"]), "wall_s": elapsed,
              "command_s": sum(e["seconds"] for e in events.values()), "events": list(events.values()),
              "feature_s": sum(e["seconds"] for e in events.values() if e["stage"] == "observe"),
              "model_costs": list(model.costs), "deadline_overrun_s": max(0, elapsed - budget["optimization_timeout_s"]),
              "measurement_invocations": sum(e["stage"] == "measure" for e in events.values()),
              "parent_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "children_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    save(root.output / "search.json", result)
    return result


def summarize(single, searches):
    rows = {}
    for variant, programs in single.items():
        ratios, detail = [], {}
        for program, path in programs.items():
            baseline = json.loads(open(single["default"][program]).read())
            measured = json.loads(open(path).read())
            ratio = baseline["median_latency_ms"] / measured["median_latency_ms"]
            ratios.append(ratio)
            detail[program] = {"speedup": ratio, "latency_ms": measured["median_latency_ms"],
                               "raw_samples_ms": [x["latency_ms"] for x in measured["measurement"]["samples"]],
                               "code_bytes": measured["measurement"]["code_bytes"],
                               "command_s": measured["measurement"]["costs"]["commands_s"],
                               "execution": measured.get("execution", {})}
        rows[variant] = {"geomean_speedup": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
                         "worst_speedup": min(ratios), "programs": detail}
    beam_rows = {}
    for variant, programs in searches.items():
        ratios = []
        for program, search in programs.items():
            base = json.loads(open(single["default"][program]).read())["median_latency_ms"]
            ratios.append(base / search["selected"]["latency_ms"])
        beam_rows[variant] = {"geomean_speedup": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
                              "worst_speedup": min(ratios)}
    return {"single": rows, "beam": beam_rows,
            "interpretation": "Exploratory single-machine process-wall ratios; five repeats, no frequency control, serial method order, no confidence interval. Small changes are not evidence of speedup. Beam uses sampled online measurements, not an offline all-pool oracle."}


def compare(name, pilot_dir):
    torch.set_num_threads(1)
    pilot = json.loads((pilot_dir / "report.json").read_text())
    output = CACHE / "comparisons" / name
    output.mkdir(parents=True, exist_ok=False)
    result = {"source_pilot": str(pilot_dir), "config": CONFIG, "models": {}, "searches": {}}
    for variant in CONFIG["evaluation"]["variants"]:
        if variant == "default":
            continue
        key = "A" if variant == "A_no_history" else "B" if variant == "B_reset" else variant
        start = time.perf_counter()
        checkpoint = torch.load(pilot_dir / f"{key}.pt", weights_only=True)
        config = {**checkpoint["config"], "history": variant not in ("current_only", "A_no_history", "B_reset")}
        model = (Interleaved if checkpoint["model"] == "A" else Recurrent)(**config)
        model.load_state_dict(checkpoint["state_dict"])
        result["models"][variant] = {"config": config, "parameters": sum(p.numel() for p in model.parameters()),
                                       "load_after_torch_import_s": time.perf_counter() - start}
        result["searches"][variant] = {}
        for program in CONFIG["programs"]:
            search = beam(f"{name}-{variant}-{program}", program, model, CONFIG["evaluation"])
            result["searches"][variant][program] = search
            save(output / "progress.json", result)
            print(json.dumps({"phase": "beam", "variant": variant, "program": program, "terminals": len(search["terminals"])}), flush=True)
    result["summary"] = summarize(pilot["single_path"], result["searches"])
    result["training"] = pilot["training"]
    result["label_acquisition_s"] = pilot["label_acquisition_s"]
    save(output / "report.json", result)
    print(json.dumps({"comparison": str(output / "report.json")}), flush=True)
