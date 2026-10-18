"""Independent checking and seeded, interleaved native timing."""
import json
from pathlib import Path
import random
import time

from .compiler import CommandError, command


def fail(record, exc):
    event = record["events"][-1] if record["events"] else {}
    record.update(status="failed", failure_stage=event.get("stage", "setup"),
                  error=str(exc))
    if event.get("stage") == "correctness":
        record["correctness"] = {"status": "failed", "error": str(exc)}


def measure(records, config, logs, cpu):
    start = time.perf_counter()
    order_seed = config.get("order_seed", config["seed"])
    rng = random.Random(order_seed)
    active = [r for r in records if r["status"] == "compiled"]
    rng.shuffle(active)
    counts = {}
    for record in active:
        record["timings_ms"] = []
        record["measurement"] = {"cpu": cpu, "threads": 1, "warmups": 3,
                                  "clock": "CLOCK_MONOTONIC", "unit": "ms/kernel",
                                  "samples": config["samples"], "order_seed": order_seed,
                                  "min_calibration_batch_ms": config["min_batch_ms"],
                                  "cache_policy": "same buffers, warm cache",
                                  "timeout_seconds": config["timeout_seconds"]}
        dest = logs / record["case_id"] / record["candidate_id"]
        dest.mkdir(parents=True, exist_ok=True)
        try:
            checks = []
            for seed in config["correctness_seeds"]:
                result = command([record["binary"], "check", seed, 1], dest / f"check-{seed}.log",
                                 record["events"], "correctness", config["timeout_seconds"], cpu)
                checks.append(dict(json.loads(result), seed=seed))
            record["correctness"] = {"status": "pass", "rtol": 1e-4, "atol": 1e-4,
                                      "reassociation": False, "checks": checks}
            fixed = next(c.get("iterations") for c in config["cases"] if c["id"]==record["case_id"])
            record["measurement"]["iteration_source"] = "fixed_config" if fixed else "all_candidate_calibration"
            if fixed:
                record["measurement"]["min_calibration_batch_ms"] = None
            iterations = fixed or 1
            if not fixed:
                while True:
                    sample = json.loads(command([record["binary"], "measure", config["seed"], iterations],
                                                dest / f"calibrate-{iterations}.log", record["events"],
                                                "calibration", config["timeout_seconds"], cpu))
                    if sample["elapsed_ms"] >= config["min_batch_ms"]:
                        break
                    iterations *= 2
                    if iterations > 2**30:
                        raise ValueError("calibration iteration limit exceeded")
            counts[record["case_id"]] = max(counts.get(record["case_id"], 0), iterations)
            record["status"] = "ready"
        except (CommandError, OSError, ValueError) as exc:
            fail(record, exc)
    # Same call count for every candidate of a case, chosen from all calibrations.
    schedule = []
    for round_id in range(config["samples"]):
        candidates = [r for r in active if r["status"] in ("ready", "ok")]
        rng.shuffle(candidates)
        for record in candidates:
            case, candidate = record["case_id"], record["candidate_id"]
            iterations = counts[case]
            record["measurement"]["iterations"] = iterations
            log = logs / case / candidate / f"sample-{round_id}.log"
            try:
                sample = json.loads(command([record["binary"], "measure", config["seed"], iterations],
                                            log, record["events"], "timing", config["timeout_seconds"], cpu))
                if not (sample["latency_ms"] > 0 and sample["iterations"] == iterations):
                    raise ValueError("invalid native timing sample")
                record["timings_ms"].append(sample["latency_ms"])
                schedule.append({"round": round_id, "case": case, "candidate": candidate,
                                 "latency_ms": sample["latency_ms"]})
                record["status"] = "ok"
            except (CommandError, OSError, ValueError) as exc:
                fail(record, exc)
    (logs / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    return time.perf_counter() - start
