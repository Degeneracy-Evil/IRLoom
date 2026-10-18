"""Minimal end-to-end M0 entry point."""
import json
from pathlib import Path
import statistics
import sys

from .compiler import ROOT, CommandError, command, compile_candidate, probe


def main():
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    metadata = probe()
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return
    output = ROOT / "tmp/m0"
    logs = ROOT / "logs/m0"
    logs.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "case_id": "matmul-32x32x32", "candidate_id": "default",
              "seed": 1, "input": {"shape": [32, 32, 32], "dtype": "f32"},
              "metadata": metadata, "events": [], "timings_ms": [],
              "fingerprints": "disabled by user instruction", "status": "pending"}
    try:
        binary = compile_candidate((ROOT / "benchmarks/matmul.mlir").read_text(), output, logs, record)
        cpu = metadata["affinity"][0]
        record["measurement"] = {"cpu": cpu, "threads": 1, "warmups": 3,
                                  "clock": "CLOCK_MONOTONIC", "unit": "ms/kernel"}
        result = command([binary, "check", "1", "1"], logs / "check.log", record["events"], "correctness", cpu=cpu)
        record["correctness"] = dict(json.loads(result), rtol=1e-4, atol=1e-4, reassociation=False)
        iterations = 1
        while True:
            sample = json.loads(command([binary, "measure", "1", iterations], logs / f"calibrate-{iterations}.log",
                                        record["events"], "calibration", cpu=cpu))
            if sample["elapsed_ms"] >= 10:
                break
            iterations *= 2
        record["measurement"]["iterations"] = iterations
        for i in range(10):
            sample = json.loads(command([binary, "measure", "1", iterations], logs / f"sample-{i}.log",
                                        record["events"], "timing", cpu=cpu))
            record["timings_ms"].append(sample["latency_ms"])
        record.update(status="ok", median_ms=statistics.median(record["timings_ms"]))
    except (CommandError, OSError, ValueError) as exc:
        record.update(status="failed", error=str(exc))
    (logs / "records.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "case_id", "timings_ms")}, indent=2))
    if record["status"] != "ok":
        raise SystemExit(record["error"])


if __name__ == "__main__":
    main()
