"""One entry point for probing, compilation, measurement and collection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .analysis import save_summary
from .compiler import ROOT, CommandError, compile_candidate, probe
from .measurement import fail, measure


def load_config(path):
    config = json.loads(path.read_text())
    if config["samples"] < 10 or config["min_batch_ms"] <= 0 or config["timeout_seconds"] <= 0:
        raise ValueError("need >=10 samples, positive calibration time and timeout")
    if not config["correctness_seeds"]:
        raise ValueError("correctness_seeds must not be empty")
    for group in (config["cases"], config["candidates"]):
        ids = [entry["id"] for entry in group]
        if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", x) for x in ids):
            raise ValueError("case/candidate ids must be unique, nonempty path-safe identifiers")
    for case in config["cases"]:
        if case["program"] != "matmul" or case["shape"] != [32, 32, 32]:
            raise ValueError("M1 currently supports matmul [32,32,32]")
    return config


def save_records(records, path):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def compile_all(config, output, logs, metadata):
    records = []
    for case in config["cases"]:
        for candidate in config["candidates"]:
            record = {"schema_version": 1, "case_id": case["id"], "program_id": case["program"],
                      "candidate_id": candidate["id"], "seed": config["seed"],
                      "input": {"shape": case["shape"], "dtype": "f32"},
                      "metadata": metadata, "events": [], "timings_ms": [],
                      "fingerprints": "disabled by user instruction", "status": "pending",
                      "transforms": candidate["transforms"]}
            records.append(record)
            dest = output / case["id"] / candidate["id"]
            log = logs / case["id"] / candidate["id"]
            try:
                compile_candidate((ROOT / "benchmarks/matmul.mlir").read_text(), dest, log,
                                  record, candidate["transforms"])
                record["status"] = "compiled"
            except (CommandError, OSError, ValueError) as exc:
                record["compile_seconds"] = sum(e["seconds"] for e in record["events"])
                fail(record, exc)
    # Compare actual artifacts directly (no hashing). IR difference is not code difference.
    for record in records:
        if record["status"] != "compiled":
            continue
        dest = Path(record["binary"]).parent
        baseline = next((r for r in records if r["case_id"] == record["case_id"] and
                         r["candidate_id"] == "default" and r["status"] == "compiled"), None)
        if baseline:
            base = Path(baseline["binary"]).parent
            record["ir_changed"] = (dest / "transformed.mlir").read_bytes() != (base / "transformed.mlir").read_bytes()
        prior = [r for r in records[:records.index(record)] if r["case_id"] == record["case_id"] and r["status"] == "compiled"]
        record["duplicate_of"] = next((r["candidate_id"] for r in prior if
                                       Path(r["binary"]).read_bytes() == Path(record["binary"]).read_bytes()), None)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="collect",
                        choices=["probe", "m0", "compile", "run", "collect", "summarize"])
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/m1.json")
    parser.add_argument("--manifest", type=Path, help="compile manifest used by run")
    parser.add_argument("--records", type=Path, help="JSONL input used by summarize")
    args = parser.parse_args()
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    if args.action == "m0":
        from .collect import main as m0
        m0()
        return
    if args.action == "probe":
        print(json.dumps(probe(), indent=2, ensure_ascii=False))
        return
    if args.action == "summarize":
        if not args.records:
            parser.error("summarize requires --records")
        records = [json.loads(line) for line in args.records.read_text().splitlines()]
        path = args.records.parent / "summary.json"
        print(json.dumps(save_summary(records, path), indent=2, ensure_ascii=False))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = ROOT / "logs" / stamp
    logs.mkdir(parents=True)
    started = time.perf_counter()
    if args.action == "run":
        if not args.manifest:
            parser.error("run requires --manifest")
        manifest = json.loads(args.manifest.read_text())
        config, records = manifest["config"], manifest["records"]
    else:
        config = load_config(args.config)
        records = compile_all(config, ROOT / "tmp" / stamp, logs, probe())
        manifest = {"config": config, "records": records}
        (logs / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if args.action != "compile":
        metadata = probe()
        cpu = config.get("cpu", metadata["affinity"][0])
        if cpu not in metadata["affinity"]:
            raise ValueError(f"CPU {cpu} outside allowed affinity")
        for r in records:
            r["measurement_host"] = metadata
        measurement_seconds = measure(records, config, logs, cpu)
    else:
        measurement_seconds = 0
    save_records(records, logs / "records.jsonl")
    save_summary(records, logs / "summary.json")
    (logs / "run.json").write_text(json.dumps({"config": config, "wall_seconds": time.perf_counter()-started,
                                               "measurement_seconds": measurement_seconds,
                                               "records": str(logs / "records.jsonl")}, indent=2) + "\n")
    print(f"records: {logs / 'records.jsonl'}\nsummary: {logs / 'summary.json'}")
    print(f"status: {sum(r['status'] in ('compiled', 'ok') for r in records)}/{len(records)} succeeded")
    if any(r["status"] == "failed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
