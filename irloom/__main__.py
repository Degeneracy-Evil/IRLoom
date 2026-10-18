"""One entry point for probing, compilation, measurement and collection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .analysis import save_summary
from .compiler import ROOT, CommandError, compile_candidate, compile_reference, probe
from .measurement import fail, measure
from benchmarks.workloads import generate


def candidates_for(config, case):
    return config.get("candidates_by_program", {}).get(case["program"], config.get("candidates", []))


def load_config(path):
    config = json.loads(path.read_text())
    if config["samples"] < 10 or config["min_batch_ms"] <= 0 or config["timeout_seconds"] <= 0:
        raise ValueError("need >=10 samples, positive calibration time and timeout")
    if not config["correctness_seeds"]:
        raise ValueError("correctness_seeds must not be empty")
    groups = [config["cases"]]+[candidates_for(config, c) for c in config["cases"]]
    for group in groups:
        ids = [entry["id"] for entry in group]
        if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", x) for x in ids):
            raise ValueError("case/candidate ids must be unique, nonempty path-safe identifiers")
    for case in config["cases"]:
        dims = 3 if case["program"] == "matmul" else 2
        if (case["program"] not in ("matmul", "reduction", "stencil") or len(case["shape"]) != dims
                or any(type(x) is not int or x <= 0 for x in case["shape"])):
            raise ValueError(f"invalid program or positive static shape: {case}")
        allowed = {"matmul": (None,"ijk","ikj"), "reduction":(None,"ij","ji"), "stencil":(None,)}
        if any(c.get("high_choice") not in allowed[case["program"]] for c in candidates_for(config,case)):
            raise ValueError("high choice incompatible with workload")
        if "iterations" in case and (type(case["iterations"]) is not int or case["iterations"]<1):
            raise ValueError("fixed iterations must be a positive integer")
    return config


def save_records(records, path):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def compile_all(config, output, logs, metadata):
    records = []
    for case in config["cases"]:
        for candidate in candidates_for(config, case):
            record = {"schema_version": 1, "case_id": case["id"], "program_id": case["program"],
                      "candidate_id": candidate["id"], "seed": config["seed"],
                      "input": {"shape": case["shape"], "dtype": "f32"},
                      "metadata": metadata, "events": [], "timings_ms": [],
                      "fingerprints": "disabled by user instruction", "status": "pending",
                      "transforms": candidate["transforms"],
                      "high_choice": candidate.get("high_choice"),
                      "low_choice": candidate.get("low_choice"),
                      "role": candidate.get("role", "factorial")}
            records.append(record)
            dest = output / case["id"] / candidate["id"]
            log = logs / case["id"] / candidate["id"]
            try:
                source, defines = generate(case["program"], case["shape"])
                record["decision_features"] = {"program":case["program"],"shape":case["shape"],
                                               "stage":"high","layout":"row-major",
                                               "operation":next(op for op in ("linalg.matmul","linalg.generic","affine.for") if op in source)}
                if record["role"] == "reference":
                    if case["program"] != "matmul":
                        raise ValueError("C++ reference only supports matmul")
                    compile_reference(dest, log, record, defines)
                else:
                    compile_candidate(source, dest, log, record, candidate["transforms"], defines,
                                      candidate.get("high_choice"))
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
        if baseline and record["role"] != "reference":
            base = Path(baseline["binary"]).parent
            record["ir_changed"] = (dest / "transformed.mlir").read_bytes() != (base / "transformed.mlir").read_bytes()
        prior = [r for r in records[:records.index(record)] if r["case_id"] == record["case_id"] and r["status"] == "compiled"]
        record["duplicate_of"] = next((r["candidate_id"] for r in prior if
                                       Path(r["binary"]).read_bytes() == Path(record["binary"]).read_bytes()), None)
        record["code_duplicate_of"] = next((r["candidate_id"] for r in prior if
                                            Path(r["disassembly"]).read_bytes() ==
                                            Path(record["disassembly"]).read_bytes()), None)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="collect",
                        choices=["probe", "m0", "m4", "m4-report", "m5-a", "m5", "m5-report", "compile", "run", "collect", "summarize", "report"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--order-seed", type=int, help="measurement order only; input seed stays fixed")
    parser.add_argument("--manifest", type=Path, help="compile manifest used by run")
    parser.add_argument("--records", type=Path, nargs="+", help="JSONL inputs for summarize/report")
    parser.add_argument("--workflow", type=Path, help="M4 workflow.json for default-equivalence evidence")
    args = parser.parse_args()
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    if args.action == "m5-a":
        from .m5 import analyze_m4
        print(analyze_m4())
        return
    if args.action == "m5":
        from .m5 import run
        run(args.config or ROOT/"experiments/m5.json")
        return
    if args.action == "m5-report":
        if not args.records:
            parser.error("m5-report requires --records")
        from .m5 import write_report
        print(write_report(args.records))
        return
    if args.action == "m4":
        from .m4 import run
        run(args.config or ROOT / "experiments/m4.json")
        return
    if args.action == "m4-report":
        if not args.records:
            parser.error("m4-report requires --records")
        from .m4 import write_report
        equivalence = json.loads(args.workflow.read_text())["default_equivalence"] if args.workflow else None
        print(write_report(args.records, equivalence=equivalence))
        return
    if args.action == "m0":
        from .collect import main as m0
        m0()
        return
    if args.action == "probe":
        print(json.dumps(probe(), indent=2, ensure_ascii=False))
        return
    if args.action == "report":
        if not args.records:
            parser.error("report requires --records")
        from .report import write_report
        print(write_report(args.records))
        return
    if args.action == "summarize":
        if not args.records:
            parser.error("summarize requires --records")
        if len(args.records) != 1:
            parser.error("summarize accepts one JSONL file; use report for repeated runs")
        records = [json.loads(line) for line in args.records[0].read_text().splitlines()]
        path = args.records[0].parent / "summary.json"
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
        config = load_config(args.config or ROOT / "experiments/phase0.json")
        if args.order_seed is not None:
            config["order_seed"] = args.order_seed
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
