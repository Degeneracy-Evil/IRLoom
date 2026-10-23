"""Serial module-boundary orchestration; compiler processes never own model state."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

from benchmarks.context_graph import generate
from irloom.compiler import ROOT, CommandError, command, probe
from .schema import Observation, TerminalMeasurement, Transition, Decision
from .lineage import Checkpoint, align, clone_state

CACHE = ROOT / "benchmark-data/architecture"
SDK = CACHE / "sdk/usr/lib/llvm-18"
MANIFEST = json.loads((ROOT / "compiler/pipelines/architecture-v1.json").read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Session:
    def __init__(self, name, context="single", size=32, multi_function=False, program=None):
        self.output = CACHE / "runs" / name
        self.logs = ROOT / "logs/architecture" / name
        self.output.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events, self.observations = [], []
        self.history, self.transitions, self.decisions = [], [], []
        self.path_lineage = []
        self.lineage_cursor = 0
        self.state, self.current = None, None
        self.remaining = MANIFEST["budget"]["attempts"]
        self.branch, self.serial = "root", 0
        self.hardware = probe()
        self.cpu = min(os.sched_getaffinity(0))
        self.manifest = MANIFEST
        self.bridge, self.opt, self.translate = CACHE / "build/irloom-bridge", SDK / "bin/mlir-opt", SDK / "bin/mlir-translate"
        self.program = program
        if program:
            from .programs import prepare
            prepare(self, program)
            save(self.output / "environment.json", self.hardware)
            save(self.output / "manifest.json", self.manifest)
            return
        source, self.defines = generate(context, (size, size))
        if multi_function:
            mem = f"memref<{size}x{size}xf32>"
            source = source.replace("@kernel(", "@compute(")
            source = source.rstrip()[:-1] + f'''  func.func @kernel(%a: {mem}, %b: {mem}, %c: {mem}) attributes {{llvm.emit_c_interface}} {{
    func.call @compute(%a, %b, %c) : ({mem}, {mem}, {mem}) -> ()
    return
  }}
}}
'''
        self.ir = self.output / "input.mlir"
        self.ir.write_text(source)
        save(self.output / "environment.json", self.hardware)
        save(self.output / "manifest.json", MANIFEST)

    def run(self, args, label):
        timeout = self.manifest["budget"]["command_timeout_s"]
        if hasattr(self, "deadline"):
            timeout = min(timeout, self.deadline - time.perf_counter())
            if timeout <= 0:
                raise CommandError("shared optimization deadline exhausted")
        return command(args, self.logs / f"{len(self.events):03}-{label}.log",
                       self.events, label, timeout=timeout, cpu=self.cpu)

    def observe(self, stage):
        snapshot = f"{self.branch}-{self.serial}"
        self.serial += 1
        graph_path = self.output / f"{snapshot}.json"
        self.run([self.bridge, self.ir, graph_path, snapshot, self.branch, stage], "observe")
        graph = json.loads(graph_path.read_text())
        for candidate in graph["candidates"]:
            candidate["snapshot"] = snapshot
        if len(graph["nodes"]) > self.manifest["budget"]["max_graph_nodes"]:
            raise RuntimeError("graph budget exceeded; refusing to truncate")
        previous = self.current.graph if self.current else None
        obs = Observation(1, self.manifest["id"], stage, snapshot, self.branch, graph, self.hardware,
                          align(previous, graph, self.path_lineage[self.lineage_cursor:]), copy.deepcopy(self.history))
        self.lineage_cursor = len(self.path_lineage)
        self.current = obs
        self.observed_ir = self.ir
        self.observations.append(asdict(obs))
        return obs

    def checkpoint(self):
        import torch
        return Checkpoint(self.ir, self.observed_ir, copy.deepcopy(self.current), clone_state(self.state),
                          copy.deepcopy(self.history), torch.get_rng_state().clone(), self.remaining, copy.deepcopy(self.path_lineage), self.lineage_cursor)

    def fork(self, name):
        child = object.__new__(Session)
        child.__dict__ = self.__dict__.copy()
        child.output, child.logs = CACHE / "runs" / name, ROOT / "logs/architecture" / name
        child.output.mkdir(parents=True, exist_ok=False)
        child.logs.mkdir(parents=True, exist_ok=True)
        for field in ("events", "observations", "transitions", "decisions"):
            setattr(child, field, copy.deepcopy(getattr(self, field)))
        child.restore(self.checkpoint(), name)
        save(child.output / "environment.json", child.hardware)
        save(child.output / "manifest.json", child.manifest)
        source_record = self.output / "program-source.json"
        if source_record.exists():
            save(child.output / "program-source.json", json.loads(source_record.read_text()))
        return child

    def restore(self, checkpoint, branch=None):
        import torch
        # Rollback cannot refund failed attempts. Explicit forks get their own
        # budget only through the searcher's common accounting.
        self.ir, self.current = checkpoint.ir, copy.deepcopy(checkpoint.observation)
        self.observed_ir, self.path_lineage = checkpoint.observed_ir, copy.deepcopy(checkpoint.lineage)
        self.lineage_cursor = checkpoint.lineage_cursor
        self.state, self.history = clone_state(checkpoint.state), copy.deepcopy(checkpoint.history)
        self.remaining = min(self.remaining, checkpoint.remaining)
        torch.set_rng_state(checkpoint.rng)
        if branch is not None:
            self.branch = branch

    def apply(self, action, inject_failure=False):
        if self.remaining <= 0:
            raise RuntimeError("attempt budget exhausted")
        self.remaining -= 1
        start = time.perf_counter()
        source = self.current.snapshot
        suffix = f"trial-{len(self.transitions)}"
        destination = f"{self.branch}-{suffix}"
        output = self.output / f"{suffix}.mlir"
        graph_path = self.output / f"{suffix}.json"
        request = self.output / f"{suffix}-action.json"
        save(request, {**action, "inject_failure": inject_failure, "destination_snapshot": destination})
        success, diagnostic, lineage = False, "", {}
        try:
            if self.ir != self.observed_ir:
                raise CommandError("observation invalidated by successful action; observe or restore before reusing targets")
            # The subprocess parses a private module. The checkpoint file is
            # never its output, even when mutation or verification fails.
            self.run([self.bridge, self.ir, graph_path, source,
                      self.branch, self.current.stage, request, output], "apply")
            self.ir = output
            lineage = json.loads(graph_path.read_text())
            success = True
        except CommandError as exc:
            diagnostic = str(exc)
        transition = Transition(source, destination if success else source, success, action,
                                {"relations": lineage.get("lineage", []),
                                 "unmapped": "unknown; refresh"}, time.perf_counter() - start, diagnostic)
        self.transitions.append(asdict(transition))
        if success:
            self.path_lineage.append(asdict(transition))
        self.history.append({"stage": self.current.stage, "kind": action["kind"], "success": success})
        return transition

    def advance(self, stage):
        path = self.output / f"after-{stage['id']}.mlir"
        self.run([self.opt, self.ir, *stage["lowering"], "-o", path], f"lower-{stage['id']}")
        self.path_lineage.append({"event": "lowering", "source_ir": str(self.ir), "destination_ir": str(path),
                                  "status": "unknown internally; next observation re-establishes only supported anchors"})
        self.ir = path

    def finalize(self):
        if self.program:
            from .programs import measure
            measurement = measure(self)
            return self.report(measurement, "complete original source through Polygeist; native process wall time measured by independent C++ parent")
        ll, obj, exe = (self.output / x for x in ("program.ll", "program.o", "program"))
        self.run([SDK / "bin/mlir-translate", "--mlir-to-llvmir", self.ir, "-o", ll], "translate")
        self.run(["clang-18", *MANIFEST["backend"], "-c", ll, "-o", obj], "codegen")
        self.run(["clang-18", "-O2", "-ffp-contract=off", *self.defines,
                  ROOT / "benchmarks/runner.c", obj, "-lm", "-o", exe], "link")
        checks = [json.loads(self.run([exe, "check", str(seed), "1"], "check")) for seed in (0, 1, 17)]
        samples = [json.loads(self.run([exe, "measure", "19", "2000"], "measure")) for _ in range(5)]
        measurement = TerminalMeasurement(checks, samples, obj.stat().st_size,
                                           {"commands_s": sum(e["seconds"] for e in self.events)})
        return self.report(measurement, "native integration smoke; no general-program performance claim")

    def report(self, measurement, scope):
        report = {"observations": self.observations, "events": self.events, "measurement": asdict(measurement),
                  "transitions": self.transitions, "decisions": self.decisions, "remaining_attempts": self.remaining,
                  "median_latency_ms": statistics.median(x["latency_ms"] for x in measurement.samples),
                  "scope": scope, "program": self.program}
        report["final_ir"] = str(self.ir)
        if self.program:
            cpu = json.loads(self.hardware["cpu"])["lscpu"]
            report["comparison_group"] = {"program": self.program, "input": self.recipe["measure_args"],
                "defines": self.recipe.get("defines", []), "pipeline": self.manifest,
                "hardware": {**{k: self.hardware.get(k) for k in ("os", "arch", "cpu_caches", "affinity", "tools", "frontend_version")},
                             "cpu": [item for item in cpu if item["field"].rstrip(":") in ("Model name", "Flags", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)")]}}
        save(self.output / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=f"smoke-{time.time_ns()}")
    parser.add_argument("--context", choices=("single", "transpose", "row_reduce"), default="single")
    parser.add_argument("--size", type=int, default=32)
    from .programs import RECIPES
    parser.add_argument("--program", choices=tuple(RECIPES["programs"]), help="whole-program frontend adapter; prepares matching tools automatically")
    parser.add_argument("--audit", action="store_true", help="N1 integration rejection/rollback/replay; scripted actions, not a model")
    parser.add_argument("--model", choices=("default", "A", "B"), default="default")
    parser.add_argument("--multi-function", action="store_true", help="call-graph integration smoke, not a real-program import")
    parser.add_argument("--incremental", action="store_true", help="B: reuse only proven unchanged base encodings")
    parser.add_argument("--state-audit", action="store_true", help="B: real fork/restore and full-vs-incremental integration checks")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--fit-report", type=Path, help="integration fit only; no held-out performance claim")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--pilot", action="store_true", help="run the checked-in whole-program collection/training/evaluation budget")
    parser.add_argument("--pilot-config", type=Path, help="versioned pilot configuration; default preserves the V1 protocol")
    parser.add_argument("--compare-pilot", type=Path, help="N5 common bounded search using frozen pilot weights")
    parser.add_argument("--profile-inference", action="store_true", help="record cold load and five identical-input steady inference samples")
    parser.add_argument("--encoder-version", choices=("legacy-v1", "structured-v2", "hierarchical-v3"), default="structured-v2")
    parser.add_argument("--structure-audit", action="store_true", help="extract and compare semantic MLIR structure fixtures")
    parser.add_argument("--interleave-pilot", type=Path, help="freeze single/beam binaries and measure in balanced random order")
    parser.add_argument("--search-report", type=Path, help="optional common beam report for frozen interleaved comparison")
    parser.add_argument("--v12", choices=("audit", "guard", "collect", "train", "evaluate", "measure", "all"), help="stage-cost learning and common guarded search")
    parser.add_argument("--v12-config", type=Path, help="versioned V1.2 protocol; defaults to pilot-v1-2.json")
    parser.add_argument("--context-audit", action="store_true", help="compiler-backed callee/body visibility fixture")
    parser.add_argument("--objective-audit", action="store_true", help="log-cost feasibility and real-label calibration audit")
    parser.add_argument("--v13", choices=("guard", "collect", "train", "evaluate", "measure", "all"), help="hierarchical context and calibrated q pilot")
    parser.add_argument("--v13-config", type=Path)
    parser.add_argument("--training-review", choices=("audit", "train", "all"), help="V1.3.1 frozen-data fixed-final training acceptance")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--campaign", choices=("probe", "collect", "diagnose", "train", "evaluate", "all"), help="frozen multi-program corpus and bounded A/B training campaign")
    parser.add_argument("--campaign-config", type=Path)
    args = parser.parse_args()
    if args.campaign:
        from .campaign import run
        run(args)
        return
    if args.training_review:
        from .training_review import run
        run(args)
        return
    if args.objective_audit:
        from .cost_objective import audit
        audit(args.name)
        return
    if args.v13:
        args.v12=args.v13
        args.v12_config=args.v13_config or ROOT/'experiments/architecture/pilot-v1-3.json'
    if args.context_audit:
        from .context_audit import audit
        audit(args.name)
        return
    if args.v12:
        from .revision import run
        run(args)
        return
    if args.structure_audit:
        from .audit import structure_audit
        structure_audit(args.name)
        return
    if args.interleave_pilot:
        from .measurement import interleave
        interleave(args.name, args.interleave_pilot.resolve(), args.search_report.resolve() if args.search_report else None)
        return
    if args.pilot:
        from .pilot import run
        run(args.name, json.loads(args.pilot_config.resolve().read_text()) if args.pilot_config else None)
        return
    if args.compare_pilot:
        from .search import compare
        compare(args.name, args.compare_pilot.resolve(), json.loads(args.pilot_config.resolve().read_text()) if args.pilot_config else None)
        return
    if args.state_audit and args.model != "B":
        parser.error("state audit requires B")
    if args.audit and args.model != "default":
        parser.error("scripted audit and neural selection are separate modes")
    run_started = time.perf_counter()
    session = Session(args.name, args.context, args.size, args.multi_function, args.program)
    model, metadata = None, {}
    if args.model != "default":
        start = time.perf_counter()
        import torch
        from .interleaved import Interleaved
        from .recurrent import Recurrent
        from .training import fit_smoke
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        config = {"hidden": args.hidden, "heads": args.heads, "history": not args.no_history, "encoder_version": args.encoder_version}
        checkpoint = torch.load(args.weights.resolve(), weights_only=True) if args.weights else None
        if checkpoint:
            if checkpoint["model"] != args.model:
                raise ValueError("checkpoint architecture mismatch")
            config = {**checkpoint["config"], "history": checkpoint["config"]["history"] and not args.no_history}
        if args.model == "B":
            config["incremental"] = args.incremental
        model = (Interleaved if args.model == "A" else Recurrent)(**config)
        if checkpoint:
            # Earlier N2/N3 weights predate the frontend-only inline action.
            # Existing rows are preserved; the new row uses the logged seed.
            key = "heads.actions.weight"
            if checkpoint["state_dict"][key].shape[0] == 3:
                checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], model.heads.actions.weight[3:].detach()])
            model.load_state_dict(checkpoint["state_dict"])
        metadata = {"architecture": args.model, "config": config, "seed": args.seed,
                    "parameters": sum(p.numel() for p in model.parameters()), "load_s": time.perf_counter() - start}
        if args.fit_report:
            start = time.perf_counter()
            metadata["integration_fit"] = fit_smoke(model, args.fit_report.resolve(), args.steps)
            metadata["training_s"] = time.perf_counter() - start
        torch.save({"model": args.model, "config": config, "state_dict": model.state_dict()}, session.output / "weights.pt")
        model.eval()
        model.costs.clear()
        if args.state_audit:
            from .audit import state_audit
            metadata["state_audit"] = state_audit(model, args.name + "-audit", args.context, args.size)
            model.incremental = args.incremental
            model.costs.clear()
    audit = {}
    for stage in session.manifest["stages"]:
        observation = session.observe(stage["id"])
        if model is not None:
            if len(observation.graph["nodes"]) * model.hidden * 4 * 8 > MANIFEST["budget"]["max_state_bytes"]:
                raise RuntimeError("neural state memory budget exceeded")
            with torch.no_grad():
                previous = session.state
                (scores, value, resources), session.state = model(observation, previous)
                if args.profile_inference and stage["id"] == "S0":
                    samples = []
                    for _ in range(5):
                        start = time.perf_counter()
                        replay, _ = model(observation, previous)
                        elapsed = time.perf_counter() - start
                        for actual, expected in zip(replay, (scores, value, resources)):
                            assert torch.equal(actual, expected)
                        samples.append({"wall_s": elapsed, **model.costs.pop()})
                    metadata["steady_inference"] = {"scope": "five identical S0 inputs after first call; output equality checked",
                                                      "samples": samples, "median_s": statistics.median(x["wall_s"] for x in samples)}
            if not torch.isfinite(scores).all():
                raise RuntimeError("non-finite neural ranking")
            ranking = scores.argsort(descending=True, stable=True).tolist()
            selected = observation.graph["candidates"][ranking[0]]
            session.decisions.append(asdict(Decision(observation.snapshot, ranking, selected, scores.tolist(),
                                                     value.item(), resources.tolist())))
            session.apply(selected)
        if args.audit:
            candidates = observation.graph["candidates"]
            selected = candidates[1] if len(candidates) > 1 else candidates[0]
            checkpoint = session.checkpoint()
            before = session.ir.read_bytes()
            if stage["id"] == "S0":
                illegal = {**selected, "target": -1}
                assert not session.apply(illegal).success
                assert session.ir.read_bytes() == before
                assert not session.apply(selected, inject_failure=True).success
                assert session.ir.read_bytes() == before
                audit["illegal_rejected_and_mutation_rolled_back"] = True
            assert session.apply(selected).success
            changed = session.ir.read_bytes()
            if stage["id"] == "S0":
                session.restore(checkpoint)
                assert session.apply(selected).success
                assert session.ir.read_bytes() == changed
                audit["replay_identical"] = True
            if selected["kind"] != "advance":
                assert changed != before
                audit[stage["id"]] = selected["kind"]
        session.advance(stage)
    report = session.finalize()
    if model is not None:
        metadata["costs"] = model.costs
        metadata["driver_wall_s"] = time.perf_counter() - run_started
        save(session.output / "model.json", metadata)
    if args.audit:
        save(session.output / "audit.json", audit)
    print(json.dumps({"report": str(session.output / "report.json"), "correct": True,
                      "median_latency_ms": report["median_latency_ms"]}))


if __name__ == "__main__":
    main()
