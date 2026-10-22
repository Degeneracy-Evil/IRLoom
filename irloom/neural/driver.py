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
    def __init__(self, name, context="single", size=32):
        self.output = CACHE / "runs" / name
        self.logs = ROOT / "logs/architecture" / name
        self.output.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events, self.observations = [], []
        self.history, self.transitions, self.decisions = [], [], []
        self.state, self.current = None, None
        self.remaining = MANIFEST["budget"]["attempts"]
        self.branch, self.serial = "root", 0
        self.hardware = probe()
        self.cpu = min(os.sched_getaffinity(0))
        source, self.defines = generate(context, (size, size))
        self.ir = self.output / "input.mlir"
        self.ir.write_text(source)
        save(self.output / "environment.json", self.hardware)
        save(self.output / "manifest.json", MANIFEST)

    def run(self, args, label):
        return command(args, self.logs / f"{len(self.events):03}-{label}.log",
                       self.events, label, timeout=MANIFEST["budget"]["command_timeout_s"], cpu=self.cpu)

    def observe(self, stage):
        snapshot = f"{self.branch}-{self.serial}"
        self.serial += 1
        graph_path = self.output / f"{snapshot}.json"
        self.run([CACHE / "build/irloom-bridge", self.ir, graph_path, snapshot, self.branch, stage], "observe")
        graph = json.loads(graph_path.read_text())
        for candidate in graph["candidates"]:
            candidate["snapshot"] = snapshot
        if len(graph["nodes"]) > MANIFEST["budget"]["max_graph_nodes"]:
            raise RuntimeError("graph budget exceeded; refusing to truncate")
        previous = self.current.graph if self.current else None
        obs = Observation(1, MANIFEST["id"], stage, snapshot, self.branch, graph, self.hardware,
                          align(previous, graph), copy.deepcopy(self.history))
        self.current = obs
        self.observations.append(asdict(obs))
        return obs

    def checkpoint(self):
        import torch
        return Checkpoint(self.ir, copy.deepcopy(self.current), clone_state(self.state),
                          copy.deepcopy(self.history), torch.get_rng_state().clone(), self.remaining)

    def restore(self, checkpoint, branch=None):
        import torch
        # Rollback cannot refund failed attempts. Explicit forks get their own
        # budget only through the searcher's common accounting (not implemented).
        self.ir, self.current = checkpoint.ir, copy.deepcopy(checkpoint.observation)
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
        output = self.output / f"{suffix}.mlir"
        graph_path = self.output / f"{suffix}.json"
        request = self.output / f"{suffix}-action.json"
        save(request, {**action, "inject_failure": inject_failure})
        success, diagnostic, lineage = False, "", {}
        try:
            # The subprocess parses a private module. The checkpoint file is
            # never its output, even when mutation or verification fails.
            self.run([CACHE / "build/irloom-bridge", self.ir, graph_path, source,
                      self.branch, self.current.stage, request, output], "apply")
            self.ir = output
            lineage = json.loads(graph_path.read_text())
            success = True
        except CommandError as exc:
            diagnostic = str(exc)
        transition = Transition(source, suffix if success else source, success, action,
                                {"relations": lineage.get("lineage", []),
                                 "unmapped": "unknown; refresh"}, time.perf_counter() - start, diagnostic)
        self.transitions.append(asdict(transition))
        self.history.append({"stage": self.current.stage, "kind": action["kind"], "success": success})
        return transition

    def advance(self, stage):
        path = self.output / f"after-{stage['id']}.mlir"
        self.run([SDK / "bin/mlir-opt", self.ir, *stage["lowering"], "-o", path], f"lower-{stage['id']}")
        self.ir = path

    def finalize(self):
        ll, obj, exe = (self.output / x for x in ("program.ll", "program.o", "program"))
        self.run([SDK / "bin/mlir-translate", "--mlir-to-llvmir", self.ir, "-o", ll], "translate")
        self.run(["clang-18", *MANIFEST["backend"], "-c", ll, "-o", obj], "codegen")
        self.run(["clang-18", "-O2", "-ffp-contract=off", *self.defines,
                  ROOT / "benchmarks/runner.c", obj, "-lm", "-o", exe], "link")
        checks = [json.loads(self.run([exe, "check", str(seed), "1"], "check")) for seed in (0, 1, 17)]
        samples = [json.loads(self.run([exe, "measure", "19", "2000"], "measure")) for _ in range(5)]
        measurement = TerminalMeasurement(checks, samples, obj.stat().st_size,
                                           {"commands_s": sum(e["seconds"] for e in self.events)})
        report = {"observations": self.observations, "events": self.events, "measurement": asdict(measurement),
                  "transitions": self.transitions, "decisions": self.decisions, "remaining_attempts": self.remaining,
                  "median_latency_ms": statistics.median(x["latency_ms"] for x in samples),
                  "scope": "native integration smoke; no general-program performance claim"}
        save(self.output / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=f"smoke-{time.time_ns()}")
    parser.add_argument("--context", choices=("single", "transpose", "row_reduce"), default="single")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--audit", action="store_true", help="N1 integration rejection/rollback/replay; scripted actions, not a model")
    parser.add_argument("--model", choices=("default", "A"), default="default")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--fit-report", type=Path, help="integration fit only; no held-out performance claim")
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    if args.audit and args.model != "default":
        parser.error("scripted audit and neural selection are separate modes")
    session = Session(args.name, args.context, args.size)
    model, metadata = None, {}
    if args.model != "default":
        start = time.perf_counter()
        import torch
        from .interleaved import Interleaved
        from .training import fit_smoke
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        config = {"hidden": args.hidden, "heads": args.heads, "history": not args.no_history}
        checkpoint = torch.load(args.weights.resolve(), weights_only=True) if args.weights else None
        if checkpoint:
            if checkpoint["model"] != args.model:
                raise ValueError("checkpoint architecture mismatch")
            config = {**checkpoint["config"], "history": not args.no_history}
        model = Interleaved(**config)
        if checkpoint:
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
    audit = {}
    for stage in MANIFEST["stages"]:
        observation = session.observe(stage["id"])
        if model is not None:
            if len(observation.graph["nodes"]) * model.hidden * 4 * 8 > MANIFEST["budget"]["max_state_bytes"]:
                raise RuntimeError("neural state memory budget exceeded")
            with torch.no_grad():
                (scores, value, resources), session.state = model(observation, session.state)
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
        save(session.output / "model.json", metadata)
    if args.audit:
        save(session.output / "audit.json", audit)
    print(json.dumps({"report": str(session.output / "report.json"), "correct": True,
                      "median_latency_ms": report["median_latency_ms"]}))


if __name__ == "__main__":
    main()
