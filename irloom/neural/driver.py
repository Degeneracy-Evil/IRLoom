"""Serial module-boundary orchestration; compiler processes never own model state."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

from benchmarks.context_graph import generate
from irloom.compiler import ROOT, command, probe
from .schema import Observation, TerminalMeasurement

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
        if len(graph["nodes"]) > MANIFEST["budget"]["max_graph_nodes"]:
            raise RuntimeError("graph budget exceeded; refusing to truncate")
        obs = Observation(1, MANIFEST["id"], stage, snapshot, self.branch, graph, self.hardware,
                          {"status": "initial" if not self.observations else "unknown", "refresh": "all internal nodes"}, [])
        self.observations.append(asdict(obs))
        return obs

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
                  "median_latency_ms": statistics.median(x["latency_ms"] for x in samples),
                  "scope": "native integration smoke; no general-program performance claim"}
        save(self.output / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=f"smoke-{time.time_ns()}")
    parser.add_argument("--context", choices=("single", "transpose", "row_reduce"), default="single")
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    session = Session(args.name, args.context, args.size)
    for stage in MANIFEST["stages"]:
        session.observe(stage["id"])
        session.advance(stage)
    report = session.finalize()
    print(json.dumps({"report": str(session.output / "report.json"), "correct": True,
                      "median_latency_ms": report["median_latency_ms"]}))


if __name__ == "__main__":
    main()
