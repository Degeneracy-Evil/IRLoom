"""Small explicit whole-program recipes; original sources stay in the source cache."""
import json
from .schema import TerminalMeasurement
from irloom.compiler import ROOT

BASE = ROOT / "benchmark-data/architecture/frontends"
RECIPES = json.loads((ROOT / "experiments/architecture/programs.json").read_text())


def prepare(session, program):
    recipe = RECIPES["programs"][program]
    source = ROOT / "benchmark-data/sources/llvm-test-suite-18.1.3/MultiSource/Benchmarks" / recipe["path"]
    session.manifest = json.loads((ROOT / "compiler/pipelines/polygeist-v1.json").read_text())
    session.bridge, session.opt, session.translate = (BASE / "build/bin" / x for x in
        ("irloom-polygeist-bridge", "polygeist-opt", "mlir-translate"))
    for tool in (session.bridge, session.opt, session.translate, BASE / "build/bin/cgeist"):
        if not tool.is_file():
            raise FileNotFoundError(f"missing {tool}; run bash scripts/prepare-polygeist.sh --build")
    session.hardware["frontend_version"] = session.run([BASE / "build/bin/cgeist", "--version"], "frontend-version")
    session.ir = session.output / "imported.mlir"
    session.run([BASE / "build/bin/cgeist", source, "-S", "-O0", "-raise-scf-to-affine", *["-D" + x for x in recipe.get("defines", [])],
                 "-std=gnu89", "-resource-dir=/usr/lib/llvm-18/lib/clang/18", "-o", session.ir], "import")
    (session.output / "program-source.json").write_text(json.dumps({"suite": RECIPES["suite"], "source": RECIPES["source"],
                                                                  **recipe, "coverage": "entire source file and included headers"}, indent=2))
    session.original_source, session.recipe = source, recipe


def measure(session):
    ll, obj, exe, reference = (session.output / name for name in ("program.ll", "program.o", "program", "reference"))
    session.run([session.translate, "--mlir-to-llvmir", session.ir, "-o", ll], "translate")
    session.run(["clang-18", *session.manifest["backend"], "-c", ll, "-o", obj], "codegen")
    session.run(["clang-18", obj, "-lm", "-o", exe], "link")
    session.run(["clang-18", *session.manifest["backend"], "-std=gnu89", "-Wno-implicit-int",
                 *["-D" + x for x in session.recipe.get("defines", [])], session.original_source, "-lm", "-o", reference], "reference-compile")
    checks = []
    for args in session.recipe["check_args"]:
        args = [x.replace("{source_dir}", str(session.original_source.parent)) for x in args]
        expected = session.run([reference, *args], "reference-check")
        actual = session.run([exe, *args], "program-check")
        if actual != expected:
            raise RuntimeError("whole-program output mismatch; candidate excluded")
        checks.append({"args": args, "status": "pass", "oracle": "independent Clang O3 original source", "bytes": len(actual)})
    samples = sample_existing(session, exe, reference)
    return TerminalMeasurement(checks, samples, obj.stat().st_size,
                               {"commands_s": sum(e["seconds"] for e in session.events),
                                "timing_scope": "lean C++ parent times native fork/exec/wait/output; neural driver launch and logging excluded from latency, included in optimization costs"})


def sample_existing(session, exe, reference, prefix=""):
    args = [x.replace("{source_dir}", str(session.original_source.parent)) for x in session.recipe["measure_args"]]
    expected = session.run([reference, *args], prefix + "reference-measure-input")
    if session.run([exe, *args], prefix + "warmup") != expected:
        raise RuntimeError("warmup output mismatch")
    samples = []
    for _ in range(5):
        metric = session.logs / f"native-{len(session.events):03}.json"
        actual = session.run([ROOT / "benchmark-data/architecture/build/irloom-time", metric, exe, *args], prefix + "measure")
        if actual != expected:
            raise RuntimeError("measured output mismatch")
        native = json.loads(metric.read_text())
        samples.append({**native, "iterations": 1, "driver_wall_ms": session.events[-1]["seconds"] * 1000})
    return samples
