"""Tool discovery, logged commands, and snapshot-level compilation lineage."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def environment():
    env = dict(os.environ)
    env.update(TMPDIR=str(ROOT / "tmp"), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def tool(name):
    override = os.environ.get(name.upper().replace("-", "_").replace("++", "XX"))
    paths = [override] if override else [
        str(ROOT / "tmp/toolchain/usr/lib/llvm-18/bin" / name),
        shutil.which(name), str(Path("/usr/lib/llvm-18/bin") / name)]
    for path in paths:
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    raise FileNotFoundError(f"missing {name}; see docs/RUNNING.md")


def probe():
    result = {"os": platform.platform(), "arch": platform.machine(),
              "affinity": sorted(os.sched_getaffinity(0)), "tools": {},
              "environment": {k: environment()[k] for k in
                              ("TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    for name in ("mlir-opt", "mlir-translate", "clang", "clang++", "llc", "llvm-config", "llvm-objdump"):
        try:
            path = tool(name)
            p = subprocess.run([path, "--version"], capture_output=True, text=True,
                               timeout=10, env=environment())
            result["tools"][name] = {"path": path, "version": p.stdout.strip(),
                                      "returncode": p.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["tools"][name] = {"error": str(exc)}
    for name, command in (("cpu", ["lscpu", "-J"]),
                          ("git_commit", ["git", "rev-parse", "HEAD"]),
                          ("git_status", ["git", "status", "--porcelain"])):
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        result[name] = p.stdout.strip()
    result["runtime_limits"] = {
        "frequency_control": "uncontrolled", "pmu": "not collected",
        "loadavg": os.getloadavg(),
        "memory": Path("/proc/meminfo").read_text(),
    }
    cache_dir = Path(f"/sys/devices/system/cpu/cpu{result['affinity'][0]}/cache")
    result["cpu_caches"] = [{"cpu": result["affinity"][0], **{
        key: (index / key).read_text().strip() for key in ("level", "type", "size", "shared_cpu_list")}}
        for index in sorted(cache_dir.glob("index*"))]
    return result


class CommandError(RuntimeError):
    pass


def command(args, log, events, stage, timeout=120, cpu=None):
    args = [str(x) for x in args]
    event = {"stage": stage, "command": args, "cwd": str(ROOT), "log": str(log)}
    events.append(event)
    start = time.perf_counter()
    try:
        p = subprocess.run(args, cwd=ROOT, env=environment(), capture_output=True,
                           text=True, timeout=timeout,
                           preexec_fn=(lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None else None)
        event.update(returncode=p.returncode, status="ok" if p.returncode == 0 else "failed")
        log.write_text(json.dumps(event, ensure_ascii=False) + "\n" + p.stdout + p.stderr)
        if p.returncode:
            raise CommandError(f"{stage}: exit {p.returncode}; see {log}")
        return p.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        event.update(status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
                     error=str(exc))
        log.write_text(json.dumps(event, ensure_ascii=False) + "\n" + str(exc))
        raise CommandError(f"{stage}: {exc}") from exc
    finally:
        event["seconds"] = time.perf_counter() - start


def compile_candidate(source, output, logs, record, transforms=(), defines=(), high_choice=None):
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    events = record["events"]
    record["snapshots"] = []
    high = output / "high.mlir"
    high.write_text(source)
    record["snapshots"].append({"stage": "high", "path": str(high), "parent": None})
    stages = []
    if high_choice in ("separate","fused"):
        stages.append(("high_decision",["--linalg-fuse-elementwise-ops"] if high_choice=="fused" else []))
        stages.append(("bufferized",["--one-shot-bufferize","--buffer-deallocation","--canonicalize","--cse"]))
    elif high_choice is not None:
        if high_choice not in ("ijk", "ikj", "ij", "ji"):
            raise ValueError(f"unsupported high choice: {high_choice}")
        schedule = output / "schedule.mlir"
        schedule.write_text((ROOT / f"compiler/schedules/{high_choice}.mlir").read_text())
        record["schedule"] = str(schedule)
        stages.append(("high_decision", [f"--transform-preload-library=transform-library-paths={schedule}",
                                         "--transform-interpreter"]))
    stages += [
        ("affine", ["--convert-linalg-to-affine-loops"]),
        ("transformed", list(transforms) + ["--canonicalize", "--cse"]),
        ("scf", ["--lower-affine"]),
        ("llvm_dialect", ["--convert-scf-to-cf", "--convert-arith-to-llvm",
                          "--finalize-memref-to-llvm", "--convert-func-to-llvm",
                          "--convert-cf-to-llvm", "--reconcile-unrealized-casts"]),
    ]
    previous = high
    for stage, flags in stages:
        path = output / f"{stage}.mlir"
        command([tool("mlir-opt"), previous, *flags, "-o", path],
                logs / f"{stage}.log", events, stage)
        record["snapshots"].append({"stage": stage, "path": str(path),
                                    "parent": str(previous), "flags": flags,
                                    "event_index": len(events) - 1})
        previous = path
    ll = output / "kernel.ll"
    command([tool("mlir-translate"), "--mlir-to-llvmir", previous, "-o", ll],
            logs / "translate.log", events, "translate")
    record["snapshots"].append({"stage": "llvm", "path": str(ll),
                                "parent": str(previous), "event_index": len(events) - 1})
    flags = ["-O3", "-march=native", "-ffp-contract=off"]
    obj = output / "kernel.o"
    command([tool("clang"), *flags, "-c", ll, "-o", obj],
            logs / "codegen.log", events, "codegen")
    return finish_native(obj, output, logs, record, defines)


def compile_reference(output, logs, record, defines):
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    source = output / "matmul_reference.cpp"
    source.write_text((ROOT / "benchmarks/matmul_reference.cpp").read_text())
    (output / "memref.h").write_text((ROOT / "benchmarks/memref.h").read_text())
    record["reference_source"] = str(source)
    record["snapshots"] = []
    obj = output / "kernel.o"
    command([tool("clang++"), "-O3", "-march=native", "-ffp-contract=off", *defines,
             "-c", source, "-o", obj], logs / "reference-codegen.log", record["events"], "codegen")
    return finish_native(obj, output, logs, record, defines)


def finish_native(obj, output, logs, record, defines):
    events = record["events"]
    assembly = command([tool("llvm-objdump"), "--disassemble", "--reloc", obj],
                       logs / "disassembly.log", events, "disassembly")
    disassembly = output / "kernel.disasm"
    # Drop only the object filename header; retain offsets, opcodes and relocations.
    disassembly.write_text("\n".join(line for line in assembly.splitlines()
                                      if "file format" not in line).strip() + "\n")
    binary = output / "runner"
    runner = output / "runner.c"
    runner.write_text((ROOT / "benchmarks/runner.c").read_text())
    (output / "memref.h").write_text((ROOT / "benchmarks/memref.h").read_text())
    command([tool("clang"), "-O2", "-ffp-contract=off", *defines,
             runner, obj, "-lm", "-Wl,--build-id=none", "-o", binary],
            logs / "link.log", events, "link")
    record.update(binary=str(binary), object=str(obj), disassembly=str(disassembly),
                  compile_seconds=sum(e["seconds"] for e in events))
    return binary
