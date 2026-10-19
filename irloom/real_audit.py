"""Function/call evidence for fixed formal workloads; never rank by count alone."""
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import shlex


def inline_records(path):
    """Read the scalar fields needed from LLVM's own remark stream."""
    record={}
    with path.open() as stream:
        for line in stream:
            if line.startswith("--- !"):
                if record:
                    yield record
                record={"kind":line.strip().split("!")[-1],"path":str(path),"stage":"lto" if path.name.startswith("lto-inline") else "frontend"}
            if "source" not in record and "DebugLoc:" in line:
                source=re.search(r"File:\s*('[^']+'|\"[^\"]+\"|[^,}]+)",line)
                if source:
                    record["source"]=source[1].strip("'\" ")
            if "source" in record and "line" not in record:
                lineno=re.search(r"Line:\s*(\d+)",line)
                if lineno:
                    record["line"]=int(lineno[1])
            match=re.match(r"^(?:\s*-\s*)?(Pass|Name|Function|Caller|Callee|Cost|Threshold|Hotness|Reason):\s*(.*?)\s*$",line)
            if match:
                key,value=match.groups()
                record[key]=value.strip("'\"")
    if record:
        yield record


def optimization_summary(build,files=None):
    counts=Counter()
    edges=[]
    if files is None:
        files=sorted(p for p in build.rglob("*.opt.yaml") if p.name!="conftest__.opt.yaml")+sorted(build.glob("lto-inline.yaml*"))
    for path in files:
        if not path.is_file():
            continue
        for record in inline_records(path):
            counts[(record.get("Pass"),record.get("Name"),record["kind"])]+=1
            if record.get("Caller") and record.get("Callee"):
                edges.append(record)
    return {"files":[str(p) for p in files],"decisions":[{"pass":p,"name":n,"kind":k,"count":v} for (p,n,k),v in counts.most_common()],"edges":edges}


def sqlite_core_remarks(bench,build,variant):
    # Upstream links shell.c/sqlite3.c and speedtest1.c/sqlite3.c concurrently:
    # their default sqlite3.opt.yaml paths collide. Ask Make for the exact
    # speedtest1 recipe, then compile only its core TU for diagnostic remarks.
    cache=build/"audit-remarks"
    cache.mkdir(exist_ok=True)
    output=cache/"speedtest1-core.opt.yaml"
    marker=cache/"configuration.json"
    identity={"settings":json.loads((build/".irloom-configure.json").read_text()),
              "source_mtime_ns":(build/"sqlite3.c").stat().st_mtime_ns,
              "binary_mtime_ns":(build/"speedtest1").stat().st_mtime_ns,"recipe":1}
    if marker.exists() and output.exists() and json.loads(marker.read_text())["identity"]==identity:
        return output,{**json.loads(marker.read_text()),"cache_hit":True}
    source=bench.source("sqlite").resolve()/"test/speedtest1.c"
    recipe=bench.command(["make","-n","-W",source,"speedtest1"],"sqlite-expanded-recipe-"+variant,cwd=build)
    lines=recipe.replace("\\\n","").splitlines()
    line=next(s for s in lines if s.startswith(bench.config["compiler"]) and " -o speedtest1 " in s)
    original=shlex.split(line)
    command=[]
    skip=False
    for item in original:
        if skip:
            skip=False
            continue
        if item=="-rpath":
            skip=True
        elif item==str(source) or item in ("-lm","-lz") or item.startswith(("-Wl,","-fuse-ld=")):
            continue
        else:
            command.append(item)
    command[command.index("-o")+1]=str(bench.work/("sqlite-core-"+variant+".o"))
    command += ["-c",f"-foptimization-record-file={output}"]
    if variant=="pgo-thinlto-native":
        command += ["-Werror=profile-instr-out-of-date"]
    bench.command(command,"sqlite-core-remarks-"+variant,cwd=build,timeout=300)
    record={"identity":identity,"cache_hit":False,"command":command,"log":bench.events[-1]["log"],"seconds":bench.events[-1]["seconds"],
            "scope":"exact speedtest1 core TU flags; frontend diagnostic only, not SQLite ThinLTO backend records"}
    marker.write_text(json.dumps(record,indent=2)+"\n")
    return output,record


def flat_profile(text):
    rows=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<4:
            continue
        try:
            percent,cumulative,own=map(float,fields[:3])
        except ValueError:
            continue
        calls=int(fields[3]) if fields[3].isdigit() else None
        name=" ".join(fields[6:] if calls is not None and len(fields)>6 else fields[3:])
        rows.append({"function":name,"self_percent":percent,"self_seconds":own,"cumulative_seconds":cumulative,"calls":calls})
    return rows


def audit(bench):
    projects=list(bench.args.project)
    if any(p not in ("zstd","sqlite") for p in projects):
        raise ValueError("audit selects --project zstd sqlite; official MultiSource remains the R1/O3 reference")
    evidence={"perf_executable":shutil.which("perf"),"perf_event_paranoid":Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
              "method":"GNU gprof sampling/call arcs on separate -pg native builds plus Clang PGO block counts and inline remarks",
              "limits":["instrumentation changes timing/inlining; these runs are not baseline performance samples",
                        "gprof samples user program text, excludes uninstrumented library call arcs and kernel time",
                        "PGO function/block counts measure execution frequency, not CPU time; source-level inline functions remain in counts",
                        "no PMU counters; sibling CPU, frequency, background traffic and cache state uncontrolled"],"profiles":[],"optimizations":{}}
    # configure probes and correctness executions are instrumented too. Give
    # them a scratch prefix so no default gmon.out lands in the repository.
    bench.env["GMON_OUT_PREFIX"]=str(bench.work/"validation-gmon")
    evidence["validation_gmon_prefix"]=bench.env["GMON_OUT_PREFIX"]
    bench.metadata["audit"]=evidence
    if not shutil.which("gprof"):
        raise RuntimeError("gprof is unavailable; preserve limitation and use existing PGO/remarks evidence")
    evidence["gprof_version"]=bench.command(["gprof","--version"],"gprof-version")
    try:
        for project in projects:
            bench.build_one(project,"audit-gprof")
            bench.variant="audit-gprof"
            bench.args.project=[project]
            bench.validate()
            jobs=bench.workloads()
            for name,args in jobs:
                prefix=bench.logs/("gmon-"+name)
                bench.command(args,"profile-"+name,cpu=True,env={"GMON_OUT_PREFIX":str(prefix)},timeout=300)
                files=sorted(bench.logs.glob(prefix.name+".*"))
                if not files:
                    raise RuntimeError(f"gprof emitted no samples: {name}")
                flat=bench.command(["gprof","--brief","--flat-profile",args[0],*files],"gprof-flat-"+name)
                flat_log=bench.events[-1]["log"]
                bench.command(["gprof","--brief","--graph",args[0],*files],"gprof-calls-"+name)
                record={"name":name,"project":project,"gprof_binary":str(args[0]),"flat_log":flat_log,"call_graph_log":bench.events[-1]["log"],
                        "top_functions":flat_profile(flat)[:25],"raw_samples":[str(p) for p in files]}
                if project in bench.profiles:
                    # Evaluation feedback is for this audit only: never merged into
                    # the frozen training profile used by the PGO-use binaries.
                    bench.variant="pgo-train"
                    instrumented=dict(bench.workloads())[name]
                    rawdir=bench.work/("audit-"+name)
                    rawdir.mkdir()
                    bench.command(instrumented,"audit-counts-"+name,cpu=True,env={"LLVM_PROFILE_FILE":str(rawdir/"eval-%p.profraw")})
                    profile=rawdir/"evaluation-only.profdata"
                    bench.command([bench.config["profdata"],"merge",*sorted(rawdir.glob("*.profraw")),"-o",profile],"audit-merge-"+name)
                    counts=bench.command([bench.config["profdata"],"show","--topn=30","--counts",profile],"audit-counts-show-"+name)
                    record["evaluation_count_profile"]={"path":str(profile),"excluded_from_training":True,"summary":counts,"log":bench.events[-1]["log"]}
                    bench.variant="audit-gprof"
                evidence["profiles"].append(record)
                bench.save("auditing")
            for variant in ("o3-native","thinlto-native","pgo-thinlto-native"):
                bench.variant=variant
                build=bench.build_dir(project)
                if not (build/".irloom-configure.json").exists():
                    continue
                diagnostic=None
                files=None
                if project=="sqlite":
                    remarks,diagnostic=sqlite_core_remarks(bench,build,variant)
                    files=[remarks]
                summary=optimization_summary(build,files)
                output=bench.logs/f"inline-{project}-{variant}.json"
                output.write_text(json.dumps(summary,indent=2)+"\n")
                evidence["optimizations"][project+"/"+variant]={"summary":str(output),"files":len(summary["files"]),"decisions":summary["decisions"],"edges":len(summary["edges"]),"diagnostic":diagnostic}
    finally:
        bench.args.project=projects
