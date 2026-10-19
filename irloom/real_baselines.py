"""Representative native workloads; reuse source, build and logging machinery."""
import io
import json
import random
import re
import statistics
import tarfile
import time
from pathlib import Path

from .real_bench import Benchmarks


class RealBaselines(Benchmarks):
    suite_targets=["bullet","tramp3d-v4"]
    suite_tests={"bullet":"Bullet/bullet", "tramp3d-v4":"tramp3d-v4/tramp3d-v4"}

    def __init__(self,args):
        super().__init__(args)
        self.variant="o3"
        self.inputs={}
        self.cmake_extra=[]
        self.link_flags=[]
        self.base_flags=list(self.config["flags"])
        self.build_records=[]
        self.profiles={}
        self.env["LLVM_PROFILE_FILE"]=str(self.work/"unassigned-%p.profraw")

    def build_dir(self,project):
        path=super().build_dir(project).parent/("formal-"+self.variant)
        if self.variant=="pgo-thinlto-native" and project in self.profiles:
            path=path/Path(self.profiles[project]["profile"]).parent.name
        path.mkdir(parents=True,exist_ok=True)
        return path

    def prepare_inputs(self):
        data=self.root/"data/formal-v1"
        data.mkdir(exist_ok=True)
        if "zstd" in self.args.project:
            # Separate whole source files for training and evaluation. No repeated padding.
            files=sorted(p for p in (self.source("zstd")/"lib").rglob("*") if p.suffix in (".c",".h"))
            for role in ("train","eval"):
                members=[p for i,p in enumerate(files) if (i%4==0)==(role=="train")]
                path=data/f"software-{role}.tar"
                recipe={"role":role,"source":"zstd 1.5.7 lib/**/*.c,*.h", "selection":"sorted file index % 4 == 0 is train, others eval",
                        "members":[str(p.relative_to(self.source("zstd"))) for p in members],"padding":"none"}
                if not path.exists():
                    partial=self.work/path.name
                    with tarfile.open(partial,"w",format=tarfile.USTAR_FORMAT) as archive:
                        for p in members:
                            payload=p.read_bytes()
                            info=tarfile.TarInfo(str(p.relative_to(self.source("zstd"))))
                            info.size=len(payload)
                            info.mode=0o644
                            archive.addfile(info,io.BytesIO(payload))
                    partial.rename(path)
                    path.with_suffix(".json").write_text(json.dumps(recipe,indent=2)+"\n")
                self.inputs["software-"+role]=path
            for role,seed,count in (("train",20261020,30000),("eval",20261021,300000)):
                path=data/f"records-{role}.jsonl"
                recipe={"role":role,"seed":seed,"records":count,"generator":"formal-v1 random.Random(seed), timestamp/id/tenant/status/value/token JSONL",
                        "distribution":"structured records with changing numeric fields and 96-bit random tokens"}
                if not path.exists():
                    rng=random.Random(seed)
                    partial=self.work/path.name
                    with partial.open("w") as output:
                        for i in range(count):
                            record={"timestamp":1700000000+i,"id":i,"tenant":rng.randrange(1000),
                                    "status":rng.choice([200,200,200,404,503]),"value":rng.randrange(1000000),"token":f"{rng.getrandbits(96):024x}"}
                            output.write(json.dumps(record,separators=(",",":"))+"\n")
                    partial.rename(path)
                    path.with_suffix(".recipe.json").write_text(json.dumps(recipe,indent=2)+"\n")
                self.inputs["records-"+role]=path
        self.metadata["formal_inputs"]={k:{"path":str(v),"bytes":v.stat().st_size} for k,v in self.inputs.items()}

    def select(self,variant,profile=None):
        self.variant=variant
        flags=list(self.base_flags)
        self.link_flags=[]
        if variant!="o3":
            flags+=["-march=native"]
        if "thinlto" in variant or variant=="pgo-train":
            flags+=["-flto=thin"]
            self.link_flags=[f"-fuse-ld={self.source('lld')/'usr/lib/llvm-18/bin/ld.lld'}", "-Wl,--thinlto-jobs=2","-Wl,--threads=2"]
        if variant=="pgo-train":
            flags+=["-fprofile-instr-generate"]
        elif variant=="audit-gprof":
            flags+=["-pg","-fno-omit-frame-pointer"]
        elif variant=="pgo-thinlto-native":
            if profile is None:
                raise ValueError("PGO-use requires this project's training profile")
            flags+=[f"-fprofile-instr-use={profile}"]
        if variant!="o3":
            flags+=["-fsave-optimization-record","-foptimization-record-passes=inline"]
        self.config["flags"]=flags
        self.cmake_extra=["-DCMAKE_AR=/usr/lib/llvm-18/bin/llvm-ar","-DCMAKE_RANLIB=/usr/lib/llvm-18/bin/llvm-ranlib",
                          "-DCMAKE_EXE_LINKER_FLAGS="+" ".join(self.link_flags)]

    def variants(self,project):
        # R2 requires two real projects; keep the official numerical programs
        # as O3 reference workloads, without multiplying their build matrix.
        return ["o3"] if project=="test-suite" else self.args.variants

    def build_one(self,project,variant,profile=None):
        self.select(variant,profile)
        if project=="zstd" and ("thinlto" in variant or variant=="pgo-train"):
            self.link_flags += [f"-Wl,--opt-remarks-filename,{self.build_dir(project)/'lto-inline.yaml'}",
                                "-Wl,--opt-remarks-passes,inline","-Wl,--opt-remarks-with-hotness"]
            self.cmake_extra[-1]="-DCMAKE_EXE_LINKER_FLAGS="+" ".join(self.link_flags)
        self.args.project=[project]
        start=time.perf_counter()
        first=len(self.events)
        super().build()
        build=self.build_dir(project)
        binaries=([build/"programs/zstd"] if project=="zstd" else [build/"sqlite3",build/"speedtest1"] if project=="sqlite" else
                  [build/"MultiSource/Benchmarks"/p for p in self.suite_tests.values()])
        record={"project":project,"variant":variant,"directory":str(build),"flags":list(self.config["flags"]),
                "link_flags":list(self.link_flags),"seconds":time.perf_counter()-start,"event_range":[first,len(self.events)],"binaries":[]}
        for binary in binaries:
            size=self.command(["/usr/lib/llvm-18/bin/llvm-size",binary],"size-"+binary.name)
            symbols=self.command(["/usr/lib/llvm-18/bin/llvm-nm","--defined-only",binary],"symbols-"+binary.name)
            record["binaries"].append({"path":str(binary),"bytes":binary.stat().st_size,"llvm_size":size.strip(),
                                       "defined_text_symbols":sum(len(s.split())>1 and s.split()[1] in ("T","t","W","w") for s in symbols.splitlines())})
        record["compile_commands"]=str(build/"compile_commands.json") if (build/"compile_commands.json").exists() else "native Makefile commands in build log"
        record["link_commands"]={str(p.relative_to(build)):p.read_text() for p in build.rglob("link.txt") if any(b.name in p.read_text() for b in binaries)}
        self.build_records.append(record)
        self.metadata["builds"]=self.build_records
        self.save("building")

    def train(self,project):
        self.select("pgo-train")
        build=self.build_dir(project)
        marker=build/".irloom-training.json"
        binary=build/("programs/zstd" if project=="zstd" else "speedtest1")
        identity={"recipe":"formal-v1", "binary_mtime_ns":binary.stat().st_mtime_ns,"binary_bytes":binary.stat().st_size}
        if marker.exists():
            cached=json.loads(marker.read_text())
            if cached["identity"]==identity and Path(cached["profile"]).is_file():
                print(f"cached independent PGO training: {project}",flush=True)
                self.profiles[project]={**cached,"cache_hit":True}
                return Path(cached["profile"])
        training=build/"training-v1"/self.stamp
        training.mkdir(parents=True)
        env={"LLVM_PROFILE_FILE":str(training/"train-%p.profraw")}
        start=time.perf_counter()
        first=len(self.events)
        if project=="zstd":
            commands=[[binary,"-b3","-i1","--single-thread",self.inputs[n]] for n in ("software-train","records-train")]
        else:
            # Different testsets, not merely a prefix of the evaluation data.
            commands=[[binary,"--memdb","--size","40","--testset","orm,cte"]]
        for command in commands:
            self.command(command,"train-"+project,cpu=True,env=env)
        raw=sorted(training.glob("*.profraw"))
        if not raw:
            raise ValueError(f"no training profiles: {project}")
        profile=training/"training.profdata"
        self.command([self.config["profdata"],"merge",*raw,"-o",profile],"merge-"+project)
        record={"identity":identity,"profile":str(profile),"commands":[[str(a) for a in c] for c in commands],
                "seconds":time.perf_counter()-start,"event_range":[first,len(self.events)],"cache_hit":False,
                "evaluation_excluded":"software-eval, records-eval" if project=="zstd" else "main testset size200"}
        marker.write_text(json.dumps(record,indent=2)+"\n")
        self.profiles[project]=record
        return profile

    def build(self):
        projects=list(self.args.project)
        self.prepare_inputs()
        try:
            for project in projects:
                for variant in self.variants(project):
                    profile=None
                    if variant=="pgo-thinlto-native":
                        self.build_one(project,"pgo-train")
                        profile=self.train(project)
                    self.build_one(project,variant,profile)
        finally:
            self.args.project=projects
        self.metadata["scope"]="formal Zstd/SQLite full-program variants; official MultiSource O3 only; no LLVM bootstrap"
        self.metadata["profiles"]=self.profiles
        self.metadata["measurement_policy"].update(order="rotate variants by round within each workload",variants=self.args.variants)

    def lit(self,name,label):
        build=self.build_dir("test-suite")
        test=build/"MultiSource/Benchmarks"/(self.suite_tests[name]+".test")
        iterations=20 if name=="tramp3d-v4" else 1
        if iterations>1:
            # Only a generated test file: unchanged official input and a reference
            # comparison for every process. timeit sums their user CPU times.
            lines=test.read_text().splitlines()
            batch=test.with_name(test.stem+"-batch20.test")
            runs=[line for line in lines if line.startswith("RUN:")]
            checks=[line for line in lines if line.startswith("VERIFY:")]
            batch.write_text("\n".join(runs*iterations+[line.replace("%o",f"%T/{batch.name}-{i}.out") for i in range(iterations) for line in checks])+"\n")
            test=batch
        output_dir=self.logs/label
        output_dir.mkdir()
        (output_dir/"tools").symlink_to(build/"tools",target_is_directory=True)
        result=self.logs/(label+".json")
        self.command(["uv","run","python",self.source("llvm")/"utils/lit/lit.py","-sv","-j1","--timeout=300","-o",result,
                      test],label,cpu=True,timeout=360,
                     env={"IRLOOM_LIT_LOG_ROOT":str(output_dir)})
        tests=json.loads(result.read_text())["tests"]
        if len(tests)!=1 or tests[0]["code"]!="PASS" or "hash" in tests[0].get("metrics",{}):
            raise ValueError(f"official reference validation failed: {result}")
        return {"wall_seconds":self.events[-1]["seconds"],"exec_seconds":tests[0]["metrics"]["exec_time"],"processes":iterations,"log":str(result)}

    def validate(self):
        for project in self.args.project:
            build=self.build_dir(project)
            if project=="zstd":
                for name in ("software-eval","records-eval"):
                    source=self.inputs[name]
                    packed=self.work/(self.variant+"-"+name+".zst")
                    restored=packed.with_suffix(".restored")
                    binary=build/"programs/zstd"
                    self.command([binary,"-q","--single-thread","-3",source,"-o",packed],"validate-compress-"+name,cpu=True)
                    self.command([binary,"-q","-d",packed,"-o",restored],"validate-decompress-"+name,cpu=True)
                    if restored.read_bytes()!=source.read_bytes():
                        raise ValueError("Zstd roundtrip differs")
            elif project=="sqlite":
                pin=next(s for s in self.config["sources"] if s["name"]==project)
                binary=build/"sqlite3"
                identity=self.command([binary,":memory:","SELECT sqlite_version()||'|'||sqlite_source_id();"],"sqlite-version")
                if identity.strip()!=pin["version"]+"|"+pin["revision"]:
                    raise ValueError("SQLite source identity differs")
                database=self.work/(self.variant+"-validation.db")
                self.command([build/"speedtest1","--size","10","--testset","main",database],"sqlite-file-validation",cpu=True)
                check=self.command([binary,database,"PRAGMA integrity_check; SELECT count(*)>0 FROM t6;"],"sqlite-integrity",cpu=True)
                if check.strip().splitlines()!=["ok","1"]:
                    raise ValueError("SQLite integrity check failed")
                known=self.command([binary,":memory:","CREATE TABLE t(k INTEGER,v TEXT); INSERT INTO t VALUES(1,'a'),(2,'b'),(3,'c'); BEGIN; INSERT INTO t VALUES(4,'d'); ROLLBACK; SELECT count(*),sum(k),group_concat(v,'') FROM t; SELECT json_extract('{\"x\":7}','$.x');"],"sqlite-known-answer",cpu=True)
                if known.strip().splitlines()!=["3|6|abc","7"]:
                    raise ValueError("SQLite known-answer SQL mismatch")
                dump=self.command([binary,database,".dump"],"sqlite-differential-dump",cpu=True)
                if hasattr(self,"sqlite_reference") and dump!=self.sqlite_reference:
                    raise ValueError("SQLite database differs across compilation variants")
                self.sqlite_reference=dump
            else:
                for name in self.suite_tests:
                    self.lit(name,"validate-"+self.variant+"-"+name)

    def workloads(self):
        jobs=[]
        if "zstd" in self.args.project:
            binary=self.build_dir("zstd")/"programs/zstd"
            for name in ("software-eval","records-eval"):
                jobs.append(("zstd-"+name,[binary,"-b3","-i2","--single-thread",self.inputs[name]]))
        if "sqlite" in self.args.project:
            jobs.append(("sqlite-main-memdb-size200",[self.build_dir("sqlite")/"speedtest1","--memdb","--size","200","--testset","main"]))
        return jobs

    def sample(self,name,args):
        text=self.command(args,name,cpu=True,timeout=300)
        result={"wall_seconds":self.events[-1]["seconds"],"log":self.events[-1]["log"]}
        if name.startswith("zstd-"):
            matches=re.findall(r"([0-9.]+) MB/s,\s*([0-9.]+) MB/s",text)
            if not matches:
                raise ValueError("Zstd throughput missing")
            result.update(compression_MB_s=float(matches[-1][0]),decompression_MB_s=float(matches[-1][1]))
        else:
            match=re.search(r"TOTAL\.+\s*([0-9.]+)s",text)
            if not match:
                raise ValueError("SQLite TOTAL missing")
            result["compute_seconds"]=float(match[1])
        return result

    def run(self):
        projects=list(self.args.project)
        try:
            for project in projects:
                self.args.project=[project]
                variants=self.variants(project)
                jobs={}
                for variant in variants:
                    self.variant=variant
                    print(f"validate: {project} {variant}",flush=True)
                    self.validate()
                    jobs[variant]=dict(self.workloads())
                names=list(jobs[variants[0]]) if project!="test-suite" else list(self.suite_tests)
                for name in names:
                    records={}
                    for variant in variants:
                        record={"name":name,"project":project,"variant":variant,"status":"collecting","cpu":self.cpu,"samples":[],
                                "median_wall_seconds":0,"mad_wall_seconds":0}
                        self.results.append(record)
                        records[variant]=record
                    for i in range(self.args.repetitions+1):
                        order=variants[i%len(variants):]+variants[:i%len(variants)]
                        for variant in order:
                            self.variant=variant
                            label=f"{name}-{variant}-round{i}"
                            sample=self.sample(label,jobs[variant][name]) if project!="test-suite" else self.lit(name,label)
                            if i:
                                sample["round"]=i
                                record=records[variant]
                                record["samples"].append(sample)
                                median=statistics.median(s["wall_seconds"] for s in record["samples"])
                                record.update(median_wall_seconds=median,mad_wall_seconds=statistics.median(abs(s["wall_seconds"]-median) for s in record["samples"]))
                                for metric in ("compression_MB_s","decompression_MB_s","compute_seconds","exec_seconds"):
                                    if metric in sample:
                                        values=[s[metric] for s in record["samples"]]
                                        med=statistics.median(values)
                                        record.setdefault("metrics",{})[metric]={"median":med,"mad":statistics.median(abs(v-med) for v in values)}
                        self.save("measuring")
                        print(f"measured: {name} round {i}/{self.args.repetitions} ({'warmup' if not i else 'sample'})",flush=True)
                    for record in records.values():
                        record["status"]="pass"
        finally:
            self.args.project=projects

    def save(self,status,error=None):
        # Keep original portable flags in the common environment; exact effective
        # flags and native link commands are attached to each isolated build.
        self.config["flags"]=list(self.base_flags)
        super().save(status,error)
        if self.results:
            lines=["", "## Native workload metrics", "", "| Workload | Variant | Metric | Median | MAD |", "| --- | --- | --- | ---: | ---: |"]
            for record in self.results:
                for metric,values in record.get("metrics",{}).items():
                    lines.append(f"| {record['name']} | {record['variant']} | {metric} | {values['median']:.6f} | {values['mad']:.6f} |")
            with (self.logs/"summary.md").open("a") as output:
                output.write("\n".join(lines)+"\n")
        if hasattr(self,"metadata") and "audit" in self.metadata:
            lines=["", "## Profiling evidence", "", "Separate instrumented builds; samples below are not baseline timings.", "",
                   "| Workload | Largest self samples | Call graph log |", "| --- | --- | --- |"]
            for profile in self.metadata["audit"]["profiles"]:
                top="; ".join(f"{f['function']}: {f['self_percent']:.1f}%" for f in profile["top_functions"][:4])
                lines.append(f"| {profile['name']} | {top} | {profile['call_graph_log']} |")
            with (self.logs/"summary.md").open("a") as output:
                output.write("\n".join(lines)+"\n")

    def audit(self):
        from .real_audit import audit
        projects=list(self.args.project)
        try:
            self.args.project=[p for p in projects if p in ("zstd","sqlite")]
            if not self.args.project:
                return
            audit(self)
        finally:
            self.args.project=projects
