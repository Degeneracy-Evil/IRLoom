"""SQLite PGO coverage A/B with frozen, independently checked SQL inputs."""
import json
from pathlib import Path
import re
import resource
import shlex
import shutil
import time

from .compiler import ROOT
from .real_ab import ZstdAB,save_comparisons
from .real_baselines import RealBaselines
from .real_audit import inline_records


class SQLiteAB(RealBaselines):
    groups=("P0","P1")
    cpu_budget=60.0
    cases={"point":(52001,2000000),"range":(62003,200000)}

    def __init__(self,args):
        super().__init__(args)
        self.args.project=["sqlite"]
        self.args.variants=list(self.groups)
        self.namespace=self.root/"builds/sqlite/3.50.4"/("clang-"+self.config["compiler_version"])/"pgo-ab-v1"
        self.generator=self.namespace/"generated"
        self.group_profiles={}
        self.programs={}
        self.native_flags=[*self.base_flags,"-march=native","-flto=thin","-gline-tables-only"]
        self.driver=ROOT/"benchmarks/sqlite_workload.cpp"
        self.lld=self.source("lld")/"usr/lib/llvm-18/bin/ld.lld"

    def command(self,*args,**kwargs):
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            return RealBaselines.command(self,*args,**kwargs)
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            if self.events:
                self.events[-1].update(user_cpu_seconds=after.ru_utime-before.ru_utime,system_cpu_seconds=after.ru_stime-before.ru_stime)
                (self.logs/"events.json").write_text(json.dumps(self.events,indent=2)+"\n")

    def save(self,status,error=None):
        super().save(status,error)
        if status=="pass" and self.results:
            save_comparisons(self)

    def generate_source(self):
        self.generator.mkdir(parents=True,exist_ok=True)
        marker=self.generator/"configuration.json"
        settings={"CC":self.config["compiler"],"CFLAGS":"-O3 -DNDEBUG -march=native"}
        if marker.exists() and json.loads(marker.read_text())!=settings:
            raise ValueError("generator configuration changed; use a new namespace")
        if not marker.exists():
            self.command([self.source("sqlite")/"configure","--disable-shared","--disable-readline","--disable-tcl","--fts5","--rtree"],
                         "ab-sqlite-configure",cwd=self.generator,env=settings)
            marker.write_text(json.dumps(settings,indent=2)+"\n")
        self.command(["make",f"-j{self.args.jobs}","sqlite3.c","sqlite3.h"],"ab-sqlite-generate",cwd=self.generator)
        source=(self.generator/"sqlite3.c").read_text()
        start=re.search(r'SQLITE_PRIVATE int sqlite3VdbeExec\([^;{]*\)\s*\{',source)
        if not start:
            raise ValueError("cannot locate VdbeExec definition")
        first=source[:start.start()].count("\n")
        lines=source.splitlines()
        stop=next((i for i in range(first+1,len(lines)) if lines[i].startswith("SQLITE_PRIVATE ")),len(lines))
        self.callsites=[{"line":i+1,"column":lines[i].index("sqlite3BtreeIndexMoveto")+1,"source":lines[i].strip()}
                        for i in range(first,stop) if "sqlite3BtreeIndexMoveto(" in lines[i]]
        if len(self.callsites)!=4:
            raise ValueError(f"expected four source callsites in the pinned VdbeExec, got {self.callsites}")
        self.metadata["protocol"]={"version":1,"groups":self.groups,"training_cpu_minimum_seconds":self.cpu_budget,
            "maximum_completion_difference_fraction":0.02,"P0":"speedtest1 --memdb --size 40 --testset orm,cte, complete units until CPU budget",
            "P1":"sqlite-workload train 42001 100003 60; point queries, integer oracle",
            "evaluation":{k:{"seed":seed,"rows":100003,"queries":n} for k,(seed,n) in self.cases.items()},
            "batches":self.args.batches,"core_pgo_only":"same native ThinLTO C++ driver in both use groups; profile use applies only to SQLite core",
            "source_callsites":self.callsites,"old_main":"not evaluated or merged"}
        self.metadata["tools"]["llvm-cov"]=self.command(["/usr/lib/llvm-18/bin/llvm-cov","--version"],"llvm-cov-version")

    def core_recipe(self,flags):
        harness=self.source("sqlite").resolve()/"test/speedtest1.c"
        text=self.command(["make","-n","CFLAGS="+" ".join(flags),"-W",harness,"speedtest1"],"ab-expanded-core",cwd=self.generator)
        line=next(s for s in text.replace("\\\n","").splitlines() if s.startswith(self.config["compiler"]) and " -o speedtest1 " in s)
        command=[]
        skip=False
        for item in shlex.split(line):
            if skip:
                skip=False
                continue
            if item=="-rpath":
                skip=True
            elif item in (str(harness),"-lm","-lz") or item.startswith(("-Wl,","-fuse-ld=")):
                continue
            else:
                command.append(item)
        return command

    def compile_program(self,group,stage,profile=None):
        build=self.namespace/group/stage
        if profile:
            build=build/profile.parent.name
        build.mkdir(parents=True,exist_ok=True)
        flags=[*self.native_flags,"-fsave-optimization-record","-foptimization-record-passes=inline"]
        driver_flags=list(self.native_flags)
        if stage=="generate":
            flags += ["-fprofile-instr-generate","-fcoverage-mapping"]
            driver_flags += ["-fprofile-instr-generate","-fcoverage-mapping"]
        else:
            flags += [f"-fprofile-instr-use={profile}","-Werror=profile-instr-out-of-date"]
        identity={"group":group,"stage":stage,"flags":flags,"driver_flags":driver_flags,
                  "driver_mtime_ns":self.driver.stat().st_mtime_ns,"source_mtime_ns":(self.generator/"sqlite3.c").stat().st_mtime_ns}
        marker=build/"configuration.json"
        start=time.perf_counter()
        first=len(self.events)
        cached=marker.exists() and json.loads(marker.read_text())==identity and (build/"sqlite-workload").exists()
        if not cached:
            core=self.core_recipe(flags)
            core[core.index("-o")+1]=str(build/"core.o")
            core += ["-c",f"-foptimization-record-file={build/'core.opt.yaml'}"]
            self.command(core,"ab-core-"+group+"-"+stage,cwd=self.generator,timeout=600)
            self.command([self.config["cxx"],*driver_flags,"-std=c++17","-I",self.generator,"-c",self.driver,"-o",build/"driver.o"],
                         "ab-driver-"+group+"-"+stage,timeout=300)
            links=[f"--ld-path={self.lld}","-Wl,--threads=2","-Wl,--thinlto-jobs=2"]
            self.command([self.config["cxx"],*driver_flags,build/"driver.o",build/"core.o","-lm","-lz",*links,
                          "-Wl,--save-temps=opt",f"-Wl,--opt-remarks-filename,{build/'lto-inline.yaml'}",
                          "-Wl,--opt-remarks-passes,inline","-Wl,--opt-remarks-with-hotness","-o",build/"sqlite-workload"],
                         "ab-link-"+group+"-"+stage,timeout=600)
            if stage=="generate":
                harness=self.source("sqlite")/"test/speedtest1.c"
                self.command([self.config["compiler"],*driver_flags,"-I",self.generator,"-DSQLITE_OS_KV_OPTIONAL",
                              "-c",harness,"-o",build/"speedtest1.o"],"ab-official-harness-"+group)
                self.command([self.config["cxx"],*driver_flags,build/"speedtest1.o",build/"core.o","-lm","-lz",*links,
                              "-o",build/"speedtest1"],"ab-trainer-link-"+group,timeout=600)
            marker.write_text(json.dumps(identity,indent=2)+"\n")
        record={**identity,"directory":str(build),"cache_hit":cached,"seconds":time.perf_counter()-start,"event_range":[first,len(self.events)],
                "binary":str(build/"sqlite-workload"),"bytes":(build/"sqlite-workload").stat().st_size,
                "llvm_size":self.command(["/usr/lib/llvm-18/bin/llvm-size",build/"sqlite-workload"],"ab-size-"+group+"-"+stage)}
        self.metadata.setdefault("ab_builds",[]).append(record)
        self.save("building")
        return build

    def train_group(self,group,build):
        marker=build/"training.json"
        identity={"recipe":1,"cpu_budget":self.cpu_budget,"driver_mtime_ns":(build/"sqlite-workload").stat().st_mtime_ns,
                  "speedtest_mtime_ns":(build/"speedtest1").stat().st_mtime_ns}
        if marker.exists():
            record=json.loads(marker.read_text())
            if record["identity"]==identity and Path(record["profile"]).exists():
                return {**record,"cache_hit":True}
        directory=build/"profiles"/self.stamp
        directory.mkdir(parents=True)
        env={"LLVM_PROFILE_FILE":str(directory/"train-%p.profraw")}
        first=len(self.events)
        start=time.perf_counter()
        cpu=0
        rounds=0
        while cpu<self.cpu_budget:
            args=[build/"speedtest1","--memdb","--size","40","--testset","orm,cte"] if group=="P0" else [build/"sqlite-workload","train","42001","100003",str(self.cpu_budget)]
            text=self.command(args,f"ab-train-{group}-{rounds}",cpu=True,env=env,timeout=180)
            if group=="P1":
                self.check_answer(json.loads(text),"train",42001)
            event=self.events[-1]
            cpu+=event["user_cpu_seconds"]+event["system_cpu_seconds"]
            rounds+=1
        profile=directory/"training.profdata"
        self.command([self.config["profdata"],"merge",*sorted(directory.glob("*.profraw")),"-o",profile],"ab-merge-"+group)
        record={"identity":identity,"cache_hit":False,"profile":str(profile),"training_cpu_seconds":cpu,"rounds":rounds,
                "seconds":time.perf_counter()-start,"event_range":[first,len(self.events)],"last_command":[str(a) for a in args]}
        marker.write_text(json.dumps(record,indent=2)+"\n")
        return record

    def build(self):
        self.generate_source()
        for group in self.groups:
            train=self.compile_program(group,"generate")
            record=self.train_group(group,train)
            self.group_profiles[group]=record
            final=self.compile_program(group,"use",Path(record["profile"]))
            self.programs[group]={"train":train,"use":final}
        cpus=[r["training_cpu_seconds"] for r in self.group_profiles.values()]
        if max(cpus)-min(cpus)>self.cpu_budget*0.02:
            raise ValueError(f"frozen equal CPU budget tolerance exceeded: {cpus}")
        self.metadata["training"]=self.group_profiles
        self.metadata["programs"]={g:{k:str(p) for k,p in v.items()} for g,v in self.programs.items()}

    def check_answer(self,record,case,seed):
        pin=next(s for s in self.config["sources"] if s["name"]=="sqlite")
        if record["sqlite_version"]!=pin["version"] or record["sqlite_source_id"]!=pin["revision"]:
            raise ValueError("SQL workload linked the wrong SQLite version")
        if record["mode"]!=case or record["seed"]!=seed or record["rows"]!=100003 or record["verified_queries"]<=0:
            raise ValueError("SQL reference workload metadata differs")
        if case!="train" and record["verified_queries"]!=self.cases[case][1]:
            raise ValueError("SQL workload query count differs")

    def sample_sql(self,group,case,label,stage="use",env=None):
        seed,count=self.cases[case]
        binary=self.programs[group][stage]/"sqlite-workload"
        text=self.command([binary,case,str(seed),"100003",str(count)],label,cpu=True,env=env,timeout=300)
        record=json.loads(text)
        self.check_answer(record,case,seed)
        return {**record,"wall_seconds":self.events[-1]["seconds"],"log":self.events[-1]["log"]}

    def coverage(self,group,binary,profile,label):
        output=self.command(["/usr/lib/llvm-18/bin/llvm-cov","export",binary,f"--instr-profile={profile}","--name-regex=sqlite3VdbeExec"],
                            "coverage-"+label,timeout=180)
        data=json.loads(output)
        files=[f for unit in data["data"] for f in unit["files"] if Path(f["filename"]).resolve()==(self.generator/"sqlite3.c").resolve()]
        if len(files)!=1:
            raise ValueError(f"coverage source mapping differs: {label}")
        segments=files[0]["segments"]
        counts=[]
        for site in self.callsites:
            point=(site["line"],site["column"])
            preceding=[s for s in segments if (s[0],s[1])<=point]
            segment=preceding[-1] if preceding else None
            count=segment[2] if segment and segment[3] and (len(segment)<6 or not segment[5]) else None
            counts.append({**site,"region_execution_count":count,"segment":segment})
        record={"group":group,"label":label,"binary":str(binary),"profile":str(profile),"log":self.events[-1]["log"],
                "source":str(self.generator/"sqlite3.c"),"callsites":counts,
                "meaning":"source call-statement region execution counts, not proof of retained physical calls"}
        self.metadata.setdefault("coverage",[]).append(record)

    def inspect(self,group):
        build=self.programs[group]["use"]
        binary=build/"sqlite-workload"
        records=[]
        for path in sorted(build.glob("*.opt.yaml"))+sorted(build.glob("lto-inline.yaml*")):
            for r in inline_records(path):
                if r.get("Caller","").startswith("sqlite3VdbeExec") and r.get("Callee","").startswith("sqlite3BtreeIndexMoveto"):
                    records.append(r)
        calls=[]
        modules=[]
        for i,path in enumerate(sorted(build.rglob("*.opt.bc"))):
            output=self.work/f"sqlite-{group}-{i}.ll"
            self.command(["/usr/lib/llvm-18/bin/llvm-dis",path,"-o",output],"sqlite-backend-ir-"+group)
            text=output.read_text()
            if not re.search(r'^source_filename = .*sqlite3\.c',text,re.M):
                continue
            destination=self.logs/f"sqlite-{group}-core.ll"
            shutil.copyfile(output,destination)
            modules.append(str(destination))
            locations=dict(re.findall(r'^!(\d+) = (.*!DILocation.*)$',text,re.M))
            function=None
            for line in text.splitlines():
                if line.startswith("define "):
                    function=re.search(r'@([^ (]+)',line)[1]
                if function and function.startswith("sqlite3VdbeExec") and re.search(r'\b(?:call|invoke)\b.*@sqlite3BtreeIndexMoveto',line):
                    debug=re.search(r'!dbg !(\d+)',line)
                    calls.append({"caller":function,"instruction":line.strip(),"debug_location":locations.get(debug[1]) if debug else None})
        if not modules:
            raise ValueError("SQLite final core backend IR missing")
        names=self.command(["/usr/lib/llvm-18/bin/llvm-nm","-S","--defined-only",binary],"sqlite-symbols-"+group)
        selected=[]
        for line in names.splitlines():
            p=line.split()
            if len(p)==4 and p[3].startswith(("sqlite3VdbeExec","sqlite3BtreeIndexMoveto")):
                selected.append({"name":p[3],"bytes":int(p[1],16)})
        assembly=None
        if selected:
            self.command(["/usr/lib/llvm-18/bin/llvm-objdump","-d","--no-show-raw-insn","--disassemble-symbols="+",".join(s["name"] for s in selected),binary],"sqlite-assembly-"+group)
            assembly=self.events[-1]["log"]
        self.metadata.setdefault("code_evidence",[]).append({"group":group,"backend_ir":modules,"retained_target_calls":calls,
            "target_inline_decisions":records,"target_symbols":selected,"assembly_log":assembly})

    def run(self):
        # All PGO-use executables already exist before evaluation feedback is
        # gathered. Evaluation counters never update either training profile.
        for group in self.groups:
            trainer=self.programs[group]["train"]/("speedtest1" if group=="P0" else "sqlite-workload")
            self.coverage(group,trainer,Path(self.group_profiles[group]["profile"]),"training-"+group)
            self.inspect(group)
        for case in self.cases:
            directory=self.work/("evaluation-only-"+case)
            directory.mkdir()
            self.sample_sql("P0",case,"evaluation-audit-"+case,stage="train",env={"LLVM_PROFILE_FILE":str(directory/"evaluation-%p.profraw")})
            profile=directory/"evaluation-only.profdata"
            self.command([self.config["profdata"],"merge",*sorted(directory.glob("*.profraw")),"-o",profile],"evaluation-merge-"+case)
            self.coverage("evaluation-only",self.programs["P0"]["train"]/"sqlite-workload",profile,"evaluation-"+case)
        for batch in range(self.args.batches):
            for case in self.cases:
                records={}
                for group in self.groups:
                    record={"name":"sqlite-"+case,"variant":group,"batch":batch+1,"status":"collecting","samples":[],
                            "median_wall_seconds":0,"mad_wall_seconds":0}
                    records[group]=record
                    self.results.append(record)
                initial=list(self.groups) if batch%2==0 else list(reversed(self.groups))
                for round in range(self.args.repetitions+1):
                    order=initial[round%len(initial):]+initial[:round%len(initial)]
                    for group in order:
                        sample=self.sample_sql(group,case,f"sqlite-{case}-{group}-b{batch+1}-r{round}")
                        if round:
                            sample["round"]=round
                            records[group]["samples"].append(sample)
                    ZstdAB.aggregate(records.values(),round==self.args.repetitions)
                    self.save("measuring")
                    print(f"batch {batch+1}: sqlite-{case} round {round}/{self.args.repetitions}",flush=True)
