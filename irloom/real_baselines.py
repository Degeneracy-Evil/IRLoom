"""Representative native workloads; reuse source, build and logging machinery."""
import io
import json
import random
import re
import statistics
import tarfile

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

    def build_dir(self,project):
        path=super().build_dir(project).parent/("formal-"+self.variant)
        path.mkdir(exist_ok=True)
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
                    with tarfile.open(path,"w",format=tarfile.USTAR_FORMAT) as archive:
                        for p in members:
                            payload=p.read_bytes()
                            info=tarfile.TarInfo(str(p.relative_to(self.source("zstd"))))
                            info.size=len(payload)
                            info.mode=0o644
                            archive.addfile(info,io.BytesIO(payload))
                    path.with_suffix(".json").write_text(json.dumps(recipe,indent=2)+"\n")
                self.inputs["software-"+role]=path
            for role,seed,count in (("train",20261020,30000),("eval",20261021,300000)):
                path=data/f"records-{role}.jsonl"
                recipe={"role":role,"seed":seed,"records":count,"generator":"formal-v1 random.Random(seed), timestamp/id/tenant/status/value/token JSONL",
                        "distribution":"structured records with changing numeric fields and 96-bit random tokens"}
                if not path.exists():
                    rng=random.Random(seed)
                    with path.open("w") as output:
                        for i in range(count):
                            record={"timestamp":1700000000+i,"id":i,"tenant":rng.randrange(1000),
                                    "status":rng.choice([200,200,200,404,503]),"value":rng.randrange(1000000),"token":f"{rng.getrandbits(96):024x}"}
                            output.write(json.dumps(record,separators=(",",":"))+"\n")
                    path.with_suffix(".recipe.json").write_text(json.dumps(recipe,indent=2)+"\n")
                self.inputs["records-"+role]=path
        self.metadata["formal_inputs"]={k:{"path":str(v),"bytes":v.stat().st_size} for k,v in self.inputs.items()}

    def build(self):
        self.prepare_inputs()
        super().build()
        self.metadata["scope"]="formal representative workloads; LLVM compiler bootstrap excluded"
        self.metadata["variant"]=self.variant

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
        self.validate()
        workloads=self.workloads()
        names=[n for n,_ in workloads]+(list(self.suite_tests) if "test-suite" in self.args.project else [])
        for name in names:
            samples=[]
            args=next((a for n,a in workloads if n==name),None)
            for i in range(self.args.repetitions+1):
                sample=self.sample(name,args) if args else self.lit(name,f"lit-{name}-{i}")
                if i:
                    samples.append(sample)
            median=statistics.median(s["wall_seconds"] for s in samples)
            self.results.append({"name":name,"variant":self.variant,"status":"pass","cpu":self.cpu,"samples":samples,
                                 "median_wall_seconds":median,"mad_wall_seconds":statistics.median(abs(s["wall_seconds"]-median) for s in samples)})
            self.save("running")
            print(f"measured: {name}, median wall {median:.3f}s",flush=True)
