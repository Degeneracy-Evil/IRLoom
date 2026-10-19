"""Pinned real-program sources, bounded native builds and baseline measurements."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile

from .compiler import ROOT,probe

CONFIG=ROOT/"experiments/real-programs.json"


class Benchmarks:
    def __init__(self,args):
        self.args=args
        self.config=json.loads(CONFIG.read_text())
        self.storage=Path(self.config["storage_root"])
        self.storage.mkdir(parents=True,exist_ok=True)
        link=ROOT/"benchmark-data"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(self.storage,target_is_directory=True)
        if not link.is_symlink() or link.resolve()!=self.storage.resolve():
            raise ValueError(f"{link} must link to {self.storage}; existing files will not be replaced")
        self.root=link
        for name in ("downloads","sources","builds","install","data","tmp"):
            (self.root/name).mkdir(exist_ok=True)
        self.stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.logs=ROOT/"logs/benchmarks"/self.stamp
        self.logs.mkdir(parents=True)
        self.work=self.root/"tmp"/self.stamp
        self.work.mkdir(parents=True)
        self.events=[]
        self.results=[]
        self.started=time.perf_counter()
        self.cpu=min(os.sched_getaffinity(0))
        self.env={**os.environ,"TMPDIR":str(self.work),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}

    def command(self,args,name,cwd=None,timeout=600,cpu=False,input=None,env=None):
        args=[str(a) for a in args]
        event={"command":args,"cwd":str(cwd or ROOT),"log":str(self.logs/f"{len(self.events):03d}-{name}.log"),
               "affinity_cpu":self.cpu if cpu else None,"timeout_seconds":timeout}
        if input is not None:
            event["stdin"]=input
        if env:
            event["environment_overrides"]=env
        self.events.append(event)
        start=time.perf_counter()
        try:
            p=subprocess.run(args,cwd=cwd or ROOT,env={**self.env,**(env or {})},input=input,text=True,capture_output=True,
                             timeout=timeout,preexec_fn=(lambda:os.sched_setaffinity(0,{self.cpu})) if cpu else None)
            event.update(returncode=p.returncode,seconds=time.perf_counter()-start)
            Path(event["log"]).write_text(json.dumps(event)+"\n"+p.stdout+p.stderr)
            if p.returncode:
                raise RuntimeError(f"{name} failed ({p.returncode}); see {event['log']}")
            return p.stdout+p.stderr
        except (OSError,subprocess.TimeoutExpired) as exc:
            event.update(error=str(exc),seconds=time.perf_counter()-start)
            Path(event["log"]).write_text(json.dumps(event)+"\n")
            raise
        finally:
            (self.logs/"events.json").write_text(json.dumps(self.events,indent=2)+"\n")

    def source(self,name):
        return self.root/next(s["directory"] for s in self.config["sources"] if s["name"]==name)

    @staticmethod
    def zip_modes(archive,dest):
        # zipfile does not restore POSIX executable bits; SQLite's configure needs them.
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                mode=(info.external_attr>>16)&0o777
                relative=Path(*Path(info.filename).parts[1:])
                path=dest/relative
                if mode and path.exists():
                    path.chmod(mode)

    def fetch(self):
        dependencies={"llvm":{"llvm","clang","llvm-cmake","llvm-third-party"},
                      "test-suite":{"test-suite","llvm"},"zstd":{"zstd"},"sqlite":{"sqlite"}}
        required={"lld"}.union(*(dependencies[p] for p in self.args.project))
        for item in self.config["sources"]:
            if item["name"] not in required:
                continue
            dest=self.root/item["directory"]
            archive=self.root/"downloads"/item["archive"]
            marker=dest/".irloom-source.json"
            if marker.exists():
                recorded=json.loads(marker.read_text())
                if any(recorded.get(k)!=item.get(k) for k in ("name","version","revision","url","archive","directory")):
                    raise ValueError(f"source pin changed: {dest}; choose a new versioned source directory")
                if archive.suffix==".zip" and not os.access(dest/"configure",os.X_OK):
                    self.zip_modes(archive,dest)
                print(f"cached source: {item['name']}",flush=True)
                continue
            if dest.exists():
                raise ValueError(f"unmarked source directory: {dest}; refusing to overwrite")
            if not archive.exists():
                part=self.work/(item["archive"]+".partial")
                print(f"download: {item['name']} {item['version']}",flush=True)
                self.command(["curl","-fLsS","--retry","2","--connect-timeout","20","--max-time","600",
                              "--max-filesize","536870912",item["url"],"-o",part],"download-"+item["name"],timeout=650)
                part.rename(archive)
            staging=self.work/("extract-"+item["name"])
            staging.mkdir()
            if archive.suffix==".deb":
                self.command(["dpkg-deb","-x",archive,staging],"extract-"+item["name"])
                extracted=staging
            else:
                if archive.suffix==".zip":
                    with zipfile.ZipFile(archive) as z:
                        z.extractall(staging)
                else:
                    with tarfile.open(archive) as t:
                        t.extractall(staging,filter="data")
                children=list(staging.iterdir())
                if len(children)!=1 or not children[0].is_dir():
                    raise ValueError(f"unexpected source archive layout: {archive}")
                extracted=children[0]
                if archive.suffix==".zip":
                    self.zip_modes(archive,extracted)
            dest.parent.mkdir(parents=True,exist_ok=True)
            extracted.rename(dest)
            marker.write_text(json.dumps(item,indent=2)+"\n")
            print(f"source ready: {item['name']} ({archive.stat().st_size} downloaded bytes)",flush=True)

    def inventory(self):
        metadata={"host":probe(),"storage":str(self.storage),"link":str(self.root),"cpu":self.cpu,
                  "python":sys.version,"python_packages":{"psutil":package_version("psutil")},
                  "configuration":self.config,"tools":{},
                  "benchmark_environment":{k:self.env[k] for k in ("TMPDIR","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
                  "measurement_policy":{"warmups":1,"repetitions":self.args.repetitions,"affinity_cpu":self.cpu,
                                        "flags":self.config["flags"],"frequency":"uncontrolled","pmu":"not collected"}}
        for name,path in (("clang",self.config["compiler"]),("clang++",self.config["cxx"]),
                          ("llvm-profdata",self.config["profdata"]),("cmake","cmake"),("make","make"),
                          ("lld",self.source("lld")/"usr/lib/llvm-18/bin/ld.lld")):
            metadata["tools"][name]=self.command([path,"--version"],"version-"+name)
        metadata["tools"]["glibc"]=self.command(["getconf","GNU_LIBC_VERSION"],"version-glibc")
        metadata["tools"]["uv"]=self.command(["uv","--version"],"version-uv")
        metadata["clang_resource_dir"]=self.command([self.config["compiler"],"-print-resource-dir"],"clang-resource-dir").strip()
        if shutil.which("tclsh"):
            metadata["tools"]["tclsh"]=self.command(["tclsh"],"version-tcl",input="puts [info patchlevel]\n")
        disk=shutil.disk_usage(self.storage)
        metadata["storage_free_bytes"]=disk.free
        metadata["sources"]=[{**s,"archive_bytes":(self.root/"downloads"/s["archive"]).stat().st_size,
                              "license_paths":[str(self.root/s["directory"]/f) for f in s["license_files"]]}
                             for s in self.config["sources"] if (self.root/s["directory"]/".irloom-source.json").exists()]
        if not re.search(r"clang version "+re.escape(self.config["compiler_version"])+r"\b",metadata["tools"]["clang"]):
            raise ValueError("compiler version differs from the pinned baseline; use a separate toolchain configuration")
        (self.logs/"environment.json").write_text(json.dumps(metadata,indent=2)+"\n")
        self.metadata=metadata
        return metadata

    def build_dir(self,project):
        version=next(s["version"] for s in self.config["sources"] if s["name"]==project)
        path=self.root/"builds"/project/version/("clang-"+self.config["compiler_version"])/"o3"
        path.mkdir(parents=True,exist_ok=True)
        return path

    def cmake(self,project,source,options,targets):
        build=self.build_dir(project)
        common=["-G","Unix Makefiles","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DCMAKE_C_COMPILER={self.config['compiler']}",f"-DCMAKE_CXX_COMPILER={self.config['cxx']}",
                f"-DCMAKE_C_FLAGS_RELEASE={' '.join(self.config['flags'])}",f"-DCMAKE_CXX_FLAGS_RELEASE={' '.join(self.config['flags'])}",
                f"-DPython3_EXECUTABLE={ROOT/'scripts/benchmark-python.sh'}",
                f"-DPYTHON_EXECUTABLE={ROOT/'scripts/benchmark-python.sh'}"]
        settings={"source":str(source),"common":common,"options":options}
        marker=build/".irloom-configure.json"
        if not marker.exists() or json.loads(marker.read_text())!=settings:
            self.command(["cmake","-S",source,"-B",build,*common,*options],"configure-"+project,timeout=600)
            marker.write_text(json.dumps(settings,indent=2)+"\n")
        else:
            print(f"cached configure: {project}",flush=True)
        self.command(["cmake","--build",build,"--parallel",self.args.jobs,"--target",*targets],"build-"+project,timeout=1800)
        return build

    def build(self):
        for project in self.args.project:
            print(f"build: {project} (-O3, jobs={self.args.jobs})",flush=True)
            if project=="llvm":
                build=self.cmake("llvm",self.source("llvm"),[
                    "-DLLVM_ENABLE_PROJECTS=clang","-DLLVM_TARGETS_TO_BUILD=X86","-DLLVM_INCLUDE_TESTS=OFF",
                    "-DLLVM_INCLUDE_EXAMPLES=OFF","-DLLVM_INCLUDE_BENCHMARKS=OFF","-DCLANG_INCLUDE_TESTS=OFF",
                    "-DLLVM_ENABLE_ZLIB=OFF","-DLLVM_ENABLE_ZSTD=OFF","-DLLVM_ENABLE_LIBXML2=OFF",
                    "-DLLVM_ENABLE_TERMINFO=OFF",f"-DCMAKE_INSTALL_PREFIX={self.root/'install/llvm-source-18.1.3'}"],["LLVMDemangle"])
                driver=ROOT/"benchmarks/llvm_demangle.cpp"
                binary=build/"irloom-demangle"
                if not binary.exists() or binary.stat().st_mtime<max(driver.stat().st_mtime,(build/"lib/libLLVMDemangle.a").stat().st_mtime,(build/".irloom-configure.json").stat().st_mtime):
                    self.command([self.config["cxx"],*self.config["flags"],"-std=c++17",driver,
                                  "-I",self.source("llvm")/"include",build/"lib/libLLVMDemangle.a","-o",binary],"link-llvm-workload")
            elif project=="zstd":
                self.cmake("zstd",self.source("zstd")/"build/cmake",[
                    "-DZSTD_BUILD_SHARED=OFF","-DZSTD_BUILD_STATIC=ON","-DZSTD_BUILD_TESTS=OFF",
                    "-DZSTD_BUILD_CONTRIB=OFF","-DZSTD_ZLIB_SUPPORT=OFF","-DZSTD_LZMA_SUPPORT=OFF",
                    "-DZSTD_LZ4_SUPPORT=OFF","-DZSTD_PROGRAMS_LINK_SHARED=OFF"],["zstd"])
            elif project=="sqlite":
                build=self.build_dir("sqlite")
                flags=" ".join(self.config["flags"])
                settings={"CC":self.config["compiler"],"CFLAGS":flags}
                marker=build/".irloom-configure.json"
                if marker.exists() and json.loads(marker.read_text())!=settings:
                    raise ValueError("SQLite flags changed in an existing O3 build; use a separate configuration directory")
                if not marker.exists() or json.loads(marker.read_text())!=settings:
                    self.command([self.source("sqlite")/"configure","--disable-shared","--disable-readline","--disable-tcl",
                                  "--fts5","--rtree",f"--prefix={self.root/'install/sqlite-3.50.4-o3'}"],
                                 "configure-sqlite",cwd=build,env=settings)
                    marker.write_text(json.dumps(settings,indent=2)+"\n")
                self.command(["make",f"-j{self.args.jobs}","sqlite3","speedtest1"],"build-sqlite",cwd=build,timeout=1200)
            elif project=="test-suite":
                build=self.cmake("test-suite",self.source("test-suite"),[
                    "-DTEST_SUITE_SUBDIRS=MultiSource","-DTEST_SUITE_BENCHMARKING_ONLY=ON","-DTEST_SUITE_RUN_TYPE=ref",
                    "-DTEST_SUITE_COLLECT_CODE_SIZE=OFF","-DTEST_SUITE_COLLECT_COMPILE_TIME=OFF",
                    f"-DTEST_SUITE_LLVM_SIZE=/usr/lib/llvm-18/bin/llvm-size",
                    f"-DTEST_SUITE_LLVM_PROFDATA={self.config['profdata']}"],
                    ["7zip-benchmark","anagram"])
                # Generated configuration only: upstream source is unchanged.
                site=build/"lit.site.cfg"
                text=site.read_text().replace("config.test_exec_root = _directory",'config.test_exec_root = os.environ["IRLOOM_LIT_LOG_ROOT"]')
                if "# IRLoom no artifact hashes" not in text:
                    text=text.replace("lit_config.load_config(","# IRLoom no artifact hashes\nconfig.test_modules = [m for m in config.test_modules if m != 'hash']\n\nlit_config.load_config(")
                site.write_text(text)

    def measure(self,name,args,parser=None,cwd=None):
        self.command(args,"warmup-"+name,cwd=cwd,cpu=True,timeout=300)
        samples=[]
        for i in range(self.args.repetitions):
            output=self.command(args,f"{name}-{i}",cwd=cwd,cpu=True,timeout=300)
            sample={"wall_seconds":self.events[-1]["seconds"],"log":self.events[-1]["log"]}
            if parser:
                sample.update(parser(output))
            samples.append(sample)
        median=statistics.median(s["wall_seconds"] for s in samples)
        self.results.append({"name":name,"status":"pass","samples":samples,"median_wall_seconds":median,
                             "mad_wall_seconds":statistics.median(abs(s["wall_seconds"]-median) for s in samples),
                             "command":[str(x) for x in args],"cpu":self.cpu})
        print(f"measured: {name}, median process wall {median:.4f} s",flush=True)

    def toolchain_capabilities(self):
        build=self.build_dir("llvm")
        common=[self.config["cxx"],*self.config["flags"],"-std=c++17",ROOT/"benchmarks/llvm_demangle.cpp",
                "-I",self.source("llvm")/"include",build/"lib/libLLVMDemangle.a"]
        lld=self.source("lld")/"usr/lib/llvm-18/bin/ld.lld"
        self.command([*common,"-flto=thin",f"-fuse-ld={lld}","-Wl,--thinlto-jobs=2","-Wl,--threads=2",
                      f"-Wl,--thinlto-cache-dir={build/'thinlto-capability-cache'}","-o",self.work/"thinlto-demangle"],"thinlto-link")
        self.command([self.work/"thinlto-demangle","1000"],"thinlto-execute",cpu=True)
        self.command([*common,"-fprofile-instr-generate","-o",self.work/"pgo-generate"],"pgo-generate-build")
        self.command([self.work/"pgo-generate","1000"],"pgo-training-smoke",cpu=True,
                     env={"LLVM_PROFILE_FILE":str(self.work/"capability-%p.profraw")})
        profiles=sorted(self.work.glob("capability-*.profraw"))
        if not profiles:
            raise ValueError("PGO capability run produced no profile")
        merged=self.work/"capability.profdata"
        self.command([self.config["profdata"],"merge",*profiles,"-o",merged],"pgo-merge")
        self.command([*common,f"-fprofile-instr-use={merged}","-o",self.work/"pgo-use"],"pgo-use-build")
        self.command([self.work/"pgo-use","1000"],"pgo-use-execute",cpu=True)
        self.capabilities={"ThinLTO":"pass","PGO_generate_merge_use":"pass","PGO_instrumentation":"Clang frontend",
                           "scope":"driver capability checks linked to O3 LLVMDemangle, not whole-project LTO/PGO comparisons"}

    def run(self):
        for project in self.args.project:
            print(f"validate/run: {project}",flush=True)
            build=self.build_dir(project)
            if project=="llvm":
                self.toolchain_capabilities()
                self.measure("llvm-demangle",[build/"irloom-demangle"],json.loads)
                self.measure("clang-compile-llvm-tu",[self.config["cxx"],*self.config["flags"],"-std=c++17",
                             "-I",self.source("llvm")/"include","-I",build/"include","-c",
                             self.source("llvm")/"lib/Demangle/ItaniumDemangle.cpp","-o",self.work/"clang-workload.o"])
            elif project=="zstd":
                binary=build/"programs/zstd"
                info=self.command([binary,"--version"],"zstd-version")
                version=next(s["version"] for s in self.config["sources"] if s["name"]=="zstd")
                if "v"+version not in info:
                    raise ValueError("unexpected Zstd executable version")
                self.metadata.setdefault("program_versions",{})["zstd"]=info.strip()
                corpus=self.root/"data"/(self.source("zstd").name+"-code-8MiB.bin")
                files=sorted((self.source("zstd")/"lib").rglob("*.c"))
                if not corpus.exists():
                    text=b"\n".join(p.read_bytes() for p in files)
                    corpus.write_bytes((text*(8*1024*1024//len(text)+1))[:8*1024*1024])
                    corpus.with_suffix(".json").write_text(json.dumps({"source":str(self.source("zstd")),"bytes":8*1024*1024,
                        "recipe":"sorted lib/**/*.c concatenated with newline, repeated and truncated",
                        "members":[str(p.relative_to(self.source("zstd"))) for p in files]},indent=2)+"\n")
                packed,restored=self.work/"corpus.zst",self.work/"corpus.restored"
                self.command([binary,"-q","--single-thread","-3",corpus,"-o",packed],"zstd-compress-check",cpu=True)
                self.command([binary,"-q","-d",packed,"-o",restored],"zstd-decompress-check",cpu=True)
                if corpus.read_bytes()!=restored.read_bytes():
                    raise ValueError("Zstd roundtrip mismatch")
                def throughput(text):
                    matches=re.findall(r"([0-9.]+) MB/s,\s*([0-9.]+) MB/s",text)
                    if not matches:
                        raise ValueError("Zstd benchmark throughput missing")
                    a,b=matches[-1]
                    return {"compression_MB_s":float(a),"decompression_MB_s":float(b)}
                self.measure("zstd-b3",[binary,"-b3","-i1","--single-thread",corpus],throughput)
                self.results[-1]["input"]={"path":str(corpus),"bytes":corpus.stat().st_size,"recipe":str(corpus.with_suffix('.json'))}
            elif project=="sqlite":
                binary=build/"sqlite3"
                info=self.command([binary,":memory:","SELECT sqlite_version()||'|'||sqlite_source_id();"],"sqlite-version")
                pin=next(s for s in self.config["sources"] if s["name"]=="sqlite")
                if info.strip()!=pin["version"]+"|"+pin["revision"]:
                    raise ValueError("SQLite version/SOURCE_ID differs from the pinned source")
                self.metadata.setdefault("program_versions",{})["sqlite"]=info.strip()
                sql=""".bail on
CREATE TABLE t(k INTEGER PRIMARY KEY,v TEXT);
INSERT INTO t VALUES(1,'a'),(2,'b'),(3,'c');
BEGIN;
INSERT INTO t VALUES(4,'rollback');
ROLLBACK;
SELECT count(*),sum(k),group_concat(v,'') FROM (SELECT * FROM t ORDER BY k);
SELECT json_extract('{"a":7}','$.a');
CREATE VIRTUAL TABLE ft USING fts5(content);
INSERT INTO ft VALUES('compiler benchmark');
SELECT count(*) FROM ft WHERE ft MATCH 'compiler';
CREATE VIRTUAL TABLE rt USING rtree(id,x0,x1,y0,y1);
INSERT INTO rt VALUES(1,0,1,0,1);
SELECT count(*) FROM rt WHERE x0=0;
PRAGMA integrity_check;
"""
                output=self.command([binary,":memory:"],"sqlite-sql-correctness",input=sql,cpu=True)
                if output.strip().splitlines()!=["3|6|abc","7","1","1","ok"]:
                    raise ValueError("SQLite SQL/reference result mismatch")
                database=self.work/"speedtest1-validation.db"
                self.command([build/"speedtest1","--size","5","--testset","main",database],"speedtest1-file-check",cpu=True)
                check=self.command([binary,database,"PRAGMA integrity_check; SELECT count(*)>0 FROM t6;"],"speedtest1-file-integrity",cpu=True)
                if check.strip().splitlines()!=["ok","1"]:
                    raise ValueError("speedtest1 output database failed integrity/row check")
                def total(text):
                    match=re.search(r"TOTAL\.+\s*([0-9.]+)s",text)
                    if not match:
                        raise ValueError("speedtest1 did not report TOTAL")
                    return {"speedtest_total_seconds":float(match[1])}
                self.measure("sqlite-speedtest1",[build/"speedtest1","--memdb","--size","25","--testset","main"],total)
            elif project=="test-suite":
                executable=build/"MultiSource/Benchmarks/7zip/7zip-benchmark"
                def throughput(text):
                    match=re.search(r"(?m)^\s*18:\s+(\d+)\s+\d+\s+\d+\s+\d+\s*\|\s*(\d+)",text)
                    if not match:
                        raise ValueError("7zip benchmark results missing")
                    return {"compression_reported_KB_s":int(match[1]),"decompression_reported_KB_s":int(match[2])}
                self.measure("test-suite-7zip",[executable,"b","1","-mmt=1","-md=18"],throughput)
                samples=[]
                for i in range(self.args.repetitions+1):
                    output_dir=self.logs/f"lit-{i}"
                    output_dir.mkdir()
                    (output_dir/"tools").symlink_to(build/"tools",target_is_directory=True)
                    result=self.logs/f"lit-{i}.json"
                    self.command(["uv","run","python",self.source("llvm")/"utils/lit/lit.py","-sv","-j1","--timeout=60","-o",result,
                                  build/"MultiSource/Benchmarks/Ptrdist/anagram/anagram.test"],f"lit-anagram-{i}",cpu=True,
                                 env={"IRLOOM_LIT_LOG_ROOT":str(output_dir)})
                    record=json.loads(result.read_text())
                    if not record["tests"] or any(t["code"]!="PASS" or "hash" in t.get("metrics",{}) for t in record["tests"]):
                        raise ValueError("lit reference-output validation failed or artifact hashing enabled")
                    if i:
                        samples.append({"wall_seconds":self.events[-1]["seconds"],"tests":record["tests"],"log":self.events[-1]["log"]})
                median=statistics.median(s["wall_seconds"] for s in samples)
                self.results.append({"name":"test-suite-anagram","status":"pass","samples":samples,"cpu":self.cpu,
                                     "median_wall_seconds":median,"mad_wall_seconds":statistics.median(abs(s["wall_seconds"]-median) for s in samples)})

    def save(self,status,error=None):
        result={"status":status,"error":error,"action":self.args.action,"projects":self.args.project,
                "jobs":self.args.jobs,"repetitions":self.args.repetitions,"wall_seconds":time.perf_counter()-self.started,
                "metadata":getattr(self,"metadata",None),"capabilities":getattr(self,"capabilities",{}),
                "results":self.results,"events":self.events}
        (self.logs/"results.json").write_text(json.dumps(result,indent=2)+"\n")
        if hasattr(self,"metadata"):
            (self.logs/"environment.json").write_text(json.dumps(self.metadata,indent=2)+"\n")
        lines=["# Real-program environment run", "",f"Status: {status}; jobs={self.args.jobs}; CPU={self.cpu}; flags={' '.join(self.config['flags'])}","",
               "Process wall times include startup; internal workload metrics are retained in results.json and raw logs.","",
               "| Workload | Repetitions | Median wall s | MAD wall s |","| --- | ---: | ---: | ---: |"]
        for r in self.results:
            lines.append(f"| {r['name']} | {len(r['samples'])} | {r['median_wall_seconds']:.6f} | {r['mad_wall_seconds']:.6f} |")
        if error:
            lines += ["",str(error)]
        (self.logs/"summary.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="all",choices=("all","fetch","probe","build","run"))
    parser.add_argument("--project",nargs="+",choices=("llvm","zstd","sqlite","test-suite"),default=["llvm","zstd","sqlite","test-suite"])
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--repetitions",type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.jobs<=8 or args.repetitions<1:
        parser.error("jobs must be 1..8; repetitions must be positive")
    bench=Benchmarks(args)
    print(f"logs: {bench.logs}",flush=True)
    try:
        bench.fetch()
        if args.action!="fetch":
            bench.inventory()
        if args.action in ("all","build","run"):
            bench.build()
        if args.action in ("all","run"):
            bench.run()
        bench.save("pass")
    except Exception as exc:
        bench.save("failed",str(exc))
        raise
    print(f"status: pass\nsummary: {bench.logs/'summary.md'}",flush=True)


if __name__=="__main__":
    main()
