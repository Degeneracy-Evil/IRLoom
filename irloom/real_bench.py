"""Pinned real-program sources, bounded native builds and baseline measurements."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
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
        self.cpu=min(os.sched_getaffinity(0))
        self.env={**os.environ,"TMPDIR":str(self.work),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}

    def command(self,args,name,cwd=None,timeout=600,cpu=False,input=None):
        args=[str(a) for a in args]
        event={"command":args,"cwd":str(cwd or ROOT),"log":str(self.logs/f"{len(self.events):03d}-{name}.log")}
        self.events.append(event)
        start=time.perf_counter()
        try:
            p=subprocess.run(args,cwd=cwd or ROOT,env=self.env,input=input,text=True,capture_output=True,
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
        for item in self.config["sources"]:
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
                  "python":sys.version,"configuration":self.config,"tools":{}}
        for name,path in (("clang",self.config["compiler"]),("clang++",self.config["cxx"]),
                          ("llvm-profdata",self.config["profdata"]),("cmake","cmake"),("make","make"),
                          ("lld",self.source("lld")/"usr/lib/llvm-18/bin/ld.lld")):
            metadata["tools"][name]=self.command([path,"--version"],"version-"+name)
        metadata["sources"]=[{**s,"archive_bytes":(self.root/"downloads"/s["archive"]).stat().st_size,
                              "license_paths":[str(self.root/s["directory"]/f) for f in s["license_files"]]}
                             for s in self.config["sources"]]
        (self.logs/"environment.json").write_text(json.dumps(metadata,indent=2)+"\n")
        return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="fetch",choices=("fetch","probe"))
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--repetitions",type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.jobs<=8 or args.repetitions<1:
        parser.error("jobs must be 1..8; repetitions must be positive")
    bench=Benchmarks(args)
    bench.fetch()
    if args.action=="probe":
        bench.inventory()
    print(f"logs: {bench.logs}",flush=True)


if __name__=="__main__":
    main()
