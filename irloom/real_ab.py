"""Bounded causal experiments, isolated from the R1–R3 source/build caches."""
import difflib
import json
from pathlib import Path
import re
import resource
import shutil
import statistics
import time

from .compiler import ROOT
from .real_baselines import RealBaselines
from .real_audit import inline_records


def paired_comparisons(records):
    """Round-matched speed ratios, kept separate by input/context/batch."""
    if not records:
        return []
    reference="original" if "treatment" in records[0] else "P0"
    indexed={(r["name"],r.get("context"),r["batch"],r.get("treatment",r["variant"])):r for r in records}
    output=[]
    for record in records:
        group=record.get("treatment",record["variant"])
        if group==reference:
            continue
        baseline=indexed[record["name"],record.get("context"),record["batch"],reference]
        reference_samples={s["round"]:s for s in baseline["samples"]}
        metrics=list(record.get("metrics",{}))
        if reference=="P0":
            metrics.append("wall_seconds")
        for metric in metrics:
            values=[]
            for sample in record["samples"]:
                a,b=sample[metric],reference_samples[sample["round"]][metric]
                values.append(100*((a/b if metric.endswith("MB_s") else b/a)-1))
            med=statistics.median(values)
            output.append({"name":record["name"],"context":record.get("context"),"batch":record["batch"],"group":group,
                "reference":reference,"metric":metric,"paired_speed_percent":values,"median_percent":med,
                "mad_percent":statistics.median(abs(v-med) for v in values),"faster_rounds":sum(v>0 for v in values)})
    return output


def save_comparisons(bench):
    comparisons=paired_comparisons(bench.results)
    (bench.logs/"comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
    if comparisons:
        lines=["", "## Paired speed ratios", "", "Positive is faster; round-matched ratios, not a significance test.", "",
               "| Input | Context | Batch | Group | Metric | Median % | MAD % |", "| --- | --- | ---: | --- | --- | ---: | ---: |"]
        for r in comparisons:
            lines.append(f"| {r['name']} | {r['context'] or 'PGO+ThinLTO'} | {r['batch']} | {r['group']} | {r['metric']} | {r['median_percent']:+.3f} | {r['mad_percent']:.3f} |")
        with (bench.logs/"summary.md").open("a") as stream:
            stream.write("\n".join(lines)+"\n")


class ZstdAB(RealBaselines):
    treatments=("original","always-inline","noinline")
    contexts=("thinlto-native","pgo-thinlto-native")

    def __init__(self,args):
        super().__init__(args)
        self.treatment="original"
        self.copies={}
        self.group_profiles={}
        self.artifacts=[]
        self.args.project=["zstd"]
        self.args.variants=list(self.treatments)

    def command(self,*args,**kwargs):
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            return super().command(*args,**kwargs)
        finally:
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            if self.events:
                self.events[-1].update(user_cpu_seconds=after.ru_utime-before.ru_utime,system_cpu_seconds=after.ru_stime-before.ru_stime)
                (self.logs/"events.json").write_text(json.dumps(self.events,indent=2)+"\n")

    def source(self,name):
        if name=="zstd" and self.copies:
            return self.copies[self.treatment]
        return super().source(name)

    def build_dir(self,project):
        path=self.root/"builds"/project/"1.5.7"/("clang-"+self.config["compiler_version"])/"inline-ab-v1"/self.treatment/self.variant
        if self.variant=="pgo-thinlto-native" and project in self.profiles:
            path=path/Path(self.profiles[project]["profile"]).parent.name
        path.mkdir(parents=True,exist_ok=True)
        return path

    def select(self,variant,profile=None):
        super().select(variant,profile)
        if variant in self.contexts:
            self.link_flags += ["-Wl,--save-temps=opt"]
            self.cmake_extra[-1]="-DCMAKE_EXE_LINKER_FLAGS="+" ".join(self.link_flags)

    def prepare_sources(self):
        original=super().source("zstd")
        relative=Path("lib/compress/zstd_compress.c")
        declaration="static size_t ZSTD_buildSeqStore(ZSTD_CCtx* zc, const void* src, size_t srcSize)"
        text=(original/relative).read_text()
        if text.count(declaration)!=1:
            raise ValueError("pinned Zstd declaration differs; refusing an approximate patch")
        for group in self.treatments:
            patch=ROOT/"experiments/patches"/f"zstd-1.5.7-{group}.patch"
            wanted=text if group=="original" else text.replace(declaration,declaration.replace("static size_t",f"static __attribute__(({group.replace('-','_')})) size_t"))
            # Clang attribute spelling differs from the treatment's label.
            if group=="always-inline":
                wanted=text.replace(declaration,declaration.replace("static size_t","static __attribute__((always_inline)) size_t"))
            difference="".join(difflib.unified_diff(text.splitlines(True),wanted.splitlines(True),fromfile="a/"+str(relative),tofile="b/"+str(relative),n=0))
            if group!="original":
                # Header hunk spelling is generated by difflib; content must match.
                changed=[line for line in difference.splitlines() if line.startswith(("-static","+static"))]
                if changed!=[line for line in patch.read_text().splitlines() if line.startswith(("-static","+static"))]:
                    raise ValueError("checked-in patch and applied declaration differ")
            destination=self.root/"sources/zstd-1.5.7-inline-ab-v1"/group
            marker=destination/".irloom-ab.json"
            identity={"upstream":json.loads((original/".irloom-source.json").read_text()),"treatment":group,"patch":difference}
            if marker.exists():
                if json.loads(marker.read_text())!=identity or (destination/relative).read_text()!=wanted:
                    raise ValueError(f"modified A/B source cache: {destination}")
            else:
                if destination.exists():
                    raise ValueError(f"unmarked A/B source directory: {destination}")
                staging=self.work/("source-"+group)
                shutil.copytree(original,staging)
                (staging/relative).write_text(wanted)
                (staging/".irloom-ab.json").write_text(json.dumps(identity,indent=2)+"\n")
                destination.parent.mkdir(exist_ok=True)
                staging.rename(destination)
            self.copies[group]=destination
        self.metadata["source_treatments"]={g:str(p) for g,p in self.copies.items()}
        prior=json.loads((ROOT/"experiments/reports/real-programs-audit.json").read_text())
        self.metadata["existing_baseline_decisions"]={k:v for k,v in prior["selected_decisions"].items() if k.startswith("zstd/")}
        native=self.root/"builds/zstd/1.5.7"/("clang-"+self.config["compiler_version"])/"formal-o3-native/programs/zstd"
        if native.exists():
            names=self.command(["/usr/lib/llvm-18/bin/llvm-nm","--defined-only",native],"existing-o3-native-symbols")
            selected=[line.split()[-1] for line in names.splitlines() if re.search(r'\bZSTD_(buildSeqStore|compressBlock_internal)(?:\.|$)',line)]
            if selected:
                self.command(["/usr/lib/llvm-18/bin/llvm-objdump","-d","--no-show-raw-insn","--disassemble-symbols="+",".join(selected),native],"existing-o3-native-assembly")
                self.metadata["existing_o3_native_assembly"]=self.events[-1]["log"]
        self.metadata["protocol"]={"target":"ZSTD_buildSeqStore; attribute only; all source calls affected",
            "contexts":self.contexts,"treatments":self.treatments,"batches":self.args.batches,
            "train":"fresh compatible profile per treatment; software-train and records-train, -b3 -i1 each",
            "evaluation":"unchanged formal-v1 software-eval and records-eval, -b3 -i2",
            "order":"rotate context/treatment order per round, reverse initial order in alternate batches"}

    def build(self):
        self.prepare_inputs()  # Always derive corpora from the unmodified official source.
        self.prepare_sources()
        for group in self.treatments:
            self.treatment=group
            self.profiles={}
            self.build_one("zstd","thinlto-native")
            self.build_records[-1]["treatment"]=group
            self.build_one("zstd","pgo-train")
            self.build_records[-1]["treatment"]=group
            profile=self.train("zstd")
            self.group_profiles[group]=dict(self.profiles["zstd"])
            self.build_one("zstd","pgo-thinlto-native",profile)
            self.build_records[-1]["treatment"]=group
        self.metadata["profiles_by_treatment"]=self.group_profiles

    def activate(self,group,context):
        self.treatment=group
        self.variant=context
        self.profiles={"zstd":self.group_profiles[group]}

    def save(self,status,error=None):
        super().save(status,error)
        if status=="pass" and self.results:
            save_comparisons(self)

    def inspect(self,group,context):
        self.activate(group,context)
        build=self.build_dir("zstd")
        binary=build/"programs/zstd"
        prefix=f"{group}-{context}"
        symbols=self.command(["/usr/lib/llvm-18/bin/llvm-nm","-S","--defined-only",binary],"symbols-"+prefix)
        selected=[]
        for line in symbols.splitlines():
            fields=line.split()
            if len(fields)==4 and re.match(r"ZSTD_(buildSeqStore|compressBlock_internal|compressContinue_internal|compress_frameChunk|compressBlock_splitBlock)",fields[3]):
                selected.append({"name":fields[3],"bytes":int(fields[1],16)})
        names=[s["name"] for s in selected]
        if names:
            self.command(["/usr/lib/llvm-18/bin/llvm-objdump","-d","--no-show-raw-insn","--disassemble-symbols="+",".join(names),binary],"assembly-"+prefix)
            assembly_log=self.events[-1]["log"]
        else:
            assembly_log=None
        decisions=[]
        for path in sorted(build.rglob("*.opt.yaml"))+sorted(build.glob("lto-inline.yaml*")):
            if not path.is_file():
                continue
            for record in inline_records(path):
                if record.get("Callee","").startswith("ZSTD_buildSeqStore"):
                    decisions.append(record)
        # Read actual optimized backend modules, including clone names and call
        # sites. A pre-link AlwaysInline remark alone is insufficient evidence.
        modules=[]
        for i,path in enumerate(sorted(build.rglob("*.opt.bc"))):
            output=self.work/f"{prefix}-{i}.ll"
            self.command(["/usr/lib/llvm-18/bin/llvm-dis",path,"-o",output],"backend-ir-"+prefix)
            text=output.read_text()
            if re.search(r'^source_filename = .*zstd_compress\.c',text,re.M):
                destination=self.logs/(prefix+"-zstd_compress.ll")
                shutil.copyfile(output,destination)
                calls=[]
                function=None
                for line in text.splitlines():
                    if line.startswith("define "):
                        function=re.search(r'@([^ (]+)',line)[1]
                    if re.search(r'\b(?:call|invoke)\b.*@ZSTD_buildSeqStore',line):
                        calls.append({"caller":function,"instruction":line.strip()})
                modules.append({"bitcode":str(path),"ir":str(destination),"remaining_helper_calls":calls})
        if not modules:
            raise ValueError(f"no final zstd_compress backend module found: {build}")
        if group=="always-inline" and (any(m["remaining_helper_calls"] for m in modules) or any(s["name"].startswith("ZSTD_buildSeqStore") for s in selected)):
            raise ValueError("always_inline was not fully applied in the final program")
        if group=="always-inline" and not {4363,4399,4512}.issubset({e.get("line") for e in decisions if e.get("Name")=="AlwaysInline"}):
            raise ValueError("missing AlwaysInline evidence for one of the three source callsites")
        if group=="noinline" and not any(m["remaining_helper_calls"] for m in modules):
            raise ValueError("noinline produced no retained helper calls")
        record={"treatment":group,"context":context,"binary":str(binary),"selected_symbols":selected,
                "assembly_log":assembly_log,"backend_modules":modules,"inline_decisions":decisions}
        (self.logs/(prefix+"-evidence.json")).write_text(json.dumps(record,indent=2)+"\n")
        self.artifacts.append(record)
        self.metadata["code_evidence"]=self.artifacts

    def run(self):
        variants=[(g,c) for c in self.contexts for g in self.treatments]
        for group,context in variants:
            self.activate(group,context)
            # Unique validation paths without changing the build context.
            for name in ("software-eval","records-eval"):
                src=self.inputs[name]
                binary=self.build_dir("zstd")/"programs/zstd"
                packed=self.work/f"{group}-{context}-{name}.zst"
                restored=packed.with_suffix(".restored")
                self.command([binary,"-q","--single-thread","-3",src,"-o",packed],"roundtrip-compress",cpu=True)
                self.command([binary,"-q","-d",packed,"-o",restored],"roundtrip-decompress",cpu=True)
                if restored.read_bytes()!=src.read_bytes():
                    raise ValueError("A/B roundtrip mismatch")
            self.inspect(group,context)
        for batch in range(self.args.batches):
            for name in ("software-eval","records-eval"):
                records={}
                for group,context in variants:
                    record={"name":"zstd-"+name,"variant":group+"/"+context,"treatment":group,"context":context,"batch":batch+1,
                            "status":"collecting","samples":[],"median_wall_seconds":0,"mad_wall_seconds":0}
                    self.results.append(record)
                    records[group,context]=record
                order=list(reversed(variants)) if batch%2 else variants
                for round in range(self.args.repetitions+1):
                    rotated=order[round%len(order):]+order[:round%len(order)]
                    for group,context in rotated:
                        self.activate(group,context)
                        binary=self.build_dir("zstd")/"programs/zstd"
                        sample=self.sample(f"zstd-{name}-{group}-{context}-b{batch+1}-r{round}",
                                           [binary,"-b3","-i2","--single-thread",self.inputs[name]])
                        if round:
                            sample["round"]=round
                            records[group,context]["samples"].append(sample)
                    self.aggregate(records.values(),round==self.args.repetitions)
                    self.save("measuring")
                    print(f"batch {batch+1}: {name} round {round}/{self.args.repetitions}",flush=True)

    @staticmethod
    def aggregate(records,done):
        for record in records:
            samples=record["samples"]
            if not samples:
                continue
            for key in ("wall_seconds","compression_MB_s","decompression_MB_s","compute_seconds"):
                if key not in samples[0]:
                    continue
                values=[s[key] for s in samples]
                med=statistics.median(values)
                mad=statistics.median(abs(v-med) for v in values)
                if key=="wall_seconds":
                    record.update(median_wall_seconds=med,mad_wall_seconds=mad)
                else:
                    record.setdefault("metrics",{})[key]={"median":med,"mad":mad}
            if done:
                record["status"]="pass"
