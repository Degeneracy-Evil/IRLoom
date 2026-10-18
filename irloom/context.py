"""Small, explicit IR windows at the affine decision boundary (before tiling)."""
from pathlib import Path
import re
import time

OPS=("affine.for","affine.load","affine.store","arith.mulf","arith.addf",
     "memref.alloc","memref.dealloc","memref.copy")


def counts(text):
    values=[len(re.findall(r"\b"+re.escape(op)+r"\b",text)) for op in OPS]
    variables=re.findall(r"affine\.for (%\w+) =",text)
    reverse=0
    if len(variables)>=2:
        reverse=len(re.findall(r"affine\.load [^\n]*\["+re.escape(variables[1])+r", "+re.escape(variables[0])+r"\]",text))
    return values+[reverse]


def first_nest(text):
    start=text.index("affine.for")
    opening=text.index("{",start)
    depth=1
    end=opening+1
    while depth:
        depth+=(text[end]=="{")-(text[end]=="}")
        end+=1
    return text[start:end]


def normalize_window(text):
    names={}
    def rename(match):
        return names.setdefault(match[0],f"%v{len(names)}")
    return re.sub(r"%[\w]+",rename,text)


def extract(record):
    stages={s["stage"]:Path(s["path"]) for s in record["snapshots"]}
    start=time.perf_counter()
    affine=stages["affine"].read_text()
    local=first_nest(affine)
    l0=counts(local)
    local_seconds=time.perf_counter()-start
    start=time.perf_counter()
    current=counts(affine)
    full_seconds=local_seconds+time.perf_counter()-start
    start=time.perf_counter()
    high=stages["high"].read_text()
    upstream=[len(re.findall(r"%d\b",high))-1,high.count("linalg.generic"),
              int("(j,i)" in high),int('"reduction"' in high)]
    history_seconds=full_seconds+time.perf_counter()-start
    return {"ops":OPS+("reversed_loads",),"local_current":l0,"full_current":current,"upstream":upstream,
            "upstream_names":["target_uses","generic_ops","transposed_consumer","reduction_consumer"],
            "local_window":normalize_window(local),
            "stages":{"local_current":"affine:first outer loop nest","full_current":"affine:whole function","upstream":"high:whole graph"},
            "seconds":{"L0":local_seconds,"L1":full_seconds,"L2":history_seconds}}
