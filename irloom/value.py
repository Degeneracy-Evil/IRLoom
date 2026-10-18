"""Small standardized ridge model using only pre-decision semantic features."""
import math
import time
import numpy as np

from .decision import latency
from .search import HIGH_INDEX,replay

FEATURES=("log_m","log_n","log_reduction","log_operations","log_bytes","log_intensity",
          "stages","has_reduction","high1","log_tile","unroll4","log_inner_stride","log_inner_extent",
          "tile_remainder","tile_over_inner","work_x_tile","stride_x_tile","high_x_tile","log_bytes_L1","log_bytes_L2")


def features(r):
    program=r["program_id"]
    m,n=r["input"]["shape"][:2]
    a=HIGH_INDEX[r["high_choice"]]
    low=r["low_choice"]
    tile=int(low[4:]) if low.startswith("tile") else 0
    if program=="matmul":
        reduction=r["input"]["shape"][2]
        operations=2*m*n*reduction
        byte_count=4*(m*reduction+reduction*n+m*n)
        stride=n if a==0 else 1
        inner=reduction if a==0 else n
    elif program=="reduction":
        reduction=n
        operations=m*n
        byte_count=4*(m*n+m)
        stride=1 if a==0 else n
        inner=n if a==0 else m
    elif program=="pointwise_chain":
        reduction=1
        operations=2*m*n
        byte_count=12*m*n
        stride=1
        inner=n
    else:
        raise ValueError(program)
    log=lambda x:math.log2(max(x,1))
    work=log(operations)
    t=log(tile) if tile else 0
    stages=2 if program=="pointwise_chain" else 1
    caches={c["level"]:c["size"] for c in r["metadata"]["cpu_caches"] if c["type"]!="Instruction"}
    def capacity(level):
        value=caches[level]
        return float(value[:-1])*{"K":1024,"M":1024**2,"G":1024**3}[value[-1]]
    return [log(m),log(n),log(reduction),work,log(byte_count),math.log2(operations/byte_count),
            stages,int(program!="pointwise_chain"),a,t,int(low=="unroll4"),log(stride),log(inner),
            (inner%tile)/tile if tile else 0,tile/inner if tile else 0,work*t,log(stride)*t,a*t,
            math.log2(byte_count/capacity("1")),math.log2(byte_count/capacity("2"))]


def fit(records,alpha):
    x=np.asarray([features(r) for r in records],dtype=float)
    y=np.log([latency(r) for r in records])
    mean=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale<1e-12]=1
    z=(x-mean)/scale
    target=float(y.mean())
    weights=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-target))
    return {"alpha":alpha,"feature_names":FEATURES,"mean":mean.tolist(),"scale":scale.tolist(),
            "weights":weights.tolist(),"intercept":target,"parameter_count":len(weights)+1}


def predict(model,records):
    x=np.asarray([features(r) for r in records])
    return ((x-np.asarray(model["mean"]))/np.asarray(model["scale"])@np.asarray(model["weights"])+model["intercept"]).tolist()


def train(records):
    started=time.perf_counter()
    train_rows=[r for r in records if r["split"]=="train" and r["status"]=="ok"]
    val_rows=[r for r in records if r["split"]=="validation"]
    if not train_rows or not val_rows:
        raise ValueError("predeclared train and validation instances required")
    results=[]
    for alpha in (0.1,1.0,10.0):
        model=fit(train_rows,alpha)
        regrets=[]
        for case in sorted({r["case_id"] for r in val_rows}):
            pool=[r for r in val_rows if r["case_id"]==case]
            prediction=dict(zip([r["candidate_id"] for r in pool],predict(model,pool)))
            decision=replay(pool,"ridge",4,predictions=prediction)
            regrets.append(decision["latency_ms"]/min(map(latency,pool)))
        results.append({"alpha":alpha,"validation_geomean_regret":math.exp(sum(map(math.log,regrets))/len(regrets))})
    selected=min(results,key=lambda x:x["validation_geomean_regret"])
    model=fit(train_rows,selected["alpha"])
    log={"numpy_version":np.__version__,"train_case_ids":sorted({r["case_id"] for r in train_rows}),
         "validation_case_ids":sorted({r["case_id"] for r in val_rows}),"train_rows":len(train_rows),
         "alpha_selection":results,"training_seconds":time.perf_counter()-started,
         "target":"candidate log latency (ms); duplicates across timing batches stay in same split",
         "blind_labels_used":False,"scope":"pilot; no claim of program or hardware generalization"}
    return model,log
