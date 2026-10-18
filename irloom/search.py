"""M6 offline budgeted policies; outcomes become visible only through query()."""
import math
import random
import time

from .decision import costs, latency

LOW_ORDER=("none","tile16","unroll4","tile8","tile32","tile4")
HIGH_INDEX={"ijk":0,"ikj":1,"ij":0,"ji":1,"separate":0,"fused":1}


def key(record):
    return HIGH_INDEX[record["high_choice"]],record["low_choice"]


def replay(records,policy,k,seed=0,predictions=None):
    start=time.perf_counter()
    pool={key(r):r for r in records}
    observed={}
    trace=[]

    def query(candidate,reason):
        r=pool[candidate]
        value=latency(r)
        observed[candidate]=value
        trace.append({"candidate_id":r["candidate_id"],"reason":reason,
                      "latency_ms":value if math.isfinite(value) else None})

    rng=random.Random(seed)
    ordered=list(pool)
    fixed=[(a,b) for b in ("none","tile16","unroll4") for a in (0,1)]
    spread=[(a,b) for b in ("tile8","tile32","none") for a in (0,1)]
    program=records[0]["program_id"]
    _,n=records[0]["input"]["shape"][:2]
    reduction=records[0]["input"]["shape"][2] if program=="matmul" else 1
    preferred=0 if program=="reduction" or (program=="matmul" and n<=32 and reduction<=32) else 1
    static=[(preferred,"none"),(preferred,"tile16"),(preferred,"unroll4"),(1-preferred,"none"),
            (preferred,"tile8"),(preferred,"tile32")]
    if policy=="random":
        plan=rng.sample(ordered,k)
    elif policy=="stratified_random":
        plan=[x for a in (0,1) for x in rng.sample([x for x in ordered if x[0]==a],k//2)]
    else:
        plan={"fixed_m5":fixed,"fixed_spread":spread,"static":static}.get(policy)
    for step in range(k):
        remaining=[x for x in ordered if x not in observed]
        if plan is not None:
            chosen=plan[step]
            reason="predeclared allocation; labels not consulted"
        elif policy in ("greedy","adaptive","ridge_feedback") and step<2:
            chosen=(step,"none")
            reason="default branch probe"
        elif policy=="greedy":
            a=min((0,1),key=lambda a:observed[a,"none"])
            chosen=next((a,b) for b in LOW_ORDER if (a,b) in remaining)
            reason="commit to branch with best observed none"
        elif policy=="adaptive":
            def bound(a):
                seen=[v for x,v in observed.items() if x[0]==a]
                return min(seen)/(1+0.5/math.sqrt(len(seen)))
            branches=[a for a in (0,1) if any(x[0]==a for x in remaining)]
            a=min(branches,key=bound)
            improvement=observed.get((a,"tile16"),float("inf"))<observed[a,"none"]/1.05
            order=("tile16","tile8","tile32","tile4","unroll4") if improvement else ("tile16","unroll4","tile8","tile32","tile4")
            chosen=next((a,b) for b in order if (a,b) in remaining)
            reason="observed branch bound and observed tile16 improvement determine next probe"
        elif policy in ("ridge","ridge_feedback"):
            def score(x):
                base=predictions[pool[x]["candidate_id"]]
                if policy=="ridge_feedback":
                    residual=[math.log(v)-predictions[pool[c]["candidate_id"]] for c,v in observed.items()
                              if c[0]==x[0] and math.isfinite(v)]
                    base+=sum(residual)/(1+len(residual))
                return base
            chosen=min(remaining,key=score)
            reason="frozen ridge prediction" if policy=="ridge" else "prediction plus same-branch observed residual"
        else:
            raise ValueError(policy)
        if chosen not in remaining:
            raise ValueError("duplicate or illegal query")
        query(chosen,reason)
    best=min(observed,key=observed.get)
    totals={}
    for x in observed:
        for name,value in costs(pool[x]).items():
            totals[name]=totals.get(name,0)+value
    return {"policy":policy,"k":k,"seed":seed,"selected":pool[best]["candidate_id"],
            "latency_ms":observed[best] if math.isfinite(observed[best]) else None,
            "queries":trace,"costs":totals,"decision_seconds":time.perf_counter()-start}
