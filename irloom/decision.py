"""Transparent offline policies. Only query() may expose a candidate label."""
import statistics
import time


def latency(record):
    if record["status"] != "ok" or record.get("correctness", {}).get("status") != "pass":
        return float("inf")
    return statistics.median(record["timings_ms"])


def costs(record):
    samples = record.get("timings_ms", [])
    result = {"candidate_compiles": 1, "compile_seconds": record.get("compile_seconds", 0),
              "timing_batches": len(samples),
              "native_timed_seconds": sum(samples)*record.get("measurement", {}).get("iterations", 0)/1000}
    events = record.get("events")
    if events is not None:
        for stage in ("correctness", "calibration", "timing"):
            selected = [e for e in events if e["stage"] == stage]
            result[stage+"_invocations"] = len(selected)
            result[stage+"_wall_seconds"] = sum(e["seconds"] for e in selected)
        result["compiler_processes"] = sum(e["stage"] not in ("correctness", "calibration", "timing", "disassembly") for e in events)
        # measure mode does one initial correctness call + three warmups outside the timer.
        result["native_kernel_calls"] = result["correctness_invocations"] + sum(
            int(e["command"][-1])+4 for e in events if e["stage"] in ("calibration", "timing"))
        result["charged_wall_seconds"] = result["compile_seconds"] + sum(result[s+"_wall_seconds"] for s in ("correctness", "calibration", "timing"))
    elif "costs" in record:
        result.update(record["costs"])
    return result


def preferred_high(program, shape):
    # Frozen using M4 only: small matmul keeps original order; otherwise contiguous j.
    # Row reduction keeps its input's contiguous reduction dimension innermost.
    if program == "reduction":
        return "ij"
    _, n, k = shape
    return "ijk" if n <= 32 and k <= 32 else "ikj"


def simulate(records, policy):
    started = time.perf_counter()
    grid = {(r["high_choice"], r["low_choice"]): r for r in records if r.get("role") == "factorial"}
    first = records[0]
    features = first.get("decision_features", {})
    program = features.get("program", first.get("program_id", "matmul" if len(first["input"]["shape"])==3 else "reduction"))
    shape = features.get("shape", first["input"]["shape"])
    highs = ("ijk", "ikj") if program == "matmul" else ("ij", "ji")
    observed, queried = {}, []

    def query(a, b):
        key = (a, b)
        if key in observed:
            return
        if len(queried) >= 4:
            raise ValueError("four-candidate decision budget exceeded")
        record = grid[key]
        observed[key] = latency(record)
        queried.append(record)

    if policy in ("static", "B1-static"):
        a = preferred_high(program, shape)
        for b in ("none", "tile16", "unroll4"):
            query(a, b)
        query(next(x for x in highs if x != a), "none")
    elif policy == "greedy":
        for a in highs:
            query(a, "none")
        a = min(highs, key=lambda x: observed[x,"none"])
        for b in ("tile16", "unroll4"):
            query(a, b)
    elif policy == "lookahead":
        for b in ("none", "tile16"):
            for a in highs:
                query(a, b)
    else:
        raise ValueError(policy)
    best = min(queried, key=latency)
    if latency(best)==float("inf"):
        raise ValueError("all queried candidates failed; no valid performance ranking")
    overhead = time.perf_counter()-started
    totals = {}
    for record in queried:
        for key, value in costs(record).items():
            totals[key] = totals.get(key, 0)+value
    return {"policy": policy, "query_ids": [r["candidate_id"] for r in queried],
            "selected": best["candidate_id"], "latency_ms": latency(best), "costs": totals,
            "unique_implementations":len({r.get("code_duplicate_of") or r["candidate_id"] for r in queried}),
            "bookkeeping_seconds": overhead}


def analyze_case(records):
    records = [r for r in records if r.get("role")=="factorial"]
    results = [simulate(records, policy) for policy in ("static", "greedy", "lookahead", "B1-static")]
    oracle = min(records, key=latency)
    if latency(oracle)==float("inf"):
        raise ValueError("candidate pool has no valid oracle")
    for result in results:
        result["regret_ratio"] = result["latency_ms"]/latency(oracle)
    highs = sorted({r["high_choice"] for r in records})
    branches = {a: min((r for r in records if r["high_choice"]==a), key=latency) for a in highs}
    greedy = next(r for r in results if r["policy"]=="greedy")
    selected = next(r for r in records if r["candidate_id"]==greedy["selected"])
    mad = lambda r: statistics.median(abs(x-latency(r)) for x in r["timings_ms"])
    noise = max(.01*latency(oracle), 3*(mad(selected)+mad(oracle)))
    return {"case_id": oracle["case_id"], "shape": oracle["input"]["shape"],
            "oracle": {"candidate": oracle["candidate_id"], "latency_ms": latency(oracle),
                       "candidate_evaluations": len(records)},
            "branch_best": {a:{"candidate":r["candidate_id"],"latency_ms":latency(r)} for a,r in branches.items()},
            "greedy_loss_above_noise": greedy["latency_ms"]-latency(oracle)>noise,
            "policies": results}
