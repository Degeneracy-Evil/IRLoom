"""Controlled multi-op context study and small, grouped information comparisons."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import statistics as st
import time

from benchmarks.context_graph import CONTEXTS,target_region
from .compiler import ROOT,command
from .decision import costs,latency
from .m4 import read_records,clipped_diff

LOWS=("none","tile8","tile32")
HIGHS=("separate","fused")


def snapshot(r,stage):
    return Path(next(s["path"] for s in r["snapshots"] if s["stage"]==stage)).read_text()


def dispersion(r):
    v=r["timings_ms"]
    median=st.median(v)
    return {"median_ms":median,"mad_ms":st.median(abs(x-median) for x in v),"min_ms":min(v),"max_ms":max(v)}


def effect(a,b):
    da,db=dispersion(a),dispersion(b)
    threshold=max(.03,3*max(da["mad_ms"]/da["median_ms"],db["mad_ms"]/db["median_ms"]))
    speedup=latency(a)/latency(b)
    rng=random.Random(1)
    ratios=[]
    for _ in range(1000):
        indices=rng.choices(range(len(a["timings_ms"])),k=len(a["timings_ms"]))
        ratios.append(st.median(a["timings_ms"][i] for i in indices)/st.median(b["timings_ms"][i] for i in indices))
    ratios.sort()
    interval=[ratios[24],ratios[974]]
    direction=1 if speedup>1+threshold and interval[0]>1 else -1 if speedup<1/(1+threshold) and interval[1]<1 else 0
    return {"speedup":speedup,"relative_threshold":threshold,"sample_interval_95":interval,"resolved_direction":direction}


def signature(r,level):
    f=r["context_features"]
    result=list(f["local_current"])
    if level in ("L1","L2"):
        result+=f["full_current"]
    if level=="L2":
        result+=f["upstream"]
    return tuple(result)


def information(runs):
    # Repetitions of a case stay together, including its complete candidate pool.
    merged={}
    for run in runs:
        for r in run["records"]:
            key=(r["case_id"],r["candidate_id"])
            if key not in merged:
                merged[key]={**r,"timings_ms":[]}
            merged[key]["timings_ms"]+=r["timings_ms"]
    all_rows=list(merged.values())
    predictions=[]
    training_start=time.perf_counter()
    for case in runs[0]["info"]["config"]["cases"]:
        for high in HIGHS:
            target=next(r for r in all_rows if r["case_id"]==case["id"] and r["high_choice"]==high)
            for fold in ("leave_shape_out","leave_context_out"):
                train=[r for r in all_rows if r["high_choice"]==high and
                       (r["input"]["shape"]!=case["shape"] if fold=="leave_shape_out" else r["context"]!=case["context"])]
                assert train and all(r["case_id"]!=case["id"] for r in train)
                for policy in ("L0","L1","L2","fixed_none","fixed_tile8","fixed_tile32","current_rule"):
                    start=time.perf_counter()
                    chosen_train=[]
                    if policy.startswith("fixed_"):
                        low=policy.removeprefix("fixed_")
                    elif policy=="current_rule":
                        low="tile32" if target["context_features"]["full_current"][-1] else "none"
                    else:
                        group=train
                        # Unknown richer signatures fall back to the simpler window, never a test label.
                        for level in ("L0","L1","L2")[:int(policy[1])+1]:
                            matching=[r for r in group if signature(r,level)==signature(target,level)]
                            if matching:
                                group=matching
                        def distance(r):
                            return sum(math.log(x/y)**2 for x,y in zip(r["input"]["shape"],case["shape"]))
                        nearest=min(map(distance,group))
                        group=[r for r in group if abs(distance(r)-nearest)<1e-12]
                        chosen_train=sorted({r["case_id"] for r in group})
                        low=min(LOWS,key=lambda b:st.geometric_mean(latency(r) for r in group if r["low_choice"]==b))
                    predictions.append({"case_id":case["id"],"context":case["context"],"shape":case["shape"],
                                        "high_choice":high,"fold":fold,"policy":policy,"low_choice":low,
                                        "train_case_ids":sorted({r["case_id"] for r in train}),
                                        "matched_train_cases":chosen_train,"ranking_seconds":time.perf_counter()-start})
    ranking_seconds=time.perf_counter()-training_start
    decisions=[]
    for batch,run in enumerate(runs,1):
        for p in predictions:
            pool=[r for r in run["records"] if r["case_id"]==p["case_id"] and r["high_choice"]==p["high_choice"]]
            chosen=next(r for r in pool if r["low_choice"]==p["low_choice"])
            policy=p["policy"]
            feature_level=policy if policy.startswith("L") else "L1" if policy=="current_rule" else None
            decisions.append({**p,"batch":batch,"selected":chosen["candidate_id"],"queries":1,
                              "latency_ms":latency(chosen),"regret":latency(chosen)/min(map(latency,pool)),
                              "costs":costs(chosen),"feature_seconds":chosen["context_features"]["seconds"][feature_level] if feature_level else 0})
    return {"scope":"grouped lookup pilot, not an estimate of general program generalization",
            "training_and_ranking_seconds":ranking_seconds,"decisions":decisions}


def write_report(paths,destination=None):
    destination=destination or ROOT/"experiments/reports"
    destination.mkdir(parents=True,exist_ok=True)
    runs=[{"path":str(p.resolve().relative_to(ROOT)),"records":read_records(p),
           "info":json.loads((p.parent/"run.json").read_text())} for p in paths]
    actions=[f"{a}-{b}" for a in HIGHS for b in LOWS]
    effects,shared,results=[],[],[]
    for batch,run in enumerate(runs,1):
        records=run["records"]
        for r in records:
            if r["status"]!="ok":
                raise ValueError(f"failed graph candidate retained in {run['path']}; fix legality before scoring")
            assert len(r["timings_ms"])>=10 and r["correctness"]["status"]=="pass"
            c=costs(r)
            assert c["candidate_compiles"]==1 and c["calibration_invocations"]==0
            assert c["correctness_invocations"]==4 and c["timing_invocations"]==run["info"]["config"]["samples"]
            assert c["native_kernel_calls"]==4+run["info"]["config"]["samples"]*(r["measurement"]["iterations"]+4)
        for case in run["info"]["config"]["cases"]:
            pool=[r for r in records if r["case_id"]==case["id"]]
            baseline=next(r for r in pool if r["candidate_id"]=="separate-none")
            for r in pool:
                results.append({"batch":batch,"case_id":case["id"],"context":case["context"],"shape":case["shape"],
                                "candidate_id":r["candidate_id"],**dispersion(r),
                                "speedup_vs_separate_none":latency(baseline)/latency(r),"costs":costs(r)})
            for low in LOWS:
                a,b=[next(r for r in pool if r["high_choice"]==h and r["low_choice"]==low) for h in HIGHS]
                effects.append({"batch":batch,"case_id":case["id"],"context":case["context"],"shape":case["shape"],
                                "kind":"fusion","action":low,**effect(a,b)})
            for high in HIGHS:
                a=next(r for r in pool if r["high_choice"]==high and r["low_choice"]=="none")
                for low in LOWS[1:]:
                    b=next(r for r in pool if r["high_choice"]==high and r["low_choice"]==low)
                    effects.append({"batch":batch,"case_id":case["id"],"context":case["context"],"shape":case["shape"],
                                    "kind":"tiling","action":f"{high}-{low}",**effect(a,b)})
        for shape in {tuple(c["shape"]) for c in run["info"]["config"]["cases"]}:
            pools=[[r for r in records if r["context"]==ctx and tuple(r["input"]["shape"])==shape] for ctx in CONTEXTS]
            for pool in pools:
                assert len(pool)==6
            # Exact matched original producer and normalized local affine region under separate.
            for pool in pools:
                assert target_region(shape) in snapshot(pool[0],"high")
            windows=[next(r for r in p if r["candidate_id"]=="separate-none")["context_features"]["local_window"] for p in pools]
            assert windows[0]==windows[1]==windows[2]
            def regret(action):
                return [latency(next(r for r in p if r["candidate_id"]==action))/min(map(latency,p)) for p in pools]
            best=min(actions,key=lambda a:st.geometric_mean(regret(a)))
            shared.append({"batch":batch,"shape":shape,"posthoc_only":True,"best_shared_action":best,
                           "gm_regret":st.geometric_mean(regret(best)),"context_regrets":dict(zip(CONTEXTS,regret(best))),
                           "context_winners":{ctx:min(p,key=latency)["candidate_id"] for ctx,p in zip(CONTEXTS,pools)}})
    info=information(runs)
    summaries=[]
    for batch in range(1,len(runs)+1):
        for fold in ("leave_shape_out","leave_context_out"):
            for high in HIGHS:
                for policy in ("L0","L1","L2","fixed_none","fixed_tile8","fixed_tile32","current_rule"):
                    group=[x for x in info["decisions"] if (x["batch"],x["fold"],x["high_choice"],x["policy"])==(batch,fold,high,policy)]
                    summaries.append({"batch":batch,"fold":fold,"high_choice":high,"policy":policy,
                                      "gm_regret":st.geometric_mean(x["regret"] for x in group),
                                      "mean_charged_seconds":st.mean(x["costs"]["charged_wall_seconds"] for x in group),
                                      "mean_feature_seconds":st.mean(x["feature_seconds"] for x in group)})
    # Context-dependent direction changes must pass both conservative noise and sample interval checks.
    reversals=[]
    for x in effects:
        if not x["resolved_direction"]:
            continue
        for y in effects:
            if (x["batch"],x["shape"],x["kind"],x["action"])==(y["batch"],y["shape"],y["kind"],y["action"]) and x["context"]<y["context"] and x["resolved_direction"]==-y["resolved_direction"]:
                reversals.append({"batch":x["batch"],"shape":x["shape"],"kind":x["kind"],"action":x["action"],
                                  "contexts":[x["context"],y["context"]],"speedups":[x["speedup"],y["speedup"]]})
    raw=[]
    keys=("case_id","candidate_id","context","input","high_choice","low_choice","status","correctness",
          "measurement","timings_ms","snapshots","code_duplicate_of","context_features","disassembly")
    for run in runs:
        raw.append({"path":run["path"],"info":run["info"],"metadata":run["records"][0]["metadata"],
                    "records":[{**{k:r[k] for k in keys},"costs":costs(r)} for r in run["records"]]})
    bundle={"schema_version":1,"offline":True,"candidate_results":results,"effects":effects,"shared_action_bounds":shared,
            "resolved_context_reversals":reversals,"information":info,"information_summaries":summaries,"runs":raw}
    (destination/"m7-results.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+"\n")
    lines=["# M7：局部相同、全局上下文不同的多算子图", "",
           "冻结协议见 docs/M7_PROTOCOL.md；本轮无神经网络、无未来LLVM/反汇编特征。"
           "不同上下文工作量不同，以下只比较各自候选相对效应。", "", "## 采集与正确性", ""]
    for i,run in enumerate(runs,1):
        r=run["records"][0]
        lines.append(f"- 轮{i}：`{run['path']}`，commit `{r['metadata']['git_commit']}`，工作区"
                     f"{'干净' if not r['metadata']['git_status'] else '有改动'}；{len(run['records'])}/{len(run['records'])} 正确，"
                     f"{sum(len(r['timings_ms']) for r in run['records'])} 个计时；collect {run['info']['wall_seconds']:.2f} s。")
    lines += ["", "目标 D=A*B 的原始producer文本在同Shape三个上下文逐字一致；separate 路径的第一个affine循环nest在SSA重命名后也一致。"
              "single输出D+A，transpose另加D转置，row_reduce另加D每行和广播。高层separate/fused，低层none/tile8/tile32。"
              "原生fusion保留多消费者D，不强制跨越共享值；generic从2/3/4变为1/2/3，alloc分别1→1、2→2、3→2。", "",
              "## 各路径相对效应", "", "speedup 以本context的 separate-none 为基准；表中为 median ms/kernel 与 speedup，原始样本/MAD/范围在JSON。", "",
              "| Case | 轮 | separate-none | separate-tile8 | separate-tile32 | fused-none | fused-tile8 | fused-tile32 |",
              "| --- | ---: | --- | --- | --- | --- | --- | --- |"]
    for batch,run in enumerate(runs,1):
        for case in run["info"]["config"]["cases"]:
            group={r["candidate_id"]:r for r in results if r["batch"]==batch and r["case_id"]==case["id"]}
            lines.append(f"| {case['id']} | {batch} | "+" | ".join(f"{group[a]['median_ms']:.6f} / {group[a]['speedup_vs_separate_none']:.3f}×" for a in actions)+" |")
    lines += ["", "## 上下文的最优路径与共享选择下界", "",
              "以下共享动作使用全池标签事后选择，仅描述严格相同局部信息必须选同一动作时的损失下界，不是部署策略或训练成绩。"
              "赢家不同但差距小于噪声阈值时，不据此声称稳定反转。", "",
              "| Shape | 轮 | single最佳 | transpose最佳 | row_reduce最佳 | 事后最佳共享动作 | 共享 GM regret |",
              "| --- | ---: | --- | --- | --- | --- | ---: |"]
    for x in shared:
        lines.append(f"| {list(x['shape'])} | {x['batch']} | "+" | ".join(x["context_winners"][ctx] for ctx in CONTEXTS)+f" | {x['best_shared_action']} | {x['gm_regret']:.4f}× |")
    lines += ["", "### 方向反转检查", "",
              "同一高层下tile相对none，或同一低层下fusion相对separate：要求一个上下文加速、另一个减速，"
              "均超过 max(3%, 三倍两候选最大相对MAD)，且1000次按计时轮重采样的探索性95%区间不跨1。"
              "区间只反映本批样本波动，不保证跨时段/硬件独立性。", ""]
    if reversals:
        for x in reversals:
            lines.append(f"- 轮{x['batch']} / {x['shape']} / {x['kind']} {x['action']}："
                         f"{x['contexts'][0]} {x['speedups'][0]:.3f}×，{x['contexts'][1]} {x['speedups'][1]:.3f}×。")
    else:
        lines.append("本批未检出满足全部阈值的上下文方向反转；不据此宣称优化普遍与上下文无关。")
    lines += ["", "## M7.2：信息窗口与透明查表", "",
              "低层决策在affine边界，高层分支固定。L0=第一个外层循环nest的9项统计；"
              "L1=L0+整个当前affine函数的9项统计；L2=L1+原始高层图的4项语义统计。"
              "归约/转置上下文的额外loop、load、alloc已存在于当前完整IR，不能把这些信息归因于历史。"
              "融合后single的局部body已含consumer，L0此时不再与其它上下文相同。", "",
              "每fold只能用其它完整case的全部三条低层标签，按匹配签名和最近log-Shape分组，"
              "选训练几何均值最低动作；未知丰富签名回退到较简单窗口。"
              "两批训练重复合并在同组，测试case全部候选/重复从训练中排除。"
              "leave_shape_out只测同图的新规模，不是新结构泛化；leave_context_out才留出整种图，且未知结构可能退化为L0。"
              "六个case仅够pilot，不拟合神经网络。fixed三动作及廉价当前IR规则（存在转置load选tile32，否则none）同为一次测试查询。", "",
              "| 高层 | 轮 | 留出 | L0 | L1 | L2 | fixed none | fixed tile8 | fixed tile32 | current rule |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for batch in range(1,len(runs)+1):
        for fold in ("leave_shape_out","leave_context_out"):
            for high in HIGHS:
                found={x["policy"]:x["gm_regret"] for x in summaries if (x["batch"],x["fold"],x["high_choice"])==(batch,fold,high)}
                lines.append(f"| {high} | {batch} | {fold} | "+" | ".join(f"{found[p]:.4f}×" for p in ("L0","L1","L2","fixed_none","fixed_tile8","fixed_tile32","current_rule"))+" |")
    lines += ["", "每个测试决策只有一次候选编译、四次正确性、15次timing、零校准，原生调用4+15×(iterations+4)。"
              "成本包含该固定高层的必要prefix；窗口读取/统计时间另计，额外编译为零。"
              "训练用的完整池标签采集不免费，全部进程费用和训练case/选路/窗口成本在JSON，不能用测试oracle选动作。"
              "计费为离线求和，不代表实际在线等墙钟优化。", "", "## 噪声与解释边界", ""]
    for i,run in enumerate(runs,1):
        relative=[dispersion(r)["mad_ms"]/latency(r) for r in run["records"]]
        shortest=min(min(r["timings_ms"])*r["measurement"]["iterations"] for r in run["records"])
        lines.append(f"- 轮{i}：MAD/median中位{100*st.median(relative):.3f}%，最大{100*max(relative):.3f}%；最短原生计时批{shortest:.3f} ms。")
    lines += ["", "固定迭代、不使用全池校准；短批次保留不事后修复。单CPU、单线程、热态复用，"
              "临时申请/释放/输出copy包含在kernel内；参考和进程启动不在原生timer内，但计入进程墙钟。"
              "频率/共享负载不可控、无PMU；向量指令和alloc/copy只是可审阅证据，不断言某种缓存或端口是唯一原因。"
              "本轮不声称Attention、持续历史状态或硬件泛化有效；人工结论见m7-interpretation.md。", "",
              "## 复现", "", "```bash", "bash scripts/phase0.sh m7", "```", "",
              "单入口两批独立编译→四种子校验→15轮计时→上下文对照与分组信息pilot。"
              "只生成M7报告，不重跑/重写M0–M6数据；完整产物与日志位于tmp/和logs/，不计算hash、不新增单元测试。", ""]
    report=destination/"m7-findings.md"
    report.write_text("\n".join(lines))
    lineage=["# M7 完整高层图与决策边界差异", "", "仅比较同一上下文内合法候选；局部原始D在三图一致。", ""]
    for ctx in CONTEXTS:
        pool=[r for r in runs[0]["records"] if r["context"]==ctx and r["input"]["shape"]==runs[0]["info"]["config"]["cases"][0]["shape"]]
        a,b=[next(r for r in pool if r["candidate_id"]==f"{h}-none") for h in HIGHS]
        lineage += [f"## {ctx}: 完整原始图", "", "```mlir",snapshot(a,"high").strip(),"```", ""]
        for stage in ("high_decision","bufferized","affine"):
            lineage += [f"### {stage}: separate → fused", "", "```diff",clipped_diff(snapshot(a,stage),snapshot(b,stage),limit=80),"```", ""]
        lineage += ["### 当前窗口与高层历史统计", "", "```json",json.dumps({h:r["context_features"] for h,r in zip(HIGHS,(a,b))},indent=2),"```", ""]
    (destination/"m7-lineage.md").write_text("\n".join(lineage))
    return report


def run(config,destination=None):
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs=ROOT/"logs"/f"m7-{stamp}"
    logs.mkdir(parents=True)
    paths,events=[],[]
    start=time.perf_counter()
    for i,seed in enumerate((1,17),1):
        print(f"M7 batch {i}/2",flush=True)
        output=command(["bash",ROOT/"scripts/phase0.sh","collect","--config",config,"--order-seed",seed],
                       logs/f"batch-{i}.log",events,"collect",3600)
        paths.append(Path(next(line.removeprefix("records: ") for line in output.splitlines() if line.startswith("records: "))))
        print(output.strip(),flush=True)
    report=write_report(paths,destination)
    (logs/"workflow.json").write_text(json.dumps({"records":[str(p) for p in paths],"events":events,"wall_seconds":time.perf_counter()-start},indent=2)+"\n")
    print(f"report: {report}\nworkflow: {logs/'workflow.json'}",flush=True)
