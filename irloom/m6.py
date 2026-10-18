"""Frozen M6 expanded pool, blind evaluation, and a small ridge pilot."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import time

from .compiler import ROOT,command
from .decision import costs,latency
from .m4 import read_records,clipped_diff
from .search import replay
from .value import features,predict,train

POLICIES=("fixed_m5","fixed_spread","static","greedy","adaptive","ridge","ridge_feedback")


def dispersion(record):
    values=record["timings_ms"]
    median=statistics.median(values)
    return {"median_ms":median,"mad_ms":statistics.median(abs(v-median) for v in values),
            "min_ms":min(values),"max_ms":max(values)}


def merged(runs):
    groups={}
    for run in runs:
        for r in run["records"]:
            groups.setdefault((r["case_id"],r["candidate_id"]),[]).append(r)
    output=[]
    for rs in groups.values():
        r=dict(rs[0])
        r["timings_ms"]=[v for row in rs for v in row["timings_ms"]]
        output.append(r)
    return output


def write_report(paths,destination=None,training_logs=None):
    destination=destination or ROOT/"experiments/reports"
    destination.mkdir(parents=True,exist_ok=True)
    runs=[]
    for path in paths:
        path=path.resolve()
        runs.append({"path":str(path.relative_to(ROOT)),"records":read_records(path),
                     "info":json.loads((path.parent/"run.json").read_text())})
    # Fit/tune before any blind baseline evaluation or label-derived statistics.
    model,training=train(merged(runs))
    model_path=destination/"m6-model.json"
    model_path.write_text(json.dumps({"model":model,"training":training},indent=2)+"\n")
    if training_logs:
        training_logs.mkdir(parents=True,exist_ok=True)
        (training_logs/"training.json").write_text(json.dumps(training,indent=2)+"\n")
    rows=[]
    detail=[]
    predictions_audit=[]
    for batch,run in enumerate(runs,1):
        for case_index,case in enumerate(run["info"]["config"]["cases"]):
            pool=[r for r in run["records"] if r["case_id"]==case["id"]]
            start=time.perf_counter()
            predictions=dict(zip([r["candidate_id"] for r in pool],predict(model,pool)))
            inference=time.perf_counter()-start
            # Oracle and RMSE are scoring labels only, after model parameters are saved.
            oracle=min(map(latency,pool))
            rmse=math.sqrt(statistics.mean((predictions[r["candidate_id"]]-math.log(latency(r)))**2 for r in pool))
            common={"batch":batch,"case_id":case["id"],"program":case["program"],"split":case["split"]}
            predictions_audit.append({**common,"log_rmse":rmse,"inference_seconds":inference})
            for k in run["info"]["config"]["budgets"]:
                for policy in POLICIES+("random","stratified_random"):
                    seeds=run["info"]["config"]["random_seeds"] if "random" in policy else [0]
                    results=[]
                    for seed in seeds:
                        r=replay(pool,policy,k,seed+1000*case_index,predictions)
                        if r["latency_ms"] is None:
                            raise ValueError(f"no valid queried candidate: {case['id']}")
                        assert len({q["candidate_id"] for q in r["queries"]})==k
                        assert r["selected"] in [q["candidate_id"] for q in r["queries"]]
                        c=r["costs"]
                        assert c["candidate_compiles"]==k and c["calibration_invocations"]==0
                        assert c["timing_invocations"]==k*run["info"]["config"]["samples"]
                        r.update(common,regret=r["latency_ms"]/oracle,oracle_ms=oracle,
                                 prediction_log_rmse=rmse,prediction_seconds=inference if policy.startswith("ridge") else 0)
                        results.append(r)
                        # Other random traces can be reconstructed from the frozen seeds.
                        if seed==0:
                            detail.append(r)
                    rows.append({**common,"policy":policy,"k":k,"seed_count":len(results),
                                 "regrets":[x["regret"] for x in results],
                                 "charged_seconds":[x["costs"]["charged_wall_seconds"] for x in results],
                                 "latencies_ms":[x["latency_ms"] for x in results]})
    if training_logs:
        (training_logs/"decisions.json").write_text(json.dumps(detail,indent=2)+"\n")
    summaries=[]
    for batch in range(1,len(runs)+1):
        for program in ("matmul","reduction","pointwise_chain"):
            for split in ("train","validation","blind"):
                for k in (2,4,6):
                    for policy in POLICIES+("random","stratified_random"):
                        group=[r for r in rows if (r["batch"],r["program"],r["split"],r["k"],r["policy"])==(batch,program,split,k,policy)]
                        if not group:
                            continue
                        # One GM per seed across cases, then arithmetic mean and range across seeds.
                        seed_scores=[statistics.geometric_mean(r["regrets"][j] for r in group) for j in range(group[0]["seed_count"])]
                        summaries.append({"batch":batch,"program":program,"split":split,"k":k,"policy":policy,
                                          "cases":len(group),"mean_gm_regret":statistics.mean(seed_scores),
                                          "min_seed_gm":min(seed_scores),"max_seed_gm":max(seed_scores),
                                          "seed_gm_regrets":seed_scores,
                                          "mean_charged_seconds_per_case":statistics.mean(statistics.mean(r["charged_seconds"]) for r in group)})
    raw=[]
    for run in runs:
        keys=("case_id","candidate_id","program_id","input","split","high_choice","low_choice","role",
              "status","correctness","measurement","timings_ms","code_duplicate_of","disassembly","snapshots")
        raw.append({"path":run["path"],"metadata":run["records"][0]["metadata"],"info":run["info"],
                    "records":[{**{k:r[k] for k in keys},"costs":costs(r),"dispersion":dispersion(r),
                                "predecision_feature_vector":features(r)} for r in run["records"]]})
    bundle={"schema_version":1,"offline":True,"model_frozen_before_blind_scoring":True,
            "summaries":summaries,"case_seed_scores":rows,"query_examples":detail,
            "prediction_metrics":predictions_audit,"runs":raw}
    (destination/"m6-results.json").write_text(json.dumps(bundle,ensure_ascii=False,separators=(",",":"))+"\n")
    lines=["# M6 首轮：扩展覆盖、反馈式探针与可学习性能 pilot", "",
           "本轮为离线回放。固定覆盖、真正依据已测反馈分配后续查询、透明 ridge 模型分开比较。"
           "不把查询次数预算声称为墙钟预算，也不声称已部署在线优化器。", "",
           "## 数据与预先划分", ""]
    for i,run in enumerate(runs,1):
        r=run["records"][0]
        lines.append(f"- 批次{i}：`{run['path']}`；commit `{r['metadata']['git_commit']}`；"
                     f"工作区{'干净' if not r['metadata']['git_status'] else '有改动'}；"
                     f"{sum(r['status']=='ok' for r in run['records'])}/{len(run['records'])} 正确；"
                     f"{sum(len(r['timings_ms']) for r in run['records'])} 样本；全池墙钟 {run['info']['wall_seconds']:.2f} s。")
    lines += ["", "候选为 A0/A1 × none/tile4/tile8/tile16/tile32/unroll4，12 条；参数和 Shape 在正式采集前冻结。"
              "训练 8 个实例（matmul/归约各4），验证4个实例（各2），盲测8个实例（各2个未见 Shape，加4个整模板留出的逐元素链）。"
              "两次计时属于同一个 case 的同一 split，训练标签为合并重复的中位延迟，不能当作独立实例。", "",
              "逐元素链 C=A*B+A 使用两个 linalg.generic；A0 保持两阶段、A1 合法融合为一阶段，"
              "与归约/三重 matmul 有不同计算结构。one-shot bufferization 后两路均有一个临时 alloc/dealloc，"
              "输出 copy 和临时申请释放包含在 kernel 延迟中。单核、固定批次次数、10次计时、四个正确性种子、热态复用保持一致。", "",
              "## 模型与可见信息", "",
              f"Ridge 参数 {model['parameter_count']} 个（含截距），NumPy {training['numpy_version']}；alpha 从 [0.1,1,10] "
              f"仅按验证集 k=4 的 regret 选为 {model['alpha']}；归一化仅在训练集拟合。"
              f"本次拟合/验证选择共 {training['training_seconds']:.4f} s，训练日志见 m6-model.json 和本机 training.json。", "",
              "特征仅来自输入结构、Shape、计划高/低层动作和静态目标 cache：操作量、字节量、归约范围、"
              "预期最内层 stride/extent、tile 参数与边界余数等；无需额外 Lowering，未读取未来 LLVM IR、代码或未查询路径延迟。"
              "特征生成只作纯数值计算，额外编译次数为零。模型/特征 schema 和精确 split 已在采集前提交；"
              "模型文件在首次盲测评分之前保存。盲测为预先留出标签，不是外部保密评审。", "",
              "## 策略与真实成本", "",
              "fixed_m5 保留 none/tile16/unroll4 的跨分支顺序；fixed_spread 预先用 tile8/tile32/none。"
              "static 使用冻结 Shape/连续访问规则；greedy 先两个 none 然后固定分支。"
              "adaptive 先两个 none，用已测最优和分支探针数的乐观界选下一分支，且依据实际 tile16 收益改变后续 tile/unroll 次序。"
              "其查询在反馈后变化，区别于固定均衡覆盖。", "",
              "ridge 只按冻结预测分配候选；ridge_feedback 先两个 none，再以同分支已测 log-residual 修正未测预测。"
              "random/stratified_random 各32种子，抽取唯一候选；oracle 只事后计分。"
              "k=2/4/6 时，各方法均 k 次候选编译、4k 次 correctness、10k 次 timing、零校准，"
              "同一 case 的原生 kernel 调用次数严格一致。JSON 保留编译/校验/采样真实费用离线求和、推理与决策开销。", "",
              "## k=4：盲测家族结果", "",
              "regret 为找到的延迟 / 全池最优；随机先对各 seed 的实例 GM 计分，再取均值和范围。"
              "同一候选池所有方法可见合法动作相同；模型权重不在盲测上更新。", "",
              "| 家族 | 轮 | 策略 | 平均 GM regret | 种子范围 | 每实例计费 s |", "| --- | ---: | --- | ---: | --- | ---: |"]
    for x in summaries:
        if x["split"]=="blind" and x["k"]==4:
            lines.append(f"| {x['program']} | {x['batch']} | {x['policy']} | {x['mean_gm_regret']:.4f}× | "
                         f"[{x['min_seed_gm']:.4f}, {x['max_seed_gm']:.4f}] | {x['mean_charged_seconds_per_case']:.3f} |")
    lines += ["", "## k=2/4/6：各盲测实例与成本", "",
              "逐 case 的所有种子 regret、实际延迟和费用分布在 m6-results.json。下表对各家族盲测实例分别汇总，"
              "不是把不同家族或两批计时当成独立训练样本。", "",
              "| 家族 | 轮 | k | fixed_m5 | fixed_spread | adaptive | ridge | ridge_feedback | random 均值 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for batch in range(1,len(runs)+1):
        for program in ("matmul","reduction","pointwise_chain"):
            for k in (2,4,6):
                found={x["policy"]:x["mean_gm_regret"] for x in summaries if (x["batch"],x["program"],x["split"],x["k"])==(batch,program,"blind",k)}
                if found:
                    lines.append(f"| {program} | {batch} | {k} | "+" | ".join(f"{found[p]:.4f}×" for p in ("fixed_m5","fixed_spread","adaptive","ridge","ridge_feedback","random"))+" |")
    lines += ["", "## 辅助预测分数与测量噪声", "",
              "RMSE 是自然对数延迟误差，不作为成功标准。推理秒数含12候选的静态特征提取；"
              "决策回放开销另见 JSON。MAD 是单候选10个原始样本的中位绝对偏差。", "",
              "| 家族 | 轮 | blind log RMSE 均值 | 每实例推理 ms | MAD/median 中位 / 最大 | 最短原生计时批 ms |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
    for i,run in enumerate(runs,1):
        for program in ("matmul","reduction","pointwise_chain"):
            group=[r for r in run["records"] if r["program_id"]==program]
            metrics=[x for x in predictions_audit if x["batch"]==i and x["program"]==program and x["split"]=="blind"]
            if not metrics:
                continue
            relative=[dispersion(r)["mad_ms"]/latency(r) for r in group]
            shortest=min(min(r["timings_ms"])*r["measurement"]["iterations"] for r in group)
            lines.append(f"| {program} | {i} | {statistics.mean(x['log_rmse'] for x in metrics):.3f} | "
                         f"{1000*statistics.mean(x['inference_seconds'] for x in metrics):.3f} | "
                         f"{100*statistics.median(relative):.2f}% / {100*max(relative):.2f}% | {shortest:.3f} |")
    lines += ["", "固定迭代可能产生不足10ms的短批次；保留样本，不为改善结果事后增加迭代。"
              "小于 max(3%, 三倍两候选最大相对MAD) 的成对差异不解释为稳定增益；"
              "两批重复仍不能消除频率漂移或共享负载。", "", "## 最优集中度和 M6.3 门槛", ""]
    for i,run in enumerate(runs,1):
        for program in ("matmul","reduction","pointwise_chain"):
            group=[r for r in run["records"] if r["program_id"]==program]
            winners=[min((r for r in group if r["case_id"]==c),key=latency) for c in sorted({r["case_id"] for r in group})]
            counts={low:sum(r["low_choice"]==low for r in winners) for low in ("none","tile4","tile8","tile16","tile32","unroll4")}
            lines.append(f"- 轮{i} / {program} 全池最佳低层动作：{json.dumps(counts)}。")
    lines += ["", "本轮只有8个训练输入实例、两个训练模板；整条融合链未参与训练/调参。"
              "低参数拟合的候选标签仍来自少数相关程序，所有结果仅是 pilot。"
              "在预先门槛（至少20个训练实例、3个训练模板，并有稳定同预算决策增益）满足前，不执行 M6.3。"
              "本轮没有 MLP、图模型、Attention 或持续神经状态，也没有参数量匹配的跨层级神经消融；记录为未触发，不宣称架构创新。", "",
              "若 cheap fixed/random 在某家族已逼近 oracle，应保留负结果并优先改进基线。"
              "即使 RMSE 改善，决策 regret 不改善仍不算成功；模型可见训练/验证标签和可见探针反馈，不能用盲测成绩回调参数。", "",
              "## 复现", "", "```bash", "bash scripts/phase0.sh m6", "```", "",
              "单入口两批独立重编译→仅 train/validation 拟合→保存冻结模型→首次 blind 回放和报告。"
              "所有完整命令/IR/二进制在 logs/ 与 tmp/；公开摘要含原始计时、成本、特征、随机分布和查询样例。"
              "模型训练与推理只增加 NumPy 数值求解，无额外 MLIR 探针；依然不计算 hash、不写单元测试。"
              "共享负载/频率不可控、同主机连续两批、随机种子不是独立程序，不能声称硬件泛化或实际在线等墙钟收益。", ""]
    path=destination/"m6-findings.md"
    path.write_text("\n".join(lines))
    # One compact fusion lineage sample, including alloc/dealloc evidence.
    chain=[r for r in runs[0]["records"] if r["program_id"]=="pointwise_chain"]
    if chain:
        case=chain[0]["case_id"]
        pair=[next(r for r in chain if r["case_id"]==case and r["high_choice"]==a and r["low_choice"]=="none") for a in ("separate","fused")]
        excerpt=["# M6 融合链快照差异", "",f"Case {case}: C=A*B+A；输入/输出相同，两个阶段与融合单阶段对比。",""]
        for stage in ("high_decision","bufferized"):
            texts=[Path(next(s["path"] for s in r["snapshots"] if s["stage"]==stage)).read_text() for r in pair]
            excerpt += [f"## {stage}","","```diff",clipped_diff(*texts,limit=90),"```",""]
        (destination/"m6-fusion-lineage.md").write_text("\n".join(excerpt))
    return path


def run(config,destination=None):
    started=time.perf_counter()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs=ROOT/"logs"/f"m6-{stamp}"
    logs.mkdir(parents=True)
    paths,events=[],[]
    for i,seed in enumerate((1,17),1):
        print(f"M6 batch {i}/2: compile/check/measure, blind labels not inspected",flush=True)
        output=command(["bash",ROOT/"scripts/phase0.sh","collect","--config",config,"--order-seed",seed],
                       logs/f"batch-{i}.log",events,"collect",3600)
        paths.append(Path(next(line.removeprefix("records: ") for line in output.splitlines() if line.startswith("records: "))))
        print(output.strip(),flush=True)
    report=write_report(paths,destination,logs)
    (logs/"workflow.json").write_text(json.dumps({"records":[str(p) for p in paths],"events":events,
                                                "wall_seconds":time.perf_counter()-started},indent=2)+"\n")
    print(f"report: {report}\nworkflow: {logs/'workflow.json'}",flush=True)
