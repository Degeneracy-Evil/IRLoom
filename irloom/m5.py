"""M5 analysis and fixed-budget offline decision experiments."""
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time

from .compiler import ROOT, command
from .decision import analyze_case, costs, latency
from .m4 import interaction, read_records, clipped_diff


def analyze_m4():
    source = ROOT/"experiments/reports/m4-interaction.json"
    data = json.loads(source.read_text())
    runs = []
    for run in data["runs"]:
        records = [r for r in run["records"] if r.get("role")=="factorial"]
        runs.append({"source":run["path"], "cases":[analyze_case([r for r in records if r["case_id"]==case])
                                                        for case in sorted({r["case_id"] for r in records})],
                     "candidate_costs":[{"case_id":r["case_id"],"candidate_id":r["candidate_id"],**costs(r)} for r in records]})
    destination = ROOT/"experiments/reports"
    (destination/"m5-a.json").write_text(json.dumps({"source":str(source.relative_to(ROOT)),"offline":True,"runs":runs},indent=2)+"\n")
    lines = ["# M5-A：既有 M4 候选池的离线决策分析", "",
             "只读取已提交的 m4-interaction.json；不重新运行编译或基准。全部结果是离线描述，不是在线优化器收益。", "",
             "贪心策略先查询两个 A 的 B=none，再固定该 A、查询另两个 B，共四个唯一候选。"
             "全池 oracle 查询全部六个，仅为延迟下界，不进入同预算排名。", "",
             "| Case | 轮 | 每个 A 的最优下游 | 贪心最终候选 | 全池最优 | 贪心/全池 | 超过描述性噪声阈值 |",
             "| --- | ---: | --- | --- | --- | ---: | --- |"]
    for i,run in enumerate(runs,1):
        for case in run["cases"]:
            greedy=next(p for p in case["policies"] if p["policy"]=="greedy")
            branches="; ".join(f"{a}: {r['candidate']} ({r['latency_ms']:.6f} ms)" for a,r in case["branch_best"].items())
            lines.append(f"| {case['case_id']} | {i} | {branches} | {greedy['selected']} | {case['oracle']['candidate']} | "
                         f"{greedy['regret_ratio']:.4f}× | {'是' if case['greedy_loss_above_noise'] else '否'} |")
    lines += ["", "阈值为 max(全池最优中位数的 1%, 贪心与最优两个 MAD 和的 3 倍)，仅用于描述性筛选。"
              "M4 的交互 D 与决策损失不同：大尺寸也有强交互，但 greedy 已选到全池最优。", "",
              "## 冻结的简单静态规则", "",
              "仅从决策前已知的程序结构和 Shape：matmul 若 N≤32 且 K≤32 保留 ijk，否则选连续 j 的 ikj；"
              "行归约保留连续归约维最内层的 ij。零测量的直接实现先用 B=none。该规则用 M4 开发，不能把这三个 Shape 当成未见测试。", "",
              "在 M4 上，规则为小矩阵选 ijk、大矩阵选 ikj，能选中高层分支；但非整齐 127×65×93 选择 ikj，"
              "错过 ijk+tile16，不能预见其后续潜力。它也不会识别小矩阵的轻微 tiling 收益。", "",
              "M5 后续固定四次候选评估：static 在静态选中的 A 查询全部三个 B，再查询另一 A 的 none；"
              "greedy 先测两个 none，再测选中 A 的另两个 B；lookahead 测两个 A 各自的 none 和 tile16。"
              "所有选择只用 query 暴露过的延迟；全池最优只在事后计算 regret。规则与 probe 集合在新增数据前冻结，不调参。", "",
              "## 已知成本（每条编译与计时分开）", "",
              "| Case | 候选 | 轮 | 编译 s | 15 批计时覆盖的原生时间 s |", "| --- | --- | ---: | ---: | ---: |"]
    for i,run in enumerate(runs,1):
        for row in run["candidate_costs"]:
            lines.append(f"| {row['case_id']} | {row['candidate_id']} | {i} | {row['compile_seconds']:.4f} | {row['native_timed_seconds']:.4f} |")
    lines += ["", "原生计时覆盖时间 = sum(timings_ms) × iterations / 1000；它不含启动、参考计算、校验、预热或校准。"
              "已提交 M4 摘要没有逐候选校准/运行器墙钟明细，不能补造这些值。完整每轮墙钟见 M4 报告；"
              "新增 M5 数据将保留按阶段的真实命令耗时，并采用预先固定迭代次数，消除离线模拟读取全池校准次数的依赖。", "",
              "下一步先建立非 matmul 的真实 A×B 和未见 Shape 数据；仅在存在规划收益时比较规则策略，不训练模型。", ""]
    path=destination/"m5-a.md"
    path.write_text("\n".join(lines))
    return path


def write_report(paths, destination=None):
    destination = destination or ROOT/"experiments/reports"
    destination.mkdir(parents=True,exist_ok=True)
    runs = []
    for path in paths:
        path=path.resolve()
        records=read_records(path)
        decisions=[analyze_case([r for r in records if r["case_id"]==case]) for case in sorted({r["case_id"] for r in records})]
        interactions=[]
        for program,highs in (("matmul",("ijk","ikj")),("reduction",("ij","ji"))):
            group=[r for r in records if r["program_id"]==program]
            for row in interaction(group,highs):
                row.update(program_id=program,high_levels=highs)
                for old,new in (("ijk_speedup","a0_speedup"),("ikj_speedup","a1_speedup"),
                                ("ijk_effect_interval","a0_effect_interval"),("ikj_effect_interval","a1_effect_interval")):
                    if old in row:
                        row[new]=row.pop(old)
                interactions.append(row)
        for r in records:
            r["costs"]=costs(r)
        for case in decisions:
            # B1 is a deliberate negative control: upstream equals current high IR here.
            b0=next(p for p in case["policies"] if p["policy"]=="static")
            b1=next(p for p in case["policies"] if p["policy"]=="B1-static")
            assert b0["query_ids"]==b1["query_ids"] and b0["selected"]==b1["selected"]
            case["B1_same_queries_and_choice_as_B0"]=True
            for p in case["policies"]:
                assert p["costs"]["candidate_compiles"]==4
                assert p["costs"]["calibration_invocations"]==0
            native={p["costs"]["native_kernel_calls"] for p in case["policies"]}
            batches={p["costs"]["timing_batches"] for p in case["policies"]}
            assert len(native)==len(batches)==1
        runs.append({"source":str(path.relative_to(ROOT)),"metadata":records[0]["metadata"],
                     "run":json.loads((path.parent/"run.json").read_text()),
                     "records":records,"decisions":decisions,"interactions":interactions})
    # Keep the small audit bundle; full per-command logs remain at source JSONL paths.
    keys=("program_id","case_id","candidate_id","input","high_choice","low_choice","role","status",
          "error","failure_stage","correctness","measurement","timings_ms","compile_seconds","costs",
          "decision_features","snapshots","schedule","disassembly","code_duplicate_of","duplicate_of")
    bundle={"schema_version":1,"offline":True,"budget":{"candidate_evaluations":4,"learned_parameters":0},"runs":[]}
    for run in runs:
        bundle["runs"].append({**{k:v for k,v in run.items() if k!="records"},
                               "records":[{k:r[k] for k in keys if k in r} for r in run["records"]]})
    (destination/"m5-results.json").write_text(json.dumps(bundle,indent=2,ensure_ascii=False)+"\n")
    instance_count=len(runs[0]["decisions"])
    timing_budget=4*runs[0]["run"]["config"]["samples"]
    lines=["# M5：有限预算的跨阶段前瞻", "",
           "本报告使用完全测量候选池做**离线**策略模拟。没有在线优化器、模型训练或硬件泛化结论。"
           "M4 的开发集分析见 [m5-a.md](m5-a.md)；本轮原始样本、查询轨迹与成本在 [m5-results.json](m5-results.json)。", "",
           "## 样本与决策点", "",
           "M4 三个 Shape 仅用于开发规则，新增 matmul Shape 不与它们重合；额外行归约家族首次加入 linalg 的 ij/ji 决策。"
           "归约保留每行 j 递增加法顺序，变化的是独立行之间的交错；B 仍是 none/tile16/unroll4。"
           f"本轮共 {instance_count} 个静态输入实例，不等于独立编写的程序模板，也不足以训练或评估家族泛化模型。", ""]
    for i,run in enumerate(runs,1):
        valid=sum(r["status"]=="ok" for r in run["records"])
        samples=sum(len(r.get("timings_ms",[])) for r in run["records"])
        lines.append(f"- 批次{i}：`{run['source']}`；提交 `{run['metadata']['git_commit']}`；工作区"
                     f"{'干净' if not run['metadata']['git_status'] else '有改动'}；{valid}/{len(run['records'])} 有效，"
                     f"{samples} 个原生批次样本；全池收集墙钟 {run['run']['wall_seconds']:.2f} s。")
        duplicates=[(r["case_id"],r["candidate_id"],r["code_duplicate_of"]) for r in run["records"] if r.get("code_duplicate_of")]
        lines.append(f"- 批次{i} 的目标反汇编重复：{json.dumps(duplicates,ensure_ascii=False)}；重复编译仍计入预算，不能当作新的独立实现。")
    lines += ["", "所有候选先以 seeds 0/1/7/19 独立 double 参考校验；单核、三次预热、禁用 fast-math/FMA contraction、复用同一缓冲区。"
              "每 Shape 的 iterations 在新数据产生前固定于 experiments/m5.json，无全池校准。"
              "两批分别用 order_seed=1/17，次数不依赖未查询候选的性能。", "",
              "## M5-B：新场景的全池最优与贪心损失", "",
              "| Case | 轮 | 全池最优 | 最优 ms | greedy 最终 | greedy/oracle | 超过描述性噪声阈值 |",
              "| --- | ---: | --- | ---: | --- | ---: | --- |"]
    for i,run in enumerate(runs,1):
        for case in run["decisions"]:
            greedy=next(p for p in case["policies"] if p["policy"]=="greedy")
            lines.append(f"| {case['case_id']} | {i} | {case['oracle']['candidate']} | {case['oracle']['latency_ms']:.6f} | "
                         f"{greedy['selected']} | {greedy['regret_ratio']:.3f}× | {'是' if case['greedy_loss_above_noise'] else '否'} |")
    lines += ["", "描述阈值同 M5-A：max(oracle 的 1%, 两者 MAD 和的 3 倍)，不是显著性检验。"
              "交互 D 与贪心损失分开：D 只说明低层效果依赖上游，不保证贪心损失或前瞻收益。", "",
              "| Case | B | 轮 | A0 下 speedup | A1 下 speedup | D | 探索性 95% 区间 |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for i,run in enumerate(runs,1):
        for x in run["interactions"]:
            if x["status"]!="ok":
                lines.append(f"| {x['case_id']} | {x['low_choice']} | {i} | incomplete | — | — | — |")
                continue
            lo,hi=x["D_interval_95"]
            lines.append(f"| {x['case_id']} | {x['low_choice']} | {i} | {x['a0_speedup']:.3f}× | {x['a1_speedup']:.3f}× | {x['D']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "D 与 2000 次同轮成对 bootstrap 沿用 M4 定义；A0/A1 分别为 ijk/ikj 或 ij/ji。"
              "区间是探索性的，未作多重比较校正，不能排除频率、共享负载和跨轮依赖。", "",
              "## M5-C：固定四候选的策略对照", "",
              "三条规则在新增标签之前冻结，均无可训练参数，合法池始终是同一个 2×3：", "",
              "- static：依据当前结构/Shape 预选 A，在该 A 测全部三个 B，再测另一 A 的 none。",
              "- greedy：先测两个 A 的 none，用已知延迟选 A，再测其 tile16/unroll4。",
              "- lookahead：每个 A 测 none 与 tile16，估计该 A 已探测下游的最小延迟，再选最佳已测实现。", "",
              "static 的规则是小 matmul（N≤32 且 K≤32）选 ijk，其他选 ikj；行归约选连续归约轴最内层的 ij。"
              "四个 query 之前没有读取其他候选的标签；查询记录写入 JSON。oracle 在各策略结束后单独评分，消耗六候选，不参与同预算排名。", "",
              "| Case | 轮 | static/oracle | greedy/oracle | lookahead/oracle | lookahead 相对 greedy 加速 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for i,run in enumerate(runs,1):
        for case in run["decisions"]:
            p={p["policy"]:p for p in case["policies"]}
            lines.append(f"| {case['case_id']} | {i} | {p['static']['regret_ratio']:.3f}× | {p['greedy']['regret_ratio']:.3f}× | "
                         f"{p['lookahead']['regret_ratio']:.3f}× | {p['greedy']['latency_ms']/p['lookahead']['latency_ms']:.3f}× |")
    lines += ["", "所有策略返回自己已测候选中的最佳值，等于 best-of-k（k=4），而不是猜测未测候选的真实延迟。"
              "best-found 加速比与 oracle regret 区分，不把全池标签输入决策器。下表按程序家族分别统计，计时重复不作为训练样本。", "",
              "| 家族 | 轮 | 策略 | regret 几何均值 | 距 oracle ≤1% 的实例数 |", "| --- | ---: | --- | ---: | ---: |"]
    for i,run in enumerate(runs,1):
        for program in ("matmul","reduction"):
            cases=[c for c in run["decisions"] if c["case_id"].startswith(program+"-")]
            for policy in ("static","greedy","lookahead"):
                ratios=[next(p["regret_ratio"] for p in c["policies"] if p["policy"]==policy) for c in cases]
                if ratios:
                    gm=statistics.geometric_mean(ratios)
                    lines.append(f"| {program} | {i} | {policy} | {gm:.4f}× | {sum(x<=1.01 for x in ratios)}/{len(ratios)} |")
    lines += ["", "## 真实记录成本与预算边界", "",
              "每策略四次独立候选编译，每次包括完整 lowering/codegen/link；四候选各四次 correctness，"
              f"共 {timing_budget} 次计时批次、0 次校准、相同 Shape 下完全相同的原生 kernel 调用次数。"
              "kernel 调用计入每个 measure 进程内额外的一次初始校验调用和三次预热；四个策略均核查一致。", "",
              "这是相同编译次数/实际测量调用数预算，不是相同墙钟预算。成本来自全池中对应真实子进程记录的离线求和，"
              "不是把策略重新在线执行后的实测端到端时间；不包含全局机器探测和文件整理。"
              "每候选完整成本和原始样本在 JSON，各策略按实际查询计费。", "",
              "| Case | 轮 | 策略 | 编译 s | 正确性 s | 采样进程墙钟 s | 原生计时区间 s | 计费合计 s | 决策记账 μs |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for i,run in enumerate(runs,1):
        for case in run["decisions"]:
            for p in case["policies"]:
                if p["policy"]=="B1-static":
                    continue
                c=p["costs"]
                lines.append(f"| {case['case_id']} | {i} | {p['policy']} | {c['compile_seconds']:.3f} | {c['correctness_wall_seconds']:.3f} | "
                             f"{c['timing_wall_seconds']:.3f} | {c['native_timed_seconds']:.3f} | {c['charged_wall_seconds']:.3f} | {p['bookkeeping_seconds']*1e6:.1f} |")
    lines += ["", "记账时间是本次 Python 离线标签查询、规则和中位数处理的开销，不是模型推理或线上搜索时间；重建报告会重新测它。"
              "没有学习模型，推理/训练开销为零。不同路径耗时不同，若需要相同墙钟预算的结论，仍须实现可中断的在线预算实验。", "",
              "## M5-D：信息可见性的小型对照", "",
              "B0 使用高层决策点的当前 IR/Shape、目标和合法动作；B1 的设想是增加上游原始语义。这里 A 决策尚未经过 lowering，"
              "上游和当前输入相同。因此 B1-static 实现为同一个零参数策略的别名，用相同查询/选择/成本核查这一零信息差对照。"
              "没有实现独立历史编码器或非平凡 B1 输入消融，不能把相同结果当作一般性的历史无用结论。", "",
              "B2 以 lookahead 表示有分支探针记录的透明搜索状态：只保留已查询的 none/tile16 延迟，"
              "同样四次评估、零训练参数。若它改善了结果，来源是覆盖了不同下游组合并用其反馈估计未来潜力，"
              "不能归因于历史 IR 编码、持续神经状态或 Attention；查询分配也改变，因此不是隔离历史表示的架构消融。", "",
              "## 判定与下一步", ""]
    for case_id in sorted({c["case_id"] for run in runs for c in run["decisions"]}):
        cases=[c for run in runs for c in run["decisions"] if c["case_id"]==case_id]
        gains=[]
        for c in cases:
            p={p["policy"]:p for p in c["policies"]}
            gains.append(p["greedy"]["latency_ms"]/p["lookahead"]["latency_ms"])
        meaning=len(cases)>=2 and all(c["greedy_loss_above_noise"] and g>1.01 for c,g in zip(cases,gains))
        lines.append(f"- {case_id}：{'重复的贪心损失与前瞻改善，值得继续检验' if meaning else '未据此认定存在稳定的前瞻必要性'}；"
                     f"离线前瞻/贪心收益范围 {min(gains):.3f}–{max(gains):.3f}×。")
    lines += ["", "不存在所有程序都该使用的前瞻策略：若静态或 greedy 已接近 oracle，继续扩大模型没有依据；"
              "lookahead 也可能漏掉未探测的 unroll4。先扩展独立模板、家族分组和线上时间预算验证。"
              "目前仅两个家族，未训练模型、未随机拆分变换或计时样本，不能宣称未见程序或硬件泛化。", "",
              "行归约的高层/affine 变换样例见 [m5-reduction-lineage.md](m5-reduction-lineage.md)。"
              "依旧是快照级 lineage，不支持精确 Operation N:M 追踪；未做 PMU 或向量化因果消融。"
              "不计算 hash，不写单元测试；验收依赖真实编译、独立数值校验、调用预算与查询轨迹核查。", "",
              "```bash", "bash scripts/phase0.sh m5", "```", ""]
    path=destination/"m5-findings.md"
    path.write_text("\n".join(lines))
    reduction=[r for r in runs[0]["records"] if r["program_id"]=="reduction"]
    if reduction:
        case=reduction[0]["case_id"]
        pair=[next(r for r in reduction if r["case_id"]==case and r["high_choice"]==a and r["low_choice"]=="none") for a in ("ij","ji")]
        excerpts=["# M5 行归约的真实高层决策", "", f"Case: {case}，每行加法顺序保留，独立行交错次序改变。", ""]
        for r in pair:
            excerpts += [f"## {r['high_choice']}", "", "```mlir",Path(r["schedule"]).read_text().strip(),"```",""]
        for stage in ("high_decision","affine"):
            texts=[Path(next(s["path"] for s in r["snapshots"] if s["stage"]==stage)).read_text() for r in pair]
            excerpts += [f"## {stage}","","```diff",clipped_diff(*texts),"```",""]
        (destination/"m5-reduction-lineage.md").write_text("\n".join(excerpts))
    return path


def run(config, destination=None):
    started=time.perf_counter()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs=ROOT/"logs"/f"m5-{stamp}"
    logs.mkdir(parents=True)
    paths,events=[],[]
    for i,seed in enumerate((1,17),1):
        print(f"M5 batch {i}/2: independent compile and fixed-iteration measurement",flush=True)
        output=command(["bash",ROOT/"scripts/phase0.sh","collect","--config",config,"--order-seed",seed],
                       logs/f"batch-{i}.log",events,"collect",3600)
        path=Path(next(line.removeprefix("records: ") for line in output.splitlines() if line.startswith("records: ")))
        paths.append(path)
        print(output.strip(),flush=True)
    # Generated analysis files must not dirty the worktree before both build probes.
    analyze_m4()
    report=write_report(paths,destination)
    (logs/"workflow.json").write_text(json.dumps({"records":[str(p) for p in paths],"events":events,
                                                "wall_seconds":time.perf_counter()-started},indent=2)+"\n")
    print(f"report: {report}\nworkflow: {logs/'workflow.json'}",flush=True)
