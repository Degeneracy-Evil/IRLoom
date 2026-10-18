"""Export a small audit bundle and a descriptive Phase 0 report."""
import itertools
import json
from pathlib import Path
import random
import statistics

from .analysis import summarize
from .compiler import ROOT


def paired_interval(baseline, candidate):
    """Exploratory paired bootstrap of round-wise speedup medians."""
    ratios = [a/b for a, b in zip(baseline, candidate)]
    rng = random.Random(1)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return [boot[49], boot[1949]]


def relative(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    if isinstance(value, list):
        return [relative(x) for x in value]
    if isinstance(value, dict):
        return {k: relative(v) for k, v in value.items()}
    return value


def write_report(paths):
    runs = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        info = json.loads((path.parent / "run.json").read_text())
        rows = summarize(records)
        for row in rows:
            if "median_ms" not in row:
                continue
            base = next((r for r in records if r["case_id"] == row["case_id"] and
                         r["candidate_id"] == "default" and r["status"] == "ok"), None)
            candidate = next(r for r in records if r["case_id"] == row["case_id"] and
                             r["candidate_id"] == row["candidate_id"])
            if base and len(base["timings_ms"]) == len(candidate["timings_ms"]):
                row["paired_speedup_interval_95"] = paired_interval(base["timings_ms"], candidate["timings_ms"])
        kept = []
        keys = ("case_id", "candidate_id", "program_id", "input", "seed", "status", "error", "failure_stage",
                "transforms", "snapshots", "binary", "disassembly", "compile_seconds", "correctness",
                "measurement", "timings_ms", "ir_changed", "duplicate_of", "code_duplicate_of", "fingerprints")
        for record in records:
            item = {k: record[k] for k in keys if k in record}
            item["build_events"] = [e for e in record["events"] if e["stage"] not in
                                    ("timing", "calibration", "correctness")]
            kept.append(item)
        runs.append({"records_path": str(path), "run": info, "metadata": records[0]["metadata"],
                     "summary": rows, "records": kept})
    report_dir = ROOT / "experiments/reports"
    report_dir.mkdir(exist_ok=True)
    bundle = {"schema_version": 1, "method": "descriptive paired-candidate analysis; no model",
              "bootstrap": {"resamples": 2000, "seed": 1, "unit": "paired measurement round",
                            "warning": "exploratory; no independence or multiple-comparison guarantee"},
              "runs": runs}
    (report_dir / "phase0-initial.json").write_text(json.dumps(relative(bundle), indent=2, ensure_ascii=False) + "\n")
    compact = [{"records_path": relative(run["records_path"]),
                "git_commit": run["metadata"]["git_commit"],
                "results": [{k: row[k] for k in ("case_id", "candidate_id", "status", "median_ms",
                                                 "mad_ms", "speedup_vs_default") if k in row}
                            for row in run["summary"]]} for run in runs]
    (report_dir / "phase0-summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Phase 0 首批证据报告", "",
             "结论：本机合法 MLIR 优化路径存在可重复的最终延迟差异；跨阶段快照与命令级 lineage 可稳定采集。"
             "本批数据尚不能回答历史 IR 是否改善预测或同预算搜索，不训练模型，不宣称研究架构有效。", "",
             "## 数据与复现", "",
             "```bash", "bash scripts/phase0.sh", "bash scripts/phase0.sh",
             "# 将两次输出的 JSONL 路径传入：",
             "bash scripts/phase0.sh report --records " + " ".join(relative(str(p)) for p in paths), "```", "",
             "详细配置为 `experiments/phase0.json`；简明汇总在 [phase0-summary.json](phase0-summary.json)，"
             "可审计摘要和原始延迟在 [phase0-initial.json](phase0-initial.json)。"
             "完整日志、IR、二进制保存在本机 logs/ 与 tmp/，不提交 Git。未计算产物 hash，无法保证密码学完整性。", ""]
    for i, run in enumerate(runs, 1):
        metadata = run["metadata"]
        rows = run["summary"]
        valid = [r for r in rows if "median_ms" in r]
        lines += [f"- 批次 {i}：`{relative(run['records_path'])}`；构建 commit `{metadata['git_commit']}`；"
                  f"工作区{'干净' if not metadata['git_status'] else '有未提交改动'}；有效 {len(valid)}/{len(rows)}；"
                  f"墙钟 {run['run']['wall_seconds']:.2f} s，编译子进程合计 {sum(r.get('compile_seconds', 0) for r in rows):.2f} s。"]
    metadata = runs[0]["metadata"]
    cpu_fields = {r["field"]: r["data"] for r in json.loads(metadata["cpu"])["lscpu"]}
    lines += ["", f"主机：{metadata['os']} / {metadata['arch']} / {cpu_fields.get('Model name:', 'see JSON')}。"
              "工具版本：" + "；".join(metadata["tools"][name]["version"].splitlines()[0]
                                    for name in ("mlir-opt", "clang", "llvm-config")) + "。"]
    first = runs[0]["records"][0]
    lines += ["", f"测量 CPU={first['measurement']['cpu']}，单线程；CLOCK_MONOTONIC，3 次预热，"
              f"每候选 {first['measurement']['samples']} 次批次计时。每个 case 采用相同批量次数，"
              "校准目标至少 10 ms，固定种子随机交错候选。时间排除启动、分配、参考计算和校验，包含 kernel 的必要输出初始化。"
              "相同缓冲区重复访问，属于热缓存实验。", "",
              "所有候选使用相同 native 目标、Clang -O3、禁用 fast-math / fma contraction；"
              "独立 double 参考使用 rtol=atol=1e-4，在候选执行前计算；零输入和三个有符号输入种子均校验。"
              "输入、输出含缓冲区哨兵，计时后复查输出。", "", "## 各路径结果", "",
              "表内为 median ms/kernel、相对同批 default 的加速比。MAD% 为轮内中位绝对偏差除以中位数。"
              "成对 95% 区间是同轮 speedup 中位数的 2000 次 bootstrap，原始区间保存在 JSON；"
              "它只描述本轮采样，不保证系统噪声独立，也未作多重比较校正。", "",
              "| Case | 候选 | 批次 | median ms | MAD% | 相对 default | 成对 95% 区间 | 编译 s |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for i, run in enumerate(runs, 1):
        for row in sorted(run["summary"], key=lambda r: (r["case_id"], r["candidate_id"])):
            if "median_ms" not in row:
                lines.append(f"| {row['case_id']} | {row['candidate_id']} | {i} | failed | — | — | — | — |")
                continue
            ci = row.get("paired_speedup_interval_95")
            interval = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            lines.append(f"| {row['case_id']} | {row['candidate_id']} | {i} | {row['median_ms']:.6f} | "
                         f"{row['mad_ms']/row['median_ms']*100:.2f} | {row.get('speedup_vs_default', 0):.3f}× | "
                         f"{interval} | {row['compile_seconds']:.3f} |")
    lines += ["", "## 重复批次、噪声与近似平局", "",
              "下面按 case 列出各批最快候选，以及本批任意两条路径之间难以区分的情况。"
              "保守规则：中位数差不超过两者中位数均值的 1%，或不超过两者 MAD 和的 3 倍，就标记为近似平局。"
              "这是描述性筛选规则，不是统计显著性检验；小差异即使跨轮同向也需要更多独立批次验证。", ""]
    for case in sorted({r["case_id"] for run in runs for r in run["summary"]}):
        winners, ties = [], set()
        for i, run in enumerate(runs, 1):
            rows = [r for r in run["summary"] if r["case_id"] == case and "median_ms" in r]
            if not rows:
                continue
            best = min(rows, key=lambda r: r["median_ms"])
            winners.append(f"批次{i} {best['candidate_id']} ({best['median_ms']:.6f} ms)")
            for a, b in itertools.combinations(rows, 2):
                bound = max(.01*(a["median_ms"]+b["median_ms"])/2, 3*(a["mad_ms"]+b["mad_ms"]))
                if abs(a["median_ms"]-b["median_ms"]) <= bound:
                    ties.add(" / ".join(sorted([a["candidate_id"], b["candidate_id"]])))
        lines.append(f"- {case}：{'；'.join(winners)}。近似平局：{'、'.join(sorted(ties)) or '按上述规则未发现'}。")
    lines += ["", "## 跨层级采集、受控因素与归因边界", "",
              "每候选保留 high、affine、transformed、scf、LLVM dialect、LLVM IR 六个边界；"
              "matmul 起点为 linalg.matmul，归约为 linalg.generic，stencil 起点已是 affine 循环，缺少命名 stencil 语义。"
              "每个快照关联父路径、具体命令和有序事件，可从 JSONL 重建编译轨迹。", "",
              "linalg-to-affine 将命名/结构化操作展开成循环与访存；tiling 拆分迭代空间，unroll 复制循环体；"
              "canonicalize/cse 可合并或消除操作；LLVM lowering 展开 ABI 和控制流。"
              "这些阶段均不保存 Operation 级 1:N/N:M 来源，clang -O3 内部只记录输入输出边界。"
              "因此当前只能准确解释快照级关系，不能承诺精确指令归因。", "",
              "default 与 tile4 / tile16 / unroll4 的受控变量是单个 affine 变换及参数，其余 lowering、后端 Flags、"
              "输入和测量预算相同。可把重复的大幅变化归于这一编译路径选择及后续优化的整体效果。"
              "不能据此单独声称缓存命中、寄存器压力或自动向量化是原因：没有禁用向量化等消融，也没有 PMU 证据。"
              "归约 tiling 的 IR 显式把行/列分块，但每行加法的逻辑顺序保持；未开启浮点重关联。", "",
              "候选 IR 变化与完整二进制字节差异分别记录，另直接比较含重定位的目标反汇编。"
              "主矩阵重复情况见 JSON；额外 1×1 边界验收中四条路径收敛为相同代码，证明 Pass 列表不同不能直接作为不同实现证据。", "",
              "## 研究问题与下一步", "",
              "路径差异问题已有可重复证据；变换收益依赖 Shape 和工作负载，不能把某个 tile 大小作为通用最优规则。"
              "先看每轮分布及重复批次，再决定是否值得做搜索。", "",
              "本批只有三个程序家族、每类两个 Shape、每 case 四条候选。重复计时不是独立程序样本。"
              "尚未做 B0/B1/B2、训练/测试划分或在线候选选择，因此历史信息价值、同预算搜索增益和硬件泛化均未验证。"
              "没有神经网络、排名模型或离线搜索模拟，不将全池测量得到的最佳候选冒充在线优化成果。", "",
              "后续先扩展独立程序/家族与受控候选，固定 affine 决策点，提取仅截至该点的透明特征；"
              "按程序家族分组拆分，比较当前 IR 与加入上游 IR/历史。未来 LLVM IR、反汇编与真实延迟只用于标签和结果审计，"
              "不得进入决策时特征。预算须同时计入编译、校准、候选测量和搜索开销。", "",
              "限制：单机、单 ISA、热缓存、固定输入分布；共享机器的 CPU 频率/邻核负载、ASLR/分配地址未控制，"
              "没有持续状态或硬件归因实验。源数据中的 loadavg 只反映探测瞬间。未计算 hash、未写单元测试；"
              "真实编译、数值校验、边界验收和失败注入构成当前验收依据。", ""]
    path = report_dir / "phase0-initial.md"
    path.write_text("\n".join(lines))
    return path
