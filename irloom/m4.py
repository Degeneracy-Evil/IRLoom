"""Small two-stage factorial experiment; raw observations, no model."""
from datetime import datetime, timezone
import difflib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

from benchmarks.workloads import generate
from .analysis import summarize
from .compiler import ROOT, command, compile_candidate


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def interaction(records, highs=("ijk", "ikj")):
    result = []
    for case in sorted({r["case_id"] for r in records}):
        grid = {(r.get("high_choice"), r.get("low_choice")): r for r in records
                if r["case_id"] == case and r.get("role") == "factorial" and r["status"] == "ok"}
        for low in ("tile16", "unroll4"):
            cells = [(a, b) for a in highs for b in ("none", low)]
            if any(cell not in grid for cell in cells):
                result.append({"case_id": case, "low_choice": low, "status": "incomplete"})
                continue
            samples = [grid[cell]["timings_ms"] for cell in cells]
            if len({len(s) for s in samples}) != 1 or len(samples[0]) < 10:
                raise ValueError("interaction requires equal sample counts >=10")

            def effects(indices):
                med = [statistics.median(s[i] for i in indices) for s in samples]
                e0, e1 = math.log(med[1]/med[0]), math.log(med[3]/med[2])
                return e0, e1, e1-e0

            e0, e1, d = effects(range(len(samples[0])))
            rng = random.Random(1)
            boot = [effects(rng.choices(range(len(samples[0])), k=len(samples[0]))) for _ in range(2000)]
            intervals = [[sorted(x[j] for x in boot)[i] for i in (49, 1949)] for j in range(3)]
            result.append({"case_id": case, "low_choice": low, "status": "ok", "D": d,
                           "ijk_speedup": math.exp(-e0), "ikj_speedup": math.exp(-e1),
                           "ijk_effect_interval": intervals[0], "ikj_effect_interval": intervals[1],
                           "D_interval_95": intervals[2],
                           "robust_sign_reversal": (intervals[0][1] < 0 < intervals[1][0] or
                                                    intervals[1][1] < 0 < intervals[0][0])})
    return result


def function_assembly(path):
    text = path.read_text()
    match = re.search(r"^[0-9a-f]+ <kernel>:\n(.*?)(?=^[0-9a-f]+ <|\Z)", text, re.M | re.S)
    return match.group(0) if match else text


def clipped_diff(a, b, limit=70):
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile="before", tofile="after", n=2))
    return "\n".join(lines[:limit]) + (f"\n... {len(lines)-limit} more diff lines; see local artifacts" if len(lines)>limit else "")


def write_lineage(records, destination):
    selected = [r for r in records if r["input"]["shape"] == [127, 65, 93]]
    if not selected:
        selected = [r for r in records if r["case_id"] == records[0]["case_id"]]
    by_id = {r["candidate_id"]: r for r in selected}
    lines = ["# M4 代表性快照与函数代码差异", "", "这里只保留紧凑文本摘录。地址与静态指令数量不作为性能指标；完整文件在所列本地产物路径。", ""]
    for before_id, after_id in (("default", "ikj-none"), ("ikj-none", "ikj-tile16")):
        before, after = by_id[before_id], by_id[after_id]
        if before["status"] != "ok" or after["status"] != "ok":
            continue
        lines += [f"## {before['case_id']}: {before_id} → {after_id}", "",
                  f"median {statistics.median(before['timings_ms']):.6f} → {statistics.median(after['timings_ms']):.6f} ms/kernel。", ""]
        for record in (before, after):
            lines += [f"`{record['candidate_id']}` 的高层 schedule：", "", "```mlir",
                      Path(record["schedule"]).read_text().strip(), "```", "", "编译命令：", "", "```text"]
            lines += [" ".join(e["command"]).replace(str(ROOT)+"/", "") for e in record["events"]
                      if e["stage"] in ("high_decision", "affine", "transformed", "codegen")]
            lines += ["```", ""]
        for stage in ("high_decision", "affine", "transformed"):
            a = next(s for s in before["snapshots"] if s["stage"] == stage)
            b = next(s for s in after["snapshots"] if s["stage"] == stage)
            lines += [f"{stage} diff（路径 `{a['path'].replace(str(ROOT)+'/', '')}` 与 `{b['path'].replace(str(ROOT)+'/', '')}`）：",
                      "", "```diff", clipped_diff(Path(a["path"]).read_text(), Path(b["path"]).read_text()) or "(identical)", "```", ""]
        a, b = [function_assembly(Path(r["disassembly"])) for r in (before, after)]
        lines += ["kernel 函数反汇编 diff（截取）：", "", "```diff", clipped_diff(a, b), "```", ""]
        for name, assembly in ((before_id, a), (after_id, b)):
            code = assembly.splitlines()
            first_mul = next((i for i, line in enumerate(code) if re.search(r"\b(v?mul(ss|ps))\b", line)), 0)
            lines += [f"{name} 首个浮点乘法附近：", "", "```asm",
                      "\n".join(code[max(0,first_mul-3):first_mul+13]), "```", ""]
    (destination / "m4-lineage.md").write_text("\n".join(lines))


def write_report(paths, destination=None, equivalence=None):
    paths = [p.resolve() for p in paths]
    destination = destination or ROOT / "experiments/reports"
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in paths:
        records = read_records(path)
        runs.append({"path": str(path.relative_to(ROOT)), "records": records,
                     "info": json.loads((path.parent / "run.json").read_text()),
                     "summary": summarize(records), "interactions": interaction(records)})
    # Export observations and provenance once per run; full command logs stay local.
    bundle = {"schema_version": 1, "bootstrap": {"resamples": 2000, "seed": 1, "paired_rounds": True},
              "default_equivalence": equivalence, "runs": []}
    keys = ("case_id", "candidate_id", "input", "high_choice", "low_choice", "role", "status",
            "failure_stage", "error", "compile_seconds", "measurement", "correctness", "timings_ms",
            "duplicate_of", "code_duplicate_of", "ir_changed")
    for run in runs:
        bundle["runs"].append({"path": run["path"], "metadata": run["records"][0]["metadata"],
                               "run": run["info"], "summary": run["summary"], "interactions": run["interactions"],
                               "records": [{k:r[k] for k in keys if k in r} for r in run["records"]]})
    # Paths in metadata/commands are local references, not portable identifiers.
    (destination / "m4-interaction.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False)+"\n")
    samples = runs[0]["info"]["config"]["samples"]
    lines = ["# M4 跨阶段决策交互实验", "", "本实验比较一个 matmul 程序家族中的两个真实编译决策点，不训练模型。", "",
             "## 受控设计与测量", "", "A：linalg generic 的迭代顺序 ijk / ikj，均由命名 matmul 经 Transform Dialect 泛化及合法 interchange 得到。"
             "B：affine 的 none / tile16 / innermost unroll4。每个 Shape 为 2×3 因子设计；C++ ikj 为单独性能参考，不进入交互计算。", "",
             "A 的两种顺序保持每个 C[i,j] 的 k 递增加法顺序；所有路径禁用 fast-math 与浮点收缩。"
             "C++ 参考为普通 row-major i-k-j 三重循环，Clang++ -O3 -march=native -ffp-contract=off，无 BLAS、restrict、手工向量化或不同布局。"
             "它共享 ABI、输入、输出清零和计时运行器，独立 double 参考负责正确性，不把它称为自动调优或最先进基线。", "",
             f"每轮独立重新编译，固定相同输入种子，分别使用 order_seed=1/17 随机交错；各候选 {samples} 次计时，先用 seeds 0/1/7/19 校验。"
             "校准至少 10 ms，同一 Shape 所有候选与参考使用同一批量次数。3 次预热后重复同一缓冲区；计时含输出清零，排除分配、参考计算与校验。"
             "大工作集不会被人为逐次驱逐，属于稳态复用，不是冷缓存或实测带宽。", ""]
    for i, run in enumerate(runs, 1):
        metadata = run["records"][0]["metadata"]
        valid = sum(r["status"] == "ok" for r in run["records"])
        lines.append(f"- 批次{i}：`{run['path']}`；构建 `{metadata['git_commit']}`；工作区{'干净' if not metadata['git_status'] else '有未提交改动'}；"
                     f"通过 {valid}/{len(run['records'])}；墙钟 {run['info']['wall_seconds']:.2f} s；编译子进程合计 "
                     f"{sum(r.get('compile_seconds',0) for r in run['records']):.2f} s。")
    metadata = runs[0]["records"][0]["metadata"]
    lines += ["", "目标 cache（sysfs，同一逻辑 CPU）：", "", "```json", json.dumps(metadata.get("cpu_caches", []), indent=2), "```", "",
              "| Shape (M,N,K) | A+B+C 工作集 |", "| --- | ---: |"]
    for case in runs[0]["info"]["config"]["cases"]:
        m,n,k = case["shape"]
        lines.append(f"| {m},{n},{k} | {4*(m*k+k*n+m*n)/1024:.2f} KiB |")
    lines += ["", "以上只计热态 kernel 的输入输出，不含运行器的 double 校验缓冲区、哨兵和代码；"
              "超过 cache 容量仅是静态工作集条件，不能替代 PMU miss 或内存带宽实测。", "",
              "## 延迟与参考", "", "| Case | 候选 | 轮 | median ms | MAD% | 相对本轮 default | 编译 s |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for i, run in enumerate(runs, 1):
        for row in sorted(run["summary"], key=lambda x:(x["case_id"],x["candidate_id"])):
            if "median_ms" not in row:
                lines.append(f"| {row['case_id']} | {row['candidate_id']} | {i} | failed | — | — | — |")
                continue
            lines.append(f"| {row['case_id']} | {row['candidate_id']} | {i} | {row['median_ms']:.6f} | "
                         f"{100*row['mad_ms']/row['median_ms']:.2f} | {row.get('speedup_vs_default',0):.3f}× | {row['compile_seconds']:.3f} |")
    lines += ["", "## 高层选择如何改变低层收益", "", "`D = log[T(ikj,B)/T(ikj,none)] - log[T(ijk,B)/T(ijk,none)]`，T 为批次内中位延迟。"
              "D=0 表示在对数延迟尺度上没有交互；D>0 表示 B 在 ikj 下的相对效果更差。"
              "下表 speedup 分别以该 A 的 none 为基线，大于 1 为改善。", "",
              "95% 区间按同一测量轮次联合重采样四个 cell，再重算中位数和 D，2000 次、固定种子。"
              "这是探索性区间，未作多重比较校正，不消除共享负载、频率、ASLR 或跨轮依赖。"
              "符号反转仅在两个 A 的低层 log-effect 区间分别完全位于零两侧时标记。", "",
              "| Case | B | 轮 | ijk 下 speedup | ikj 下 speedup | D | 探索性 95% 区间 | 收益符号反转 |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for i, run in enumerate(runs, 1):
        for x in run["interactions"]:
            if x["status"] != "ok":
                lines.append(f"| {x['case_id']} | {x['low_choice']} | {i} | incomplete | — | — | — | — |")
                continue
            lo, hi = x["D_interval_95"]
            lines.append(f"| {x['case_id']} | {x['low_choice']} | {i} | {x['ijk_speedup']:.3f}× | {x['ikj_speedup']:.3f}× | "
                         f"{x['D']:.4f} | [{lo:.4f}, {hi:.4f}] | {'是' if x['robust_sign_reversal'] else '否'} |")
    lines += ["", "## 跨批次判读", ""]
    for key in sorted({(x["case_id"],x["low_choice"]) for run in runs for x in run["interactions"]}):
        observations = [x for run in runs for x in run["interactions"] if (x["case_id"],x["low_choice"])==key and x["status"]=="ok"]
        supported = (len(observations)==len(runs) and len(runs)>=2 and
                     (all(x["D_interval_95"][0]>0 for x in observations) or all(x["D_interval_95"][1]<0 for x in observations)))
        reversal = len(observations)==len(runs) and all(x["robust_sign_reversal"] for x in observations)
        lines.append(f"- {key[0]} / {key[1]}：{'各批次区间同向排除零，观察到重复的路径交互' if supported else '尚无跨批次一致的交互证据'}；"
                     f"{'各批次均出现收益符号反转' if reversal else '不宣称重复的收益符号反转'}。")
    lines += ["", "## 当前 IR 能解释什么", "", "B 决策点是 affine 快照；上游 ijk/ikj 的区别仍显式体现在当前循环顺序和 B 的下标访问中。"
              "下表直接读取当前 affine IR，不使用真实延迟或未来 LLVM IR。", "",
              "| Case | A | 最内层 IV | B 矩阵访问 | 内层连续访问 B |", "| --- | --- | --- | --- | --- |"]
    for record in runs[0]["records"]:
        if record.get("low_choice") != "none" or record["status"] != "ok":
            continue
        snapshot = next(s for s in record["snapshots"] if s["stage"]=="affine")
        text = Path(snapshot["path"]).read_text()
        inner = re.findall(r"affine.for (%[\w]+)", text)[-1]
        access = re.findall(r"affine.load %arg1\[([^\]]+)\]", text)[-1]
        lines.append(f"| {record['case_id']} | {record['high_choice']} | {inner} | {access} | {'是' if access.split(',')[-1].strip()==inner else '否'} |")
    lines += ["", "当前 IR 的循环/访问特征足以区分高层选择；静态启发式可以据此识别连续访问和 unroll 所作用的维度。"
              "本实验未验证这些启发式能预测全部 tile/unroll 的收益或精确排名。即使存在交互，也没有证明历史信息具有当前 IR 之外的增量价值。"
              "未开展 B0/B1/B2、反馈回溯、模型训练或候选选择策略比较，不进入神经架构阶段。", "",
              "代表性高层/中层 diff、实际优化命令和 kernel 函数反汇编见 [m4-lineage.md](m4-lineage.md)。"
              "快照级 parent 链保留；linalg 展开、tiling、unroll、CSE 和 LLVM 优化仍没有 Operation 级 N:M 来源。"
              "指令片段只提供定性证据，不能单独断言 cache、端口、寄存器或向量化是性能原因。", "",
              "## 原默认路径与产物重复", ""]
    lines += ["命名 matmul 的原默认路径与本次 generalize + identity-interchange 的 default 直接比较目标反汇编（不计算 hash）：", "",
              "```json", json.dumps(equivalence, indent=2, ensure_ascii=False), "```", ""]
    duplicates = [(i,r["case_id"],r["candidate_id"],r.get("code_duplicate_of")) for i,run in enumerate(runs,1)
                  for r in run["records"] if r.get("code_duplicate_of")]
    lines += [f"主实验目标反汇编重复：{json.dumps(duplicates, ensure_ascii=False)}。不同字节不自动证明算法或动态指令不同。", "",
              "## 复现与边界", "", "```bash", "bash scripts/phase0.sh m4", "```", "",
              "单入口完成两轮重编译、独立正确性、交错计时、默认代码等价检查和报告导出。"
              "[m4-interaction.json](m4-interaction.json) 保存配置、主机/工具/Git 状态、原始计时和区间；完整日志和产物只保留在 logs/ 与 tmp/。", "",
              "局限：一个程序家族，三个 Shape，两次独立启动但同主机连续实验；频率/共享负载/ASLR 未控制，无 PMU；热态复用不等于冷缓存带宽。"
              "参考 C++ 是有根据的常规实现，不是成熟自动调优系统的最优值。未计算 hash、未写单元测试。"
              "下一步先扩展独立家族和重复批次，围绕当前 IR 的透明规则建立对照，只有出现历史增量价值的可靠证据后再讨论持续状态。", ""]
    (destination / "m4-interaction.md").write_text("\n".join(lines))
    write_lineage(runs[0]["records"], destination)
    return destination / "m4-interaction.md"


def verify_default(records, output, logs):
    results = []
    for record in records:
        if record["candidate_id"] != "default" or record["status"] not in ("ok", "compiled"):
            continue
        original = {"events": []}
        source, defines = generate("matmul", record["input"]["shape"])
        compile_candidate(source, output/record["case_id"], logs/record["case_id"], original, (), defines)
        same = Path(original["disassembly"]).read_bytes() == Path(record["disassembly"]).read_bytes()
        record_path = logs/record["case_id"]/"record.json"
        record_path.write_text(json.dumps(original, indent=2)+"\n")
        results.append({"case_id": record["case_id"], "identical_disassembly": same,
                        "compile_seconds": original["compile_seconds"],
                        "record_path": str(record_path.relative_to(ROOT)),
                        "original_disassembly": str(Path(original["disassembly"]).relative_to(ROOT))})
        if not same:
            raise ValueError(f"generalized default differs from original: {record['case_id']}")
    return results


def run(config, destination=None):
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = ROOT / "logs" / f"m4-{stamp}"
    logs.mkdir(parents=True)
    paths, events = [], []
    for i, order_seed in enumerate((1,17),1):
        print(f"M4 batch {i}/2: recompiling and measuring (order_seed={order_seed})", flush=True)
        stdout = command(["bash", ROOT/"scripts/phase0.sh", "collect", "--config", config,
                          "--order-seed", order_seed], logs/f"batch-{i}.log", events, "collect", timeout=3600)
        path = Path(next(line.removeprefix("records: ") for line in stdout.splitlines() if line.startswith("records: ")))
        paths.append(path)
        print(stdout.strip(), flush=True)
    equivalence = verify_default(read_records(paths[0]), ROOT/"tmp"/f"m4-{stamp}"/"named-default", logs/"named-default")
    report = write_report(paths, destination, equivalence=equivalence)
    (logs/"workflow.json").write_text(json.dumps({"records": [str(p) for p in paths], "events": events,
                                                "wall_seconds": time.perf_counter()-started,
                                                "default_equivalence": equivalence}, indent=2)+"\n")
    print(f"report: {report}\nworkflow: {logs/'workflow.json'}", flush=True)
