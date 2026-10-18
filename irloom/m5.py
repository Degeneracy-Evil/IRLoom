"""M5 analysis and fixed-budget offline decision experiments."""
import json

from .compiler import ROOT
from .decision import analyze_case, costs


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
