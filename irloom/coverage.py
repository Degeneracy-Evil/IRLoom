"""M6.0: read-only exhaustive coverage and seeded random references on M5."""
import itertools
import json
import random
import statistics

from .compiler import ROOT
from .decision import costs, latency


def analyze():
    source=ROOT/"experiments/reports/m5-results.json"
    data=json.loads(source.read_text())
    output=[]
    for batch,run in enumerate(data["runs"],1):
        for family in ("matmul","reduction"):
            records=[r for r in run["records"] if r["program_id"]==family]
            cases={c:[r for r in records if r["case_id"]==c] for c in sorted({r["case_id"] for r in records})}
            ids=[r["candidate_id"] for r in next(iter(cases.values()))]
            pools={c:{r["candidate_id"]:r for r in rs} for c,rs in cases.items()}
            oracles={c:min(map(latency,p.values())) for c,p in pools.items()}

            def evaluate(plans):
                ratios=[]
                charge=0
                for c,selected in plans.items():
                    ratios.append(min(latency(pools[c][x]) for x in selected)/oracles[c])
                    charge+=sum(costs(pools[c][x])["charged_wall_seconds"] for x in selected)
                return {"geomean_regret":statistics.geometric_mean(ratios),
                        "oracle_covered":sum(x<=1+1e-12 for x in ratios),
                        "charged_wall_seconds":charge}

            subsets=[]
            for selected in itertools.combinations(ids,4):
                subsets.append({"candidate_ids":selected,**evaluate({c:selected for c in pools})})
            random_rows=[]
            for k in (2,4,6):
                for policy in ("random","stratified_random"):
                    seeds=[]
                    example=None
                    for seed in range(100):
                        rng=random.Random(seed)
                        plans={}
                        for c,pool in pools.items():
                            if policy=="random":
                                selected=rng.sample(ids,k)
                            else:
                                highs=sorted({r["high_choice"] for r in pool.values()})
                                selected=[x for a in highs for x in rng.sample([i for i in ids if pool[i]["high_choice"]==a],k//2)]
                            plans[c]=selected
                        if seed==0:
                            example=plans
                        seeds.append(evaluate(plans))
                    regrets=[x["geomean_regret"] for x in seeds]
                    random_rows.append({"policy":policy,"k":k,"seed_count":100,"seed_range":[0,99],
                                        "mean_geomean_regret":statistics.mean(regrets),"min":min(regrets),"max":max(regrets),
                                        "median":statistics.median(regrets),"seed0_queries":example,"seeds":seeds})
            output.append({"batch":batch,"family":family,"case_count":len(cases),"all_fixed_subsets":subsets,
                           "full_coverage_subsets":sum(s["oracle_covered"]==len(cases) for s in subsets),
                           "mean_subset_geomean_regret":statistics.mean(s["geomean_regret"] for s in subsets),"random":random_rows})
    dest=ROOT/"experiments/reports"
    (dest/"m6-coverage.json").write_text(json.dumps({"source":str(source.relative_to(ROOT)),"offline":True,"results":output},indent=2)+"\n")
    lines=["# M6.0：M5 固定覆盖与随机预算参考", "",
           "仅读取已提交 M5 JSON，不编译或运行基准，也不修改 M5 冻结策略及原报告。"
           "旧 lookahead 是预先固定的 balanced coverage，反馈只用于最后选最优，不据反馈改变查询。", "",
           "## 同一固定四候选子集用于一个家族全部 Shape", "",
           "每个六候选池共有 C(6,4)=15 个子集。先对每个子集计算逐实例 regret 的几何均值，再对十五个结果作算术平均；"
           "不是把每个实例事后最佳子集当成可部署策略。", "",
           "| 家族 | 轮 | 全 Shape 覆盖 oracle 的子集 | 十五子集平均 GM regret |", "| --- | ---: | ---: | ---: |"]
    for r in output:
        lines.append(f"| {r['family']} | {r['batch']} | {r['full_coverage_subsets']}/15 | {r['mean_subset_geomean_regret']:.6f}× |")
    lines += ["", "## 随机和分层随机：每实例独立抽取唯一候选", "",
              "每个策略 100 个固定种子 0–99；分层随机均分两个 A。种子分布不是 100 个独立程序。"
              "表格为每个种子的家族 GM regret 再取均值及范围；JSON 保留每种子结果、实际记录成本和 seed=0 的查询。", "",
              "| 家族 | 轮 | k | 策略 | 平均 GM regret | 种子范围 |", "| --- | ---: | ---: | --- | ---: | --- |"]
    for r in output:
        for x in r["random"]:
            lines.append(f"| {r['family']} | {r['batch']} | {x['k']} | {x['policy']} | {x['mean_geomean_regret']:.4f}× | [{x['min']:.4f}, {x['max']:.4f}] |")
    lines += ["", "随机抽取不查看标签；全池数据只用于事后计分。每个 k 对齐候选编译和原生调用预算，记录墙钟离线求和，未实际在线随机运行。"
              "穷举后的最优子集只是回顾性覆盖上界，不用于后续盲测选策略。M5 的固定覆盖成功并非反馈式或学习型优化的证据。", "",
              "下一步冻结 2×6 池、额外融合模板、train/validation/blind 分组和固定/随机/反馈规则，再采集正式标签。", ""]
    path=dest/"m6-coverage.md"
    path.write_text("\n".join(lines))
    return path
