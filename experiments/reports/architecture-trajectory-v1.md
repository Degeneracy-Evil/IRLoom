# LoomTrace V1：真实顺序轨迹监督

2026-11-01，规划基线 `c5e1da2`。按 [H1–H3计划](../../docs/architecture/LOOMTRACE_V1_PLAN.md) 开展有界 pilot，复用 A/B、hierarchical-v3、角色化读取和固定修复版 SDK。旧恢复数据、权重及报告保留。

## H1：已完成

冻结配方 `experiments/architecture/loomtrace-v1.json`：训练 PC1/Lists/Puzzle/ReedSolomon，验证 Matrix/DES，六个不同家族；每程序 12 条真实随机顺序轨迹和一个根默认，共 78 个终端尝试，低于 120 上限。选择依据是已见域、家族和已有双阶段结构，不按新性能筛选。

`visible-structural-random-v1`：advance 0.4，非 advance 0.6 先均匀抽结构层（kind、所属函数、循环深度、callee），再均匀抽该层候选；没有非 advance 时退为概率1的 advance。每轨迹独立预定种子，S0/S1/S2 逐步只看当前可见 IR/注册候选，记录实际动作概率。S2 仅 advance，未执行候选没有标签。

72/72 随机轨迹完整输出正确，默认另计。按完整 ELF 去重后 37 个代表产物完成六轮随机/反向计时，222 次计时逐次检查输出。程序级不同 ELF 数（含默认）为 PC1 5、Lists 3、Puzzle 4、ReedSolomon 6、Matrix 7、DES 12。成功、失败、别名和默认分别计费，原始图、动作、原生样本及命令都留外部缓存。

独立新进程重放 PC1 `pc1:3` 的 inline→unroll2→advance 路径通过：全部模型可见节点、边、候选、关系映射、历史和硬件特征相同，生成完整 ELF 字节相同且独立完整输出正确。运行标签和未供模型消费的环境字段允许不同，不声称原始文件的所有路径字符串相同。

数据目标为 `log(T_rollout/T_root_default)`，只给该真实轨迹 S0/S1 已执行动作；默认仅为参照，S2 不训练 q。这是固定行为策略后缀的一次 Monte Carlo 样本，不是 all-advance 反事实、最优Q或独立双样本。

复现（仓库根目录）：

```sh
bash scripts/architecture.sh --loomtrace collect --name loomtrace-v1-pilot
bash scripts/architecture.sh --loomtrace replay --name loomtrace-v1-pilot
```

数据在 `benchmark-data/architecture/trajectories/loomtrace-v1-pilot/`，日志在 `logs/architecture`，临时文件在 `tmp`。已完成缓存直接复用；部分 session 不覆盖。H1 命令账本 70.4501 秒、六程序采集 wall 合计 109.3731 秒，不含后续训练/重放/审计。频率及外部主机负载未完全控制。

## 较大真实程序：已完成有界负结果

唯一预选 Susan 完整源码 2125 行，固定前端在 K&R float 参数的 f32/f64 调用检查失败，未获得有效图或终端，没有改算法、关 verifier 或更换程序。详见 [独立报告](architecture-trajectory-v1-workload.md)。该失败不阻塞现有 A/B 训练，不代表已完成工业规模评价。

## H2–H3

H1 阶段此处尚未报告训练或编译收益。随后按冻结配方进行四模型各200次更新、真实跨阶段梯度验收和有限已见开发域 raw/guarded；结果完成后在此补充。旧 `stage-q-v1` 与新 `rollout-q-v1` 保持独立，不自动重跑历史矩阵。
