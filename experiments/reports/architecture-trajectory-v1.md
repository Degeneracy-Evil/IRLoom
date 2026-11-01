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

## H2：已完成

四模型均从冻结 seed=23 的新初始化训练200步，Adam lr=0.0003，SmoothL1 beta=0.05。每步每训练程序取一条真实轨迹，程序内固定种子打乱后循环；先按程序、再家族等权（本批每家族一个程序）。每模型800次轨迹暴露、1600个实际S0/S1动作标签；不是800条不同采集轨迹。仅监督实际选中 q，两个阶段共享该轨迹末端样本，无未执行动作或S2标签、无ranking/辅助资源损失、无穿越编译器的反向传播。

A及其独立无记忆版本820229参数，B及其对照644101参数；每对step0权重逐张量相同，四模型使用完全相同的轨迹日程和暴露计数。沿每条自身S0→S1 Observation重新计算连续状态，未detach。无记忆版本仍共用原Encoder对离散历史的摘要（长度、成功和动作计数）及实际当前IR，完整动作日志仍保留，不能称完全没有历史信息。

| 模型 | train MAE step0→200 | validation MAE step0→200 | S1损失对S0隐藏状态梯度L1（训练前→后） | 秒 |
| --- | --- | --- | --- | ---: |
| A | .386594→.023060 | .326496→.057397 | 2.055336→.513199 | 190.595 |
| B | .289827→.025293 | .261693→.039540 | 1.931354→.446084 | 195.868 |
| A_current_only | .340583→.023873 | .291433→.057804 | 断连→断连 | 187.921 |
| B_current_only | .262945→.021954 | .240411→.041708 | 断连→断连 | 193.522 |

MAE为实际采样动作的根相对log代价误差，S0/S1先均值再按程序宏平均；本批家族宏平均相同。快照0/50/200均保留，总训练/快照/梯度检查wall合计767.907秒。只使用固定末轮，没有按验证表现选择较早checkpoint；例如A的step50验证MAE约.049838低于末轮，但未据此改配方。误差下降不能替代实际选路成功；Huber回归和有限随机后缀样本也不保证精确校准到条件期望。

`heads.py` 和 `guided.py` 均显式识别 `rollout-q-v1`，输出/排序采用负的共同根相对代价；S2保留S1分数，不走policy log-softmax。checkpoint记录πb、工具链及目标定义，旧all-advance权重改名成新协议会被拒绝，旧stage-q权重正常加载。旧数据、旧q语义和旧控制器路径保留。

实际失败/失效审计：专用检查点副本内注入变换失败、拒绝成功变换后的陈旧观察，父IR不变且两次失败扣预算；失败轨迹无目标时被数据读取器排除，带伪造数值目标则拒绝。没有额外链接终端或性能标签。证据为缓存 `integration.json`，不是单元测试工程。

```sh
bash scripts/architecture.sh --loomtrace train --name loomtrace-v1-pilot
bash scripts/architecture.sh --loomtrace audit --name loomtrace-v1-pilot
```

## H3

随后只执行已见PC1与Matrix上的四模型raw/guarded共16次有限编译验收。此处尚未宣称实际性能或记忆收益，完成后补充；不自动重跑历史矩阵。
