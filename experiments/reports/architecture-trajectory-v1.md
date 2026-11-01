# LoomTrace V1：真实顺序轨迹监督

2026-11-01，规划基线 `c5e1da2`。按 [H1–H3计划](../../docs/architecture/LOOMTRACE_V1_PLAN.md) 完成有界 pilot，复用 A/B、hierarchical-v3、角色化读取和固定修复版 SDK。旧恢复数据、权重及报告保留。实现里程碑为 `4fdbaf3`（H1）、`94e6c42`（H2）；采集时真实Git状态另存各环境记录，不能把后来的提交号当采集当时HEAD。[机器摘要](architecture-trajectory-v1.json) 保留范围、覆盖、误差、梯度、动作、预算及实际成本。

## H1：已完成

冻结配方 `experiments/architecture/loomtrace-v1.json`：训练 PC1/Lists/Puzzle/ReedSolomon，验证 Matrix/DES，六个不同家族；每程序 12 条真实随机顺序轨迹和一个根默认，共 78 个终端尝试，低于 120 上限。选择依据是已见域、家族和已有双阶段结构，不按新性能筛选。

`visible-structural-random-v1`：advance 0.4，非 advance 0.6 先均匀抽结构层（kind、所属函数、循环深度、callee），再均匀抽该层候选；没有非 advance 时退为概率1的 advance。每轨迹独立预定种子，S0/S1/S2 逐步只看当前可见 IR/注册候选，记录实际动作概率。S2 仅 advance，未执行候选没有标签。

72/72 随机轨迹完整输出正确，默认另计。按完整 ELF 去重后 37 个代表产物完成六轮随机/反向计时，222 次计时逐次检查输出。程序级不同 ELF 数（含默认）为 PC1 5、Lists 3、Puzzle 4、ReedSolomon 6、Matrix 7、DES 12。成功、失败、别名和默认分别计费，原始图、动作、原生样本及命令都留外部缓存。

| 程序 | split | S0非advance / S1非advance / 两者都有 | 相对根默认的有益/中性/有害（描述性） |
| --- | --- | --- | --- |
| PC1 | train | 5 / 9 / 3 | 3 / 9 / 0 |
| Lists | train | 7 / 7 / 4 | 0 / 12 / 0 |
| Puzzle | train | 7 / 7 / 3 | 0 / 10 / 2 |
| ReedSolomon | train | 5 / 8 / 3 | 4 / 8 / 0 |
| Matrix | validation | 6 / 7 / 2 | 0 / 9 / 3 |
| DES | validation | 7 / 11 / 6 | 0 / 12 / 0 |

共21条双阶段非advance路径，60/72中性；22条与根默认完整ELF相同。描述性效果沿用绝对0.05ms、相对2%、2×MAD分辨率，不用于额外ranking监督。有效轨迹既有重复路径，也有不同动作同码，不能把每条轨迹或两个阶段当独立程序。

PC1、Lists、DES的抽样S0前缀没有改变结构身份定义的S1候选集合；Puzzle、ReedSolomon、Matrix存在候选变化。缓存 `coverage.json` 按实际S0前缀列出抽到的S1后缀数、不同完整ELF和末端y范围；`data-audit.json` 分别记录同前缀随机后缀样本标准差。它们是有限抽样证据，不是所有动作组合的因果分解，随机后缀方差也不能等同六次原生计时的MAD。当前能形成真实、正确且不同码的联合路径，未证明相对于单阶段动作的稳定联合性能增益。

独立新进程重放 PC1 `pc1:3` 的 inline→unroll2→advance 路径通过：全部模型可见节点、边、候选、关系映射、历史和硬件特征相同，生成完整 ELF 字节相同且独立完整输出正确。运行标签和未供模型消费的环境字段允许不同，不声称原始文件的所有路径字符串相同。

数据目标为 `log(T_rollout/T_root_default)`，只给该真实轨迹 S0/S1 已执行动作；默认仅为参照，S2 不训练 q。πb恰好随机抽到全advance的路径仍是行为样本，与人为强制的默认参照分开。这是固定行为策略后缀的一次 Monte Carlo 样本，不是 all-advance 反事实、最优Q或独立双样本。

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

作为事后只读的误差背景，始终预测根相对代价0的MAE为 train .016270、validation .016152，均低于四模型末轮MAE；它没有拟合或额外编译。训练RMSE有下降，但验证预测没有建立可靠优势。不能仅凭随机初始化到末轮的loss下降宣称已学好联合选路，更不能把Huber的稳健位置拟合称无偏的条件期望估计。数据、训练与控制器局限需要分开解释。

`heads.py` 和 `guided.py` 均显式识别 `rollout-q-v1`，输出/排序采用负的共同根相对代价；S2保留S1分数，不走policy log-softmax。checkpoint记录πb、工具链及目标定义，旧all-advance权重改名成新协议会被拒绝，旧stage-q权重正常加载。旧数据、旧q语义和旧控制器路径保留。

实际失败/失效审计：专用检查点副本内注入变换失败、拒绝成功变换后的陈旧观察，父IR不变且两次失败扣预算；失败轨迹无目标时被数据读取器排除，带伪造数值目标则拒绝。没有额外链接终端或性能标签。证据为缓存 `integration.json`，不是单元测试工程。

```sh
bash scripts/architecture.sh --loomtrace train --name loomtrace-v1-pilot
bash scripts/architecture.sh --loomtrace audit --name loomtrace-v1-pilot
```

## H3：有限闭环已完成

已见PC1（train）与Matrix（validation），四个冻结末轮模型各raw/guarded，共16次，完整输出与确认全部正确，实际32个终端、112次动作尝试、240次原生计时；没有失败或测试域调用。raw每次3尝试/1终端，guarded每次11尝试/3终端，同一配方均未超过18/3上限。所有scorer确为 `rollout-q-v1`，每个proposal的最终guidance等于实际S1动作分数，S2未覆盖，未落入policy概率累计。

| 模型 | PC1 raw / guarded相对默认比 | Matrix raw / guarded | PC1 guarded输出 |
| --- | --- | --- | --- |
| A | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 默认 |
| B | 1.0902 / 1.0900 | 1.0000 / 1.0000 | inline `assemble` path[0,0,191] |
| A_current_only | 1.0000 / 1.0570 | 1.0000 / 1.0000 | inline `assemble` path[0,0,131] |
| B_current_only | 1.0000 / 1.0926 | 1.0000 / 1.0000 | inline `assemble` path[0,0,191] |

完整ELF与默认相同的比值规范为1，实际搜索/确认原始时间均保留；其它比值为该模型guarded默认的搜索中位数除以选中产物五次确认中位数。raw与guarded不是同批平衡交错测量，此表仅为有界开发域编译反馈，不作新的正式泛化性能表。guarded分别支付默认、候选、特征、推理、失败尝试及确认，记录日志命令时间29.0168秒、16次guided调用wall合计40.8898秒；后者未包含外层四次模型加载和Python编排，不能称全部评价阶段总耗时。所有方法共享协议；默认保护、beam与实测选优不能算连续记忆收益。

B在PC1 raw选中的内联与B_current_only guarded相同，两者完整ELF一致；全部最终选中路径的S1都是advance。S0部署没有先前连续状态，模型间S0差别可以受训练梯度影响，但不是现场读取跨阶段记忆的证据。PC1选中的非默认S0动作均不在这12条行为采样覆盖内，真实执行正确性和时间已取得，但新贪心/beam策略的状态动作分布已偏离πb。Matrix的模型/系统产物均与默认相同。尚无稳定连续记忆收益或新blind结论。

因此闭环局限同时存在：当前动作多数同码/中性，有益S1联合监督稀疏；有限数据下采样动作误差仍不够好；贪心/beam还会选到行为数据未覆盖的动作。不能把无收益全部归因于其中单一因素，也不能据本轮断言A/B整体必然无效。本轮不以加层、增步、补测试标签或搜索新benchmark应对。

## 缓存、总成本与停点

单一包装入口及细分入口：

```sh
bash scripts/architecture.sh --loomtrace all --name loomtrace-v1-pilot
# 只执行已冻结模型的有限开发域编译：
bash scripts/architecture.sh --loomtrace evaluate --name loomtrace-v1-pilot
```

`all` 顺序包含采集/审计、新进程联合重放、四模型训练、失败/初始化审计和有限编译；同名完整缓存直接复用。fresh-process replay 的子进程日志单独保存。更换名字会重新执行同一有限配方，部分session/训练快照不会覆盖；本轮没有必要再采一遍。新checkpoint的πb、目标、revision与源dataset都明确记录；不会将旧修复数据混入rollout监督。

同名all重复通过，330个被检查文件字节保持不变（含旧campaign/恢复数据、旧新权重、元数据、旧SDK二进制），200个实际run目录不增，无额外编译/原生计时。不匹配旧q协议的缓存在创建会话前拒绝。正负验收见 `cache-audit.json`、`integration.json`，shell语法与diff检查通过，无hash或测试工程。

采集78个、联合重放1个、有限闭环32个，共111个正确链接终端；失败/失效审计另有3次部分动作尝试、0新终端，Susan另有1次失败导入、0终端。原生计时合计462次；训练总wall约767.907秒，采集109.373秒、闭环40.890秒分别记录，不把重放、环境探测、报告与缓存审计假装为零成本，也不混进原生运行延迟。硬件/CPU0亲和性、工具flags和原始命令在各session；频率、外部服务负载、fork/exec开销未完全控制，Matrix短程序尤其受计时下限影响。

H1–H3首轮停点已达到：真实顺序轨迹可重放、两个决策阶段使用整条路径末端监督、A/B跨阶段反传及无记忆对照通过、四模型同预算训练和有限编译完成。没有要求正收益才收束，也没有扩大候选或新blind。更大程序/通用动作扩展和正式扩规模需另行冻结，本轮到此停止。
