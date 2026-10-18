# M5 前瞻决策：冻结规则与评估协议

依据 [M5_PLAN.md](M5_PLAN.md) 先完成只读 M5-A，再采集和比较。M5-A 的实现及冻结策略提交为 `32cdeee`；不以新增数据调阈值、挑掉不利 Shape 或更改探针集合。

## 候选池与实例

配置 `experiments/m5.json` 保持 2×3 因子设计：matmul 的高层 ijk/ikj，行归约的高层 ij/ji；低层均为 none/tile16/unroll4。每个实例六条合法路径，不把 oracle 列入同预算排名。

| 家族 | 新 A×B 的 Shape | 预先固定 iterations |
| --- | --- | ---: |
| matmul | 48,48,48 | 2048 |
| matmul | 64,96,48 | 1024 |
| matmul | 95,65,33 | 2048 |
| matmul | 128,32,96 | 1024 |
| matmul | 32,192,128 | 512 |
| matmul | 192,48,64 | 512 |
| row reduction | 32,96 | 16384 |
| row reduction | 96,32 | 16384 |
| row reduction | 127,259 | 2048 |
| row reduction | 256,128 | 2048 |

六个 matmul Shape 与 M4 不重合，包含方形、非整齐和不同长宽比。行归约是额外的非 matmul 决策家族，复用 Phase 0 的独立 C double 参考；部分尺寸已有低层基线经验，不能冒充完全未知的程序家族泛化测试。两个程序模板、十个实例不足以训练或验证神经网络。

行归约的 linalg.generic 原始迭代轴是 parallel i、reduction j；合法交换为 reduction j、parallel i。每行 j 仍递增，不修改输出或浮点归约次序。变换匹配 generic，不影响输出清零的 linalg.fill；后续 affine Pass 可作用于所有循环 nest，成本和结论都针对整体 Pass，不孤立归因主循环。

MLIR 18 的 Transform schedule 由 `compiler/schedules/ij.mlir`、`ji.mlir` 明示。源输入、schedule、high_decision、affine、transformed、LLVM 和目标代码沿用既有采集。预验收已实际编译 60/60 路径，并在两类各一个 Shape 的 12 条路径上完整校验、采样和导出。

## 固定调用预算，避免全池校准依赖

旧实验统一取全池校准所需的最大迭代次数。它适合描述路径差异，但离线选路若据此计费，会隐含使用未查询候选的校准结果。本轮每个 Shape 在新标签产生前固定 iterations；不执行任何自适应校准。不承诺固定批次必定达到旧校准的 10 ms 目标，实际原始计时全部保留。

每候选先校验 seeds 0/1/7/19，再做 15 次计时；每个 measure 进程还含一次初始校验调用、三次预热。所有策略在同一 Shape 使用四个候选，因此严格对齐：

- 四次完整候选编译（每次八个编译/翻译/链接工具进程，另有反汇编）；
- 十六次 correctness 进程、六十次 timing 进程、零次 calibration；
- `4 × [4 + 15 × (iterations + 4)]` 次原生 kernel 调用。

超时、编译失败、数值失败继续保留状态和日志。失败消耗预算但不能成为最佳候选。完全重复机器代码另标记，不凭变换名称认定新实现。

## 冻结的策略与信息边界

输入只包括决策前 high IR 的程序类别、静态 Shape/行主布局、目标描述和计划合法动作；同一主机的硬件特征为常量。静态规则：matmul 在 N≤32 且 K≤32 时选 ijk，否则 ikj；row reduction 选连续 j 最内层的 ij。

三种策略都返回自己四个已测候选中的最好值：

1. static：静态选 A，查询该 A 全部 B，然后查询另一 A 的 none。分配不读取测量标签。
2. greedy：查询 A0/none、A1/none，以已知默认延迟选 A，再查询该 A 的 tile16/unroll4。
3. lookahead：查询两个 A 各自的 none、tile16，以各分支已探测的最优值估计未来潜力。它会漏掉未探测的 unroll4，不预设总是优于另外两种策略。

`irloom/decision.py` 的 query 限制四个唯一候选，策略只使用已暴露的延迟。完整候选池最优仅在策略完成后用于 regret，不能决定分支或探针。该设计为**离线模拟**，没有实际按所选次序在线重跑编译器。顺序/缓存状态变化的成本尚不能从离线记录精确还原。

按独立家族和 Shape 报告两批结果，不随机拆分变换或计时样本，无训练/测试拟合。M4 是开发集，新 matmul 是未在 M4 见过的尺寸，归约是额外的受控决策家族；不声称学得的程序/硬件泛化。

## 成本与对照目的

按 query 对真实子进程成本离线求和：候选 compile_seconds、correctness/calibration/timing 进程墙钟、计时原生区间、编译器进程数及原生调用数。校准为零。全池采集的实际总墙钟另列；不能把不同策略的反事实费用相加成实际执行时间。

这是严格相同编译/执行调用数预算，**不等于相同墙钟预算**。慢路径的费用可能更高，必须披露。Python 记账/决策时间单独实测，包含标签查表和中位数计算，不伪装成模型推理；无模型、零训练参数，训练/模型推理成本为零。

B0=当前 IR 的 static；当前 A 决策点尚未经过 lowering，上游原始语义与当前输入等价。因此 B1-static 使用同一个零参数策略别名做零信息差核查；没有独立历史编码器或非平凡输入消融。核查同样 query/选择/成本，不把相同结果作为“历史普遍无用”的证据。

B2 用 lookahead 表示显式分支与已测反馈，四次预算相同。它的探针分配也变了，若有提升，应归因为被探测下游覆盖和反馈，而非历史 IR 编码或专用架构。此对照不足以隔离持续状态和 Attention 的增量价值。

## 单入口与停止原则

```bash
bash scripts/phase0.sh m5-a   # 已提交 M4 数据的分析，不重跑基准
bash scripts/phase0.sh m5     # 两轮新数据 + A/C/D 的报告
```

两轮独立重编译、order_seed=1/17，其他输入和参数相同；同主机连续执行，不是严格独立硬件重复。所有原始日志/JSONL 在 logs/、生成 IR/二进制在 tmp/。派生结论、原始样本摘要与查询成本提交到 experiments/reports/。

静态规则近最优的家族应保留负结果并停止扩大模型；若有规划收益，也只说明这个池中的有限前瞻值得继续检验。下一步仍优先独立模板、线上可中断的墙钟预算与更强启发式，不开发 Transformer、RL 或持续神经状态。不计算 hash，不写单元测试。

## 正式验收

已从干净 `32fc5b2` 执行完整单入口，两轮各 60/60 正确、900 个计时样本；查询预算、原生调用数及快照父链均核查通过。三种 Shape 的 matmul 规划损失在两批重复，归约简单策略已近最优。结果见 [m5-findings.md](../experiments/reports/m5-findings.md)，对结论范围和 B1 别名核查的解释见 [m5-interpretation.md](../experiments/reports/m5-interpretation.md)。未开展学习或架构级历史表示消融。
