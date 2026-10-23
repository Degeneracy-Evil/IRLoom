# IRLoom

**MLIR-first neural compiler architecture for cross-level performance optimization.**

IRLoom 研究神经网络与多级编译过程的内部耦合，目标是在正确性与编译成本约束下生成运行性能更好的完整程序。MLIR 是主要集成平台，LLVM 保留为标准后端；方法不绑定某个厂商、固定层数配比或单一工作负载。

## 当前阶段：架构 V1 开发

前期 MLIR 采集、规则实验、真实程序基线与 PGO/Pmix 研究已收束。当前不再寻找“简单规则必须失败”的前置证明，而是实现已确定的两套架构，再进行公平性能实验与组件消融。

**A / Interleaved** 在每个关键 IR 阶段后插入阶段专用 Transformer，接收当前 IR 与上一阶段隐藏表示，并选择真实执行的合法变换。

**B / Recurrent** 使用跨阶段共享的门控更新器，维护模块、函数和区域的持续状态，在相同阶段与动作空间中作决策。

两者共用结构化 IR 图、来源账本、硬件输入、动作执行器、正确性检查和有限预算搜索器。网络在编译过程中选择并执行合法变换。**N0–N5 首轮原型和完整程序 pilot 已完成，尚未建立跨阶段记忆带来增量性能收益的证据。** LLU/PC1/trie 的训练、验证、留出评估及同预算 beam、断连/重置、容量匹配对照见 [首轮结果](experiments/reports/architecture-v1-pilot.md)。这三个规模有限的完整程序不代表大型项目覆盖。

V1.1 修正了结构顺序、边关系/方向/端口和来源使用统计，新增按程序组内归一化的多程序训练、独立训练的 A/B 无持续记忆对照，以及冻结产物的平衡交错测量。正式配置为 `experiments/architecture/pilot-v1-1-canonical.json`，结果与限制见 [V1.1 报告](experiments/reports/architecture-v1-1-pilot.md)。两次新进程导入检查用于排除无序 DLTI 打印带来的输入变化；旧试跑和 V1 产物均保留。

V1.2 已接通匹配检查点的阶段 q 监督与实际动作排序，并加入同预算计费的默认保护；保留 A/B 骨干。12 套 checkpoint、192 次 raw/guarded 编译及 2040 次冻结交错计时见 [V1.2 报告](experiments/reports/architecture-v1-2-pilot.md)。新 DES 测试族未显示可靠加速，默认保护的效果不算记忆收益。完整入口：`bash scripts/architecture.sh --v12 all --name my-v12`。

V1.3 新增由内容产生的分层摘要和角色化候选读出，修正 q 的校准目标，并以固定结构种子扩充到49个匹配动作终端。八套最小对照、120次 raw/guarded 编译及1320次冻结交错计时见 [V1.3 报告](experiments/reports/architecture-v1-3-pilot.md)。Perlin 新测试族仍无可靠收益；结构/数值修正通过验收，数据与验证选择仍有限。完整入口：`bash scripts/architecture.sh --v13 all --name my-v13`。

V1.3.1 已完成六模型固定20次更新、共享根成对 margin、12次训练/验证真实编译重放及缓存验收。B与独立无记忆B都选到已知有益动作，A仍未学好；未建立记忆或新泛化收益。入口：`bash scripts/architecture.sh --training-review all --name my-v131`，见 [训练修订报告](experiments/reports/architecture-v1-3-training-review.md)。保留旧权重与数据，不重跑历史性能矩阵。

## 当前阅读入口

当前 C1–C2 正确性审查已完成：定位并隔离修复固定 LLVM 的 affine unroll cleanup carry 接线错误，原失败路径、32 条匹配动作与 3 个冻结模型调用通过回归。273 条旧标签中，13 条（Puzzle/DES）修复后机器码改变，需要新性能标签；本轮未训练或重跑性能矩阵。见 [正确性与候选分析报告](experiments/reports/architecture-campaign-correctness.md)。复现：`bash scripts/architecture.sh --correctness-audit all --name c1-review`。新完整程序运行显式使用 `--compiler-revision cleanup-positional-v1`；历史默认保留 legacy。

G1–G3已完成首批活动：10训练/3验证/3留出完整程序、273条匹配记录、四模型×两种子各200次更新。S1监督与拟合有进展，仍无连续记忆收益；3条raw语义错误被拒，留出guarded全为默认产物。入口为 `bash scripts/architecture.sh --campaign all --name my-campaign`，见 [数据覆盖](experiments/reports/architecture-training-corpus.md) 和 [训练/部署报告](experiments/reports/architecture-training-campaign.md)。12/4/4覆盖目标和大应用接入的缺口保留。

- [架构设计 V1](docs/architecture/V1_DESIGN.md) 定义阶段、双流、图表示、A/B、状态对齐与训练边界。
- [开发规划 N0–N5](docs/architecture/V1_DEVELOPMENT.md) 定义实现顺序、并行的大程序接入和验收标准。
- [架构运行指南](docs/architecture/RUNNING.md) 提供依赖、统一入口、复现命令和已知限制。
- [AGENTS.md](AGENTS.md) 是本地 Agent 的当前任务入口。

| 里程碑 | 交付目标 |
| --- | --- |
| N0 | 锁定 MLIR 阶段与数据接口，默认路径能够暂停、保存并继续。 |
| N1 | 完成结构化图、受控动作、来源账本、检查点与失败恢复。 |
| N2 | 实现阶段专用 Transformer A，完成跨层训练连接和真实编译闭环。 |
| N3 | 在相同接口中实现共享状态 B，并验证分支隔离与恢复。 |
| N4 | 采集面向架构的真实轨迹，完成两套模型的首轮训练与留出评估。 |
| N5 | 比较完整程序性能、总优化成本及跨层连接/状态机制的消融。 |

真实程序进入 MLIR 的工作与 N1–N3 并行。优先选择前端实际支持的完整多函数/多区域程序；小样例只用于 smoke。未通过 MLIR 的 native 程序不能被宣传为 MLIR 架构主结果。前端、SDK、工具链与依赖按 revision 独立锁定，保留原实验工具链和缓存。

统一入口自动准备项目内依赖并增量构建。源代码、SDK、IR 和权重使用 `benchmark-data` 指向的外部缓存；首次完整程序运行需要构建固定版本 Polygeist。

```bash
bash scripts/architecture.sh --audit
bash scripts/architecture.sh --structure-audit
bash scripts/architecture.sh --model B --state-audit --multi-function
bash scripts/architecture.sh --program pc1
```

下一步先审查并独立重采受修复影响的标签及匹配参照，再决定后续架构活动；不自动启动旧全矩阵。完整历史训练和对照命令见运行指南。

## 已有可复用基础

旧系统已经具备 MLIR 多阶段快照、编译/运行日志、独立正确性检查、原始性能样本、受控候选与真实程序构建缓存。它们将作为新架构的基础，而不是继续扩展为 PGO 调权研究。

```bash
bash scripts/phase0.sh                         # 旧 MLIR 实验入口
bash scripts/benchmarks.sh                     # 真实程序环境/smoke
bash scripts/benchmarks.sh all --suite formal  # 既有强编译基线
```

运行约束与缓存位置见 [MLIR 运行指南](docs/RUNNING.md) 和 [真实程序运行指南](docs/REAL_BENCHMARKS.md)。源码、构建、profile、模型与大规模 IR 不提交到 Git；继续执行实际正确性和失败审计。按既有用户约定不额外计算产物 hash、不新建单元测试工程，但必要的集成 smoke 与回归验收不可省略。

## 历史研究与已完成结果

本节仅供追溯。旧文档中的“下一步”与模型开发门槛不自动延续到当前阶段，最新架构文档及用户指令优先。

| 阶段 | 文档与结论位置 |
| --- | --- |
| 初始定位与基础实验 | [原研究计划](docs/RESEARCH_PLAN.md)、[Phase 0 报告](experiments/reports/phase0-initial.md)。 |
| 跨阶段交互与规则搜索 | [M4](experiments/reports/m4-findings.md)、[M5](experiments/reports/m5-interpretation.md)。 |
| 线性模型与上下文试点 | [M6](experiments/reports/m6-interpretation.md)、[M7](experiments/reports/m7-interpretation.md)。 |
| 前期研究重评估 | [研究备忘录](docs/RESEARCH_REASSESSMENT.md)。 |
| 真实程序环境与强基线 | [环境验收](experiments/reports/real-programs-initial.md)、[R1–R3 基线与热点审计](experiments/reports/real-programs-baselines.md)。 |
| 局部内联与 PGO 覆盖 | [Zstd A/B](experiments/reports/real-zstd-inline-ab.md)、[SQLite A/B](experiments/reports/real-sqlite-pgo-ab.md)。 |
| 混合 PGO 收束 | [Pmix 报告](experiments/reports/real-sqlite-pmix.md)。 |

旧结果未构成 A/B 神经架构的有效性证明；新架构的首轮结果也需要扩大程序覆盖后再判断。研究顺序是：**实现架构 -> 训练与真实程序实验 -> 按消融证据改进架构**。
