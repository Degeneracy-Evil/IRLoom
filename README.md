# IRLoom

**MLIR-first neural compiler architecture for cross-level performance optimization.**

IRLoom 研究神经网络与多级编译过程的内部耦合，目标是在正确性与编译成本约束下生成运行性能更好的完整程序。MLIR 是主要集成平台，LLVM 保留为标准后端；方法不绑定某个厂商、固定层数配比或单一工作负载。

## 当前阶段：架构 V1 开发

前期 MLIR 采集、规则实验、真实程序基线与 PGO/Pmix 研究已收束。当前不再寻找“简单规则必须失败”的前置证明，而是实现已确定的两套架构，再进行公平性能实验与组件消融。

**A / Interleaved** 在每个关键 IR 阶段后插入阶段专用 Transformer，接收当前 IR 与上一阶段隐藏表示，并选择真实执行的合法变换。

**B / Recurrent** 使用跨阶段共享的门控更新器，维护模块、函数和区域的持续状态，在相同阶段与动作空间中作决策。

两者共用结构化 IR 图、来源账本、硬件输入、动作执行器、正确性检查和有限预算搜索器。网络必须在编译中起作用，不能只是编译后打分或为整条固定 Pipeline 选择名称。现阶段是设计与开发计划，**尚未实现或验证 A/B 神经编译原型的性能**。

## 当前阅读入口

- [架构设计 V1](docs/architecture/V1_DESIGN.md) 定义阶段、双流、图表示、A/B、状态对齐与训练边界。
- [开发规划 N0–N5](docs/architecture/V1_DEVELOPMENT.md) 定义实现顺序、并行的大程序接入和验收标准。
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

新 `scripts/architecture.sh` 是规划中的统一入口，待实现后才提供真实命令。当前可用的旧入口如下。

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

旧结果未构成 A/B 神经架构的有效性证明，也不否定尚未实现架构的潜力。当前顺序是：**完成设计中的架构 -> 训练与真实程序实验 -> 按消融证据改进架构**。
