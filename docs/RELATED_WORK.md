# 相关研究与技术参考（持续更新）

这是一份**避免重复造轮子、明确研究边界**的简短阅读清单，不是对现有领域的穷尽性综述。以下仅列出已核实的论文页、会议页、项目页或官方文档。引用这些工作不意味着它们的实验结果已在本项目环境中复现。

## 1. 最相关的既有研究

| 工作 | 与 IRLoom 的交集 | 要特别避免的重复 |
| --- | --- | --- |
| [BO3: Workload-Specific Compilation Pipelines for Multi-Level Compilers（CASES 2026）](https://zenodo.org/records/21825045) | 在 MLIR/IREE 多阶段编译中使用合法性约束、图表示与强化学习搜索 Pass 顺序。含实现与实验材料。 | 单纯「多阶段 RL + GNN 选择 Pass」不够新。 |
| [Multi-Stage Sequential RL for MLIR Meta-Optimization（EuroLLVM 2026）](https://llvm.swoogo.com/2026eurollvm/session/4050964/multi-stage-sequential-rl-for-mlir-meta-optimization) | 直接讨论 MLIR 多阶段优化中学习策略的组织。 | 共享编码器、不同阶段独立 Agent，并非空白命题。 |
| [PRISM: Structure-augmented LLMs for High-Level Synthesis Pragma Optimization（MLCAD 2026）](https://arxiv.org/abs/2609.38601) | 通过交叉注意力把 AST/CFG/DFG 的结构信息注入 LLM，用于 HLS 优化。 | 「结构输入 + Cross-Attention」本身已经有先例。 |
| [FAIR: Flow Type-Aware Pre-Training of Compiler Intermediate Representations（ICSE 2024）](https://arxiv.org/abs/2309.04828) | 针对编译器 IR 建立区分不同流类型的图 Transformer 表示。 | 「把 IR 转成图、用图 Attention」不能单独称为创新。 |
| [Mirage: A Multi-Level Superoptimizer for Tensor Programs（OSDI 2025）](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) | 跨 GPU Kernel/Thread Block/Thread 层级统一表示和联合搜索。 | 跨层级联合优化已经有不依赖神经网络的强先例。 |
| [TVM MetaSchedule（官方文档）](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) | 硬件实测、可学习代价模型、合法调度空间、搜索预算管理。 | 「学习性能代价 + 搜索候选 + 实机测量」不新。 |
| [DFA-Net（CC 2025）](https://conf.researchr.org/details/CC-2025/CC-2025-main-conference/9/DFA-Net-A-Compiler-Specific-Neural-Architecture-for-Robust-Generalization-in-Data-Fl) | 将编译器数据流分析的计算模式融入神经架构。 | 从编译器算法派生网络架构是一条已有路线，但该工作主要聚焦数据流分析泛化。 |

建议阅读顺序为 **BO3 → EuroLLVM 多阶段 RL → Mirage → PRISM → FAIR → MetaSchedule → DFA-Net**。这让我们先认清已经具备哪些编译优化能力，再思考是否值得引入新的持续神经表示。

## 2. MLIR / LLVM 官方基础设施

- [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)：用 IR 显式控制作用于 Payload IR 的变换；可用于建立可控的合法候选集合。
- [Transform Dialect 教程](https://mlir.llvm.org/docs/Tutorials/transform/)：理解 Transform 与 Payload 的关系和具体操作。
- [MLIR Action Tracing](https://mlir.llvm.org/docs/ActionTracing/)：拦截或追踪 Pass、Pattern 等不同粒度的变换动作；初期也可用 IR before/after 打印记录快照。
- [MLIR TrackingListener](https://mlir.llvm.org/doxygen/classmlir_1_1transform_1_1TrackingListener.html)：跟随部分 IR 替换更新 Transform State；它不能自动保证一般 1:N/N:M 变换的完整语义来源追踪。
- [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)：理解 Pass Manager、Pipeline、Pass 生命周期与打印选项。
- [LLVM llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html)：对目标微架构指令调度与资源压力进行静态分析。结果是近似估计，不能代替真实性能测量。
- [LLVM TargetTransformInfo](https://llvm.org/doxygen/classllvm_1_1TargetTransformInfo.html)：可查询目标相关的部分代价与硬件特征，说明现代编译器并非只知道 ISA。

## 3. 当前的新颖性假设

我们尚未确认有现成工作完整实现以下组合：**持续跟随 IR 演化的学习状态 + 显式变换历史或来源关联 + 不同抽象层级的信息交互 + 使用低层真实执行结果修正高层优化决策**，并以更高的实际代码性能或更低的搜索预算作为主要收益。

这只表示目前检索尚未发现完全同构的研究，并不等同于证明空白。即使组件组合以前未出现，也必须拿出新的机制、可复现的有效性证据和严格的消融，才能构成研究贡献。

## 4. 后续重点检查的问题

1. 既有多阶段优化方法是否已隐式地保存完整历史？我们的持续状态相比全部重编码是否有计算或学习优势？
2. 历史 IR 信息对当前 IR 的有限容量表示究竟补充了什么？如果最终性能完全由当前状态决定，增益可能来自搜索效率而不是信息量增加。
3. 显式高低层关联能否带来可检验的性能归因与回溯策略，而不仅是可视化解释？
4. 在相同候选空间、实测次数和总优化时间下，架构优于随机搜索、静态启发式和现有学习代价模型吗？
5. 是否存在足够通用的目标机器能力表示，以便在 ARM、RISC-V、Ascend 等平台上复用优化机制？

## 5. 维护规范

每新增一项论文，记录：发布时间与状态、发表会议/期刊或预印本、代码/数据位置、优化任务、目标硬件、模型输入和动作空间、实际性能收益与测试基线、与 IRLoom 的重合点。未亲自验证的论文性能数字需要注明来源，不要直接用于和我们未来的实验横向比较。
