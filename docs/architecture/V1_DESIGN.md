# IRLoom V1：MLIR 优先的双流神经编译架构

状态：架构设计基线。2026-10-23 已完成 N0–N5 首轮原型与有限程序 pilot，尚未建立记忆机制的增量收益证据；实现范围和限制见 [实验报告](../../experiments/reports/architecture-v1-pilot.md)。本文保留设计要求，不代表全部研究目标已达成。开发顺序见 [V1_DEVELOPMENT.md](V1_DEVELOPMENT.md)。

## 1. 目标与本阶段决定

目标是在相同语义、目标硬件和优化资源约束下，通过神经网络与 MLIR 编译阶段的内部耦合，改善最终完整程序的执行性能。最终评价同时报告运行时间、编译/搜索时间、模型推理和内存开销、代码体积。预测误差不是最终成功标准。

正式保留两个架构，不预判胜负：

- **A / Interleaved**：每个关键 IR 阶段后有专用 Transformer Block；当前 IR 编码与前一阶段隐藏表示交互，模型选择的动作在进入下一阶段前真正改变程序。
- **B / Recurrent**：同一组门控更新参数在不同阶段复用，持续维护模块、函数和区域状态；阶段标识与当前 IR 变化作为输入。

MLIR 是主要集成平台，不改成以 LLVM Pass selector 或 PGO 调权为核心。LLVM 保留为标准代码生成后端。先实现 A，再在共同基础上实现 B；Hybrid、强化学习、定制 KDA/Mamba 算子均不进入首版关键路径。

两套架构都必须完成第一轮实现和对照。旧 M6/M7 的“先证明廉价规则失败才能开发神经网络”门槛不再适用于本阶段。性能不佳时依据组件消融调整架构，而不是继续追加无期限的前置实验。

## 2. 真正的编译内耦合

```text
当前 MLIR I_s
    -> 结构化图编码 X_s
    -> 神经模块 A_s 或共享模块 B_theta
    -> 有类型的候选动作排序
    -> 编译器在检查点副本上执行动作
    -> 校验、更新来源关联、继续必要 lowering
    -> 下一阶段 MLIR I_(s+1)

跨阶段连续表示 H_s / S_s 跟随以上过程传递。
完整候选生成机器代码后，正确性与真实性能返回该候选的搜索记录。
```

只有预测分数、不实际影响编译路径，不算完成架构。只有选择一个预先整条流水线的名字，也不算逐层耦合。前后阶段必须能够针对变换后 IR 重新观察和决定动作。

神经表示不是可执行程序；编译器仍维护真实的 SSA、类型、控制流与内存语义。模型不直接生成任意 IR 文本，不允许修改精度、fast-math、ABI 或工作负载来换取速度。

## 3. MLIR 阶段与首版动作范围

Dialect 可以共存，不能将 dialect 名字机械地当成网络深度。由一个版本化 pipeline manifest 定义阶段、入口/出口不变量、必需 lowering 和可选动作。

首个原生结构化 MLIR 适配器采用三个决策边界：

| 边界 | 主要表示 | 首版候选动作 |
| --- | --- | --- |
| S0：高层结构 | module/func、linalg/tensor 及数据流 | 在指定生产者—消费者关系上选择受支持的融合或保留边界；多函数输入可增加受合法性接口支持的内联。 |
| S1：结构化调度 | 仍具有张量/循环语义的 linalg、scf、affine、vector，按实际 pipeline 混合 | 对指定区域执行受支持的分块、循环次序或向量化；不是只扫全函数 tile-size。 |
| S2：缓冲与向量实现 | 完成必要 bufferization 后的 memref、scf/vector | 选择已注册的向量/transfer lowering 策略；能够证明安全时可加入分配提升等变换。 |
| 出口 | LLVM dialect -> LLVM IR -> native | 使用固定标准后端；首版不另造 LLVM 机器调度模型。 |

上表是接口与动作类别，不保证任意版本/任意输入支持所有动作。每个适配器须公开真实支持集合；缺失阶段显式标记，不编造高层语义。首个完整原型须在至少两个阶段具有有效的非 no-op 选择，第三阶段可先观察，随后补齐动作。

张量上的 tiling/fusion 等必须在其要求的语义尚在时完成，不能为了整齐的三层图而一律提前 bufferize。C/C++ 前端产生的 scf/memref 路径可能没有 tensor 阶段，使用单独 manifest，不伪装成 linalg 输入。

每阶段允许有限步决策，默认第一版每阶段一次；后续重复动作复用该阶段 A Block，B 则继续复用共享更新器。必须提供 no-op/advance 和最大步数，避免编译循环。

## 4. 共享结构化表示

使用 MLIR C++ API 遍历真实 IR，不继续用文本正则统计作为核心模型输入。图包含 Operation、Value、Block/Region 与函数/模块摘要；显式边包括 SSA def-use、控制流、嵌套、直接调用、已知副作用/内存依赖。间接调用和未知别名使用 unknown/conservative 标志，不假装解析出精确关系。

节点特征来自操作类别、类型、shape、布局/索引、循环范围、已有分析结果及可见 profiling。硬件输入包括可获得的 ISA/向量能力、cache 等静态能力，缺失项带 mask。所有方法使用相同的机器信息；首轮单机不宣称硬件泛化。

全局计算采用函数/区域摘要，局部计算采用真实依赖邻域。大函数分区后必须保留跨分区边界摘要；不把所有 Operation 送进全程序稠密 Attention，也不通过随意截断丢掉余下程序。稀疏 Attention 的实现应实际按邻接/选定关系取值，不能只给稠密 N×N 矩阵加 mask 却声称线性开销。

## 5. 跨阶段对齐与来源账本

首版维护函数/区域级锚点，并对自己控制的变换记录精确的保留、拆分、合并、删除关系。Operation 标识是当前快照内的唯一 ID，跨快照关系由账本表达，不能用内存地址、SSA 打印名字或源码行号充当永久身份。

来源边记录 old IDs、new IDs、变换事件及 known/partial/unknown 状态。一般 lowering 无法追踪的细节直接降级到区域/函数锚点，重新编码受影响区域。新节点来自当前 IR 编码；删除节点不得留下可执行动作目标；多对一可池化，一对多可用旧状态加新节点特征。位置元数据仅用于辅助，不是等价性证据。

受支持的 Transform tracking/listener 可帮助追踪，但不能保证任意 Pass 的完整 N:M 对应。不要求首版实现通用全编译器增量分析系统。

检查点必须绑定 payload IR、来源账本、A/B 神经状态、动作历史、随机状态和预算。分叉时分别复制/分叉状态；回退不能把另一个分支的隐藏状态误接到当前 IR。

## 6. A / Interleaved 的具体计算

设当前图编码为 X_s，跨快照映射为 P_s，硬件上下文为 W，合法候选描述为 C_s：

```text
Z_s = Fuse(X_s, Align(P_s, H_previous), stage_embedding, W)
H_s = StageTransformer_s(Z_s, dependency_edges, summary_edges)
scores, value, resource_estimates = SharedHeads(H_s, C_s)
```

不同阶段的 StageTransformer 参数不共享；图编码接口和动作/性能头可共享。全局摘要与局部节点之间交换信息，上一阶段 H 通过对齐后的残差/交叉注意力进入本阶段。既不能退化为各层独立的预测器，也不能让所有阶段参数共享后仍称其为 A。

初版容量采用小模型，例如 hidden=128、4 个注意力头，每阶段一个结构化 Block。此数值仅为可改配置，不是研究贡献；不得据此声称 optimal。关键是完整跨层神经计算和真实决策接口。

## 7. B / Recurrent 的具体计算

```text
R_s = Read(S_previous, anchors, dependency_edges)
U_s = SharedUpdate_theta(X_s, R_s, delta_summary, stage_embedding, W)
S_s = GatedWrite(S_previous, U_s, lineage)
scores, value, resource_estimates = SharedHeads(Read(S_s), C_s)
```

状态分成 module/function/region 的记录，不把整个大程序塞进一个向量。首版采用清楚可审查的门控残差或 GRU 式更新，使用共享参数，并支持分支隔离与失效重编码；它不是已经实现的 KDA 或 Mamba，也不主张具有二者的理论/性能性质。后续可在同一接口替换为关联矩阵记忆等实现，但先完成 A/B 首轮比较。

A/B 在相同阶段调用，接收相同可见 IR、机器信息、历史事件与已测反馈。不能让 B 获得额外探针标签再把收益归因于状态架构。第一次全量重编码用于正确性参考；B 的增量更新优化独立统计和消融。

## 8. 模型—编译器接口

首版共享五类记录：

- Observation：schema、pipeline/stage、snapshot/branch、图、摘要、机器特征、来源关系、已可见反馈。
- CandidateAction：目标快照与节点、动作种类、参数、前提、适用状态、作用范围。普通候选不包含隐藏真实性能。
- Decision：候选排序、选中动作和预测值；不携带可执行任意代码。
- Transition：是否成功、真实 IR 变化、来源更新、分析失效范围、编译成本与诊断。
- TerminalMeasurement：完整程序正确性、原始计时、代码大小和总成本；只在候选已实际运行后可见。

统一接口形态是 observe -> propose/rank -> apply -> transition -> advance -> finalize/measure。候选生成器限制到已注册变换，执行器再次检查前提。试执行在副本上进行；失败记录、扣除预算并回到有效检查点，不使用损坏状态。

MLIR verifier 验证结构/类型等约束，不等于任意变换的语义等价证明。正确性依靠原有变换的前提与实现、显式数值/ABI约束及独立输出验证共同保证；不能把“verify 通过”直接写成形式化证明。

全局神经状态由顶层 driver/session 持有。在模块边界取得一致快照，首版决策串行；禁止让并发函数 Pass 无同步地读写同一个全局记忆或遍历正在被其它线程修改的函数。

## 9. 训练和在线执行

第一轮使用离线真实变换轨迹和末端性能监督，不通过机器代码运行时间反向传播。结构化图输入是常量，但 A 的跨阶段隐藏连接和 B 的状态更新仍在 PyTorch 中保持可微，不能阶段间全部 detach 后声称联合训练。

监督优先采用同一程序/硬件/负载的候选相对性能与排序，再加代码大小/编译成本辅助头。比较某个动作时尽量固定其余后续策略；完整轨迹标签代表实际采样 continuation 的结果，不把有限候选最小值称为真实 Q*。无性能标签的阶段可作结构训练，不使用“有更多测量的一方”取得不公平优势。

按完整程序、图族和输入家族划分训练/验证/测试。同一程序的快照、动作分支、重复计时不得泄漏到多个 split；PGO 训练输入与正式评测输入也隔离。数据采集、训练、推理、搜索成本分别报告。

先实现单路径编译：每阶段由 A/B 选择动作，末端运行。再用相同的有限 beam/检查点搜索器比较 A/B；搜索器不作为两种架构之间的新变量。保持独立的无神经标准流水线作为安全回退，错误候选不能进入性能排名。

## 10. 实现技术边界

C++ 实现薄 MLIR bridge/driver，负责结构提取、受控变换、来源和快照；Python/PyTorch 实现模型、训练及预算编排。第一版使用阶段级文件/消息边界，模型进程常驻，不每个操作启动 Python。不要求首版嵌入 LibTorch、自定义 CUDA/Triton kernel、通用 RPC 或新 dialect。

旧 MLIR 18.1.3 测量工具链与数据保留。架构工具链单独锁定 LLVM/MLIR revision、C++ SDK、运行库和模型依赖；需要 frontend/IREE 时随其匹配的 LLVM 固定构建，禁止混链不同版本 MLIR 库或覆盖旧工具。最小目标构建与缓存优先，不做无关 LLVM 全量 bootstrap。

## 11. 大程序接入与结论边界

小 MLIR 程序仅作结构/正确性 smoke，不再承担科学主结论。真实程序接入与 A/B 开发并行：优先检查 LLVM MultiSource 的真实 C/C++ 程序经 Polygeist 等前端进入 MLIR 的可用路径，至少保留多函数/区域关系，不只抽一个热点函数。不要默认 SQLite、Zstd 或整个 Clang 已能无修改通过前端。

若 C/C++ 接入受阻，使用 MLIR 原生端到端路径（例如固定 IREE 工具链的完整模型/多区域程序）继续验证架构，同时记录覆盖限制；这仍以完整工作负载为测量对象，而非回到单算子参数竞赛。C/C++ 与完整模型结果分组呈现，不互相替代“通用程序优化”证据。

所有正式 A/B 主结果须真实穿过多个 MLIR 决策边界。只把 LLVM IR 包进 LLVM dialect 不算恢复高层多级语义；未接入 MLIR 的 Zstd/SQLite 继续作为外部对照。架构实现验收与实测性能胜出是两个不同结论。

## 技术依据（具体 API 以锁定版本为准）

- MLIR Pass Infrastructure：https://mlir.llvm.org/docs/PassManagement/ 。尤其遵守并行 OperationPass 的访问与可变状态约束。
- MLIR Interfaces：https://mlir.llvm.org/docs/Interfaces/ 。用于通用结构/语义接口，不无限添加操作名特判。
- Transform Dialect 与教程：https://mlir.llvm.org/docs/Dialects/Transform/ 、https://mlir.llvm.org/docs/Tutorials/transform/Ch1/ 。动作指向 payload，但 handle 消耗/失效不等于永久来源标识。
- Bufferization：https://mlir.llvm.org/docs/Bufferization/ 。阶段顺序应遵循真实变换的语义要求。
- Polygeist：https://polygeist.llvm.org/ 。C/C++ 到结构化 MLIR 的候选接入路径，兼容性需逐项目验收。
- IREE：https://iree.dev/ 、https://iree.dev/developers/general/developer-overview/ 。完整 MLIR 编译/运行载体的备选，不能缩减成只调 tuning 参数。
