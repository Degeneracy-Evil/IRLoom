# IRLoom V2：MLIR 原生多层优化环境与 A/B 联合验证

状态：下一轮正式开发任务（尚未实现、尚无新性能结果）。审查基线 `9183c7893b40ec7b4d8bb5089eb903cc2462648a`，制定日期 2026-11-09。

## 0. 研究判断与本轮取舍

N0–N5、V1.1–V1.3.1、G1–G3、正确性修复与 LoomTrace V1 已证明：**A/B 神经结构、来源/状态、真实编译动作、离线训练、完整输出检查和有限搜索的工程链路可运行**。但 LoomTrace 72 条真实轨迹中 60 条与默认性能不可分辨，验证预测未优于始终预测零变化。PC1 的收益来自 S0 内联，S1 多为 advance，独立无记忆模型也可得到同一可执行文件；因此既没有可信的多阶段协同收益，也没有持续记忆的增量证据。

现在最应改变的是**编译器提供的优化空间与真实工作负载**，而不是马上再训练十倍 LoomTrace、换损失、换网络骨干、切到 PGO 或重新从 N0 开始。

**首选路线：锁定项目现有 MLIR 18.1.3 原生 Linalg/Tensor → LLVM → CPU 的执行链，在一份完整、真实、多计算区域的 MLIR 工作负载上实现由 A/B 逐阶段控制的融合、分块及尽可能的向量化。** C/Polygeist 内联+unroll、修复版 `cleanup-positional-v1` 保留为历史回归，不再承担主论文性能结论。若原生 MLIR 18.1.3 某条 API 不支持，按锁定源码核对后选择等价受支持接口，不照抄最新文档的可能已变化语法。不能混合不同版本 MLIR SDK/静态库。

本轮保留 A（阶段专用 Transformer）、B（共享门控记忆）、`hierarchical-v3`、可信来源映射、检查点/失败恢复、正确性协议。允许为**真实新动作的参数表示**增补共同输入层，但不重写 A/B 状态公式，也不引入 RL/Hybrid、自定义加速 kernel。

## 1. 先选一个可持续的完整工作负载入口（M1）

1. 使用仓库已有 `compiler/pipelines/architecture-v1.json`、MLIR 18.1.3 SDK、`compiler/mlir_bridge/main.cpp` 和现有 native runner 作为起点。检查真实 API/Pass 顺序与动态库链接：当前原生 S0 在 fuse 后立即 bufferize 并转成 affine loops，使 S1 已无法做 Linalg 的 tensor-level tiling/fusion；这一顺序必须调整，而非只在 JSON 中添加动作名。
2. 选取**来自可追溯公共项目/模型的完整多算子计算图**作为第一主工作负载，例如包含多层 matmul/conv、bias、activation、残差或归一化的完整前向计算。入口应覆盖至少数个存在生产者-消费者关系的计算区域；记录源项目/版本/许可证、完整运算链、输入规模和调用范围。**不要把单个 GEMM、复制的同一小算子、或手写放大的 toy 称为完整应用**。可先利用 native 小例做集成 smoke，但最终只把完整、可执行、具有正确输入输出的程序放进研究评估。
3. 用独立 CPU/NumPy/PyTorch 源实现或原模型执行结果作为 oracle；原始输入、随机种子、精度、公差、入口 ABI、实际输出一并冻结。浮点重排引入的舍入差异使用预先明确的误差界，不能在看见结果后放松；无 fast-math、无省略必须运行的算子。支持至少一种实际规模与另一种合理输入尺寸，不根据加速效果挑尺寸。
4. 在上述真实程序上**先保证默认路径从输入 MLIR 到 native 正确执行**。与此同时直接推进 M2 的注册动作，不把“先证明现有规则失败”设置为阻塞 A/B 开发的门槛。
5. 第二份不同拓扑/来源的完整工作负载用于开发域交叉检查。若第一种导入失败，保留失败并在预先登记的候选顺序尝试下一条；不再次无限排查 SQLite/Susan 这类历史前端阻断。只有必要时才单独用匹配版本 IREE 的 native runner 作替代，**IREE 属于隔离工具链路线，不与旧 MLIR 18.1.3 混链，也不同时建设两个完整编译平台**。

必须报告：实际完整图规模（操作/region/函数）、阶段前后 dialect 和数据流、可独立检查的输出、默认编译/运行耗时、真实内存/代码大小以及工具链版本。不以源码行数冒充图复杂性。

## 2. 将优化放在正确的 IR 层次（M2，核心工作）

当前 `architecture-v1.json` 的 S0 lowering 为
`--one-shot-bufferize --buffer-deallocation --canonicalize --cse --convert-linalg-to-affine-loops`，过早消灭 tensor/linalg 语义。应建立版本化新 pipeline（例如 `native-structured-v2.json`），**不要覆盖旧 manifest**：

```text
真实完整 Linalg/Tensor/func 模块
  S0 (tensor): 注册合法 producer-consumer fusion / 保留边界
  -> canonicalize/cse，保留 tensor/linalg
  S1 (tensor/linalg): 对具体目标执行合法 tile / 不 tile；按实际接口可选有限、明确的 tile shape
  -> canonicalize/cse，保留适合后续 vectorize 的结构
  S2 (linalg/vector，若可支持): 目标化 vectorize / 保留常规方式
  -> one-shot-bufferize + deallocation + 必要 linalg/scf/affine/vector lowering
  -> LLVM IR -> 固定 Clang -O3 -> native
```

对 MLIR 18.1.3 若 vectorization/bufferization 的真实先后约束不同，按**已执行通过的具体 IR 不变量**调整（例如部分 vector 变换与 bufferization 顺序）；不能为迁就示意图执行非法 pipeline。S2 在尚未支持时明确仍为观察阶段，**但 M2 至少要实际接通 S0 与 S1 两个具有非 no-op 候选的阶段**，不能仅新增未消费的 candidate JSON。

- S0 复用已有 `linalg::fuseElementwiseOps` 等确实支持的结构化 legality API；若工作负载需要 contraction/tile-and-fuse，选用锁定版本已有的 TilingInterface / Transform Dialect API，不能让不安全的内存/数值重排静默通过。
- S1 注册结构化 tiling，记录 target、tile sizes、必要 interchange 及前提；不再只是 affine loop 的 `unroll2`。只提供固定、适度的合法参数集合（例如 shape/硬件支持的少量候选），不开展无界 tile-size brute force。检查切片、余数、循环结果和下游 consumers 的语义。
- S2 若能在本轮以受支持 API 合法实现目标化 vectorization，则纳入；若成本明显过大，维持 S2 观察并给出明确原因，**不阻塞已承诺的 A/B 两阶段真实闭环**。不能只为达成 3 个阶段而加无效动作。
- 每一步均在独立私有 snapshot 中注册、按实际 ID/参数校验并执行，失败扣预算且回滚，改写后重新获取完整合法候选；来源只在可确定时维持，不能跨一般 lowering 伪造精确锚点。
- **重要：新动作参数必须实际进入 A/B 读出。** 当前 `Heads.parameter_embedding` 基本只读 `operand`，无法可靠区分同一 target 上不同 tile shape/vector size。新增同一套有类型动作参数编码，并实际用于候选评分；对不存在的参数使用 mask。版本化旧 checkpoint/读出，避免悄悄改变 LoomTrace 的旧语义。特征不得包含程序名称或未来实测延迟。

用现有 integration smoke 检查真实融合、tiling、余数/边界、参数区分、候选排序与实际 IR 变化、独立数值输出、重放一致、branch/rollback；不新建庞大单元测试工程或计算 hash。旧 C/Polygeist 正确性修复保持隔离。

## 3. 建立两种真正可比较的编译基线（M3）

除了全 `advance`，还应给出**事先固定的结构化确定性优化管线**作为强实践基线（合理的固定融合/分块/向量化规则；如有现成匹配编译器默认策略也保留）；两者使用完全相同的输入、精度/ABI、LLVM 后端和正确性。

本轮不是先运行无限规则搜索再“允许神经网络开发”。M2 一旦能够生成正确不同程序，直接把 A/B 接到真实 S0/S1（可选 S2）动作，使用既有结构编码、记忆/无记忆对照及状态 checkpoint。不要把整个预先配置好的流水线名称当成单个动作；应由神经模型在真实阶段观察之后选择目标与变换。

用固定预算做一个**小而清楚的多阶段交互审计**：选同一程序中的少量合法高层前缀，在其下选择少量合法后续动作，验证不同前缀是否改变后续可达候选、是否改变不同动作的实际效果。只选择具有真实来源/合法性的目标，最多每个工作负载约 4 个高层前缀×3 个后续候选，**不是笛卡尔穷举全部动作空间，也不依据加速表现筛掉负例**。对无法跨前缀对应的目标，注明不具可比性，不能硬套旧 snapshot ID。

交互证据优先使用同组对应的完整可执行文件和实际性能，而不仅是不同文本 IR。实验既能报告交互明显，也能报告后端 O3 抹掉差异；两者是实际科学结论，不因此无限追加 benchmarks 或手调参数。

## 4. 有限神经训练和评估（M4）

在 M1–M3 的真实环境上复用 LoomTrace 顺序数据采集基础，轨迹由冻结、依赖当前合法候选的行为策略产生；包含 **S0 fusion → S1 tile / 可能的 S2 vectorization** 的真实路径和终端性能。训练目标/行为策略语义必须明确，未选择的动作不能伪造标签。默认及强固定策略是基线，不以它们的终端延迟悄悄改写另一路的监督目标。

首批只训练四个主模型：A、B、A_current_only、B_current_only。相同数据、种子、模型可见信息、编译预算和候选参数编码，冻结训练方案，先一轮小规模训练及开发域正确闭环；不要再上 8–16 模型×上千次的旧矩阵。保持“已见开发域结果”与未来按完整程序家族冻结的 blind 留出严格区分，不得将同一图不同形状/尺寸放到不同 split 充作独立家族。

优先报告四层事实：

1. **优化机会**：不同正确 ELF、全 advance 与固定强基线之间的性能空间、同一前缀及跨前缀动作效果；
2. **模型学习**：实际训练轨迹、校准、真正选中动作的覆盖、S1 对 S0 state 的梯度、无记忆差别；
3. **端到端程序**：A/B 的完整真实执行性能、失败/默认回退、完整系统工作量及复测，而非仅 q 的 MAE；
4. **编译成本**：pass/IR 编码/图摘要/模型前向/检查点/多进程和原生执行成本，单列可控 pass 时间与整体编译总时间，**不把跳过一个 pass 的节省当成整个神经编译器的节省**。

编译时间作为本轮同步采集的未来贡献入口；**本轮不临时改目标为“最快编译”、不把 compiler-time 和 runtime 相加后以未定义权重声称加速**。以后可设计约束 `T_run <= (1+ε)T_fixed` 下最小 `C_compile` 的新实验。

## 5. 交付、资源上限和停点

按 M1（真实完整工作负载及原生执行）→ M2（有效的多层注册变换和参数读出）→ M3（基线与少量多层对照）→ M4（A/B 小规模训练和编译闭环）连续开发，各阶段独立提交并 push；**M1 不应变成无限的前端兼容性研究**。当前 repo 的历史工具、权重、数据、失败证据不覆盖，新的构建与大量 IR 留项目忽略缓存。

建议新增 `compiler/pipelines/native-structured-v2.json`、对应 C++ bridge 的版本化执行分派、一个可追溯的真实工作负载入口、`experiments/architecture/mlir-v2.json` 和 `experiments/reports/mlir-v2-pilot.md`；补充运行文档与 `AGENTS.md` 状态。可以增设一个 `--mlir-v2` 子命令，复用已存在 Session/编译/测量框架，不搭第二套通用优化平台。

首批资源约束：1 个主要真实完整程序 + 1 个不同结构的开发交叉检查程序；先单机 CPU 固定输入与少量形状。基线和交互诊断每程序控制在约 20 条实际完整程序路径以内（失败也计费）；神经训练/有限 raw/guarded 如需要另外版本化小预算，不自动运行所有历史对照或任何新 blind。未获得大程序时如实称为原型/开发实验；不通过扩大 toy 结构补足代表性。

**必须完成的停点**：有正确的完整 Linalg/Tensor native 程序，S0/S1 至少两个阶段真实可执行的候选，A/B 实际能根据新的合法参数选动作并保持状态/恢复，强确定性基线与完整性能/编译时间记录，以及有限开发域消融。若暂时缺乏明显加速，也不能直接宣称网络结构无效；同时不再无限刷新旧数据，报告是动作覆盖、后端归一化、模型学习还是编译成本限制结果。

之后是否扩到 IREE、完整模型族、新的 blind 或编译时间优化方向，以 M1–M4 交付后的证据和用户决定为准，不让 Agent 自动外延。

## 技术依据（运行时以锁定版本为准）

- MLIR Linalg 的结构化融合、分块、向量化: https://mlir.llvm.org/docs/Dialects/Linalg/
- MLIR Transform Dialect/教程（控制 payload 具体变换，注意与 pinned 18.1.3 API 的差异）: https://mlir.llvm.org/docs/Dialects/Transform/ ; https://mlir.llvm.org/docs/Tutorials/transform/
- MLIR One-Shot Bufferization（tile/fuse 等 tensor-level 变换应在 bufferization 前评估）: https://mlir.llvm.org/docs/Bufferization/
- 若原生路线因固有 ABI/前端限制确实失败，IREE 可作为隔离版本的完整 CPU 路线: https://iree.dev/developers/general/developer-overview/ ; https://iree.dev/guides/deployment-configurations/cpu/
