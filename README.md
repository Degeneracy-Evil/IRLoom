# IRLoom

**Cross-level, hardware-aware learning for compiler optimization.**

IRLoom 是一个面向编译器性能优化的研究项目。我们的出发点是：多级 IR 使编译器能够逐步降低程序抽象层次，但跨阶段语义、历史优化选择与目标机器的实际性能信息，往往没有被充分用于后续决策。我们希望研究一种可学习的跨层级优化机制，在保留编译器正确性约束的同时，帮助其发现更高性能的实现。

项目以 **MLIR / LLVM** 为首选实验基础，但研究方法不绑定固定的 Transformer 配比、特定算子或 NVIDIA GPU。硬件相关知识应作为输入和反馈，而不是写死在通用优化核心里。我们的目标是**改善最终生成代码的实际性能**，而不只是提高模型对 IR 的理解分数。

## 当前阶段：Phase 0 — Evidence before architecture

先建立可复现的实验路径，采集不同优化轨迹的多阶段 IR 与最终运行性能，检验「跨层级历史信息是否在相同优化/搜索预算下改善性能决策」。这一阶段**不训练大型模型，不重写整个编译器**，也不预设我们的研究假设必然成立。

研究涉及三条候选机制：跨 Lowering 持续表示（Persistent IR State）、跨层级信息交互（Cross-Level Interaction），以及硬件感知的性能反馈和优化回溯（Hardware-Aware Feedback）。它们是等待验证的架构假设，不是已得到证明的贡献。

## 阅读顺序

1. [研究定位与假设](docs/RESEARCH_PLAN.md)：问题定义、架构候选、贡献边界及研究路线。
2. [Phase 0 实验方案](docs/PHASE0_PROTOCOL.md)：数据采集、对照实验、指标、正确性与复现要求。
3. [本地 Agent 启动任务](docs/AGENT_START.md)：按顺序实施的具体任务与第一阶段验收条件。
4. [相关研究与技术参考](docs/RELATED_WORK.md)：优先复核的工作及官方技术文档。

## 工作原则

- **最终执行性能优先。** 编译通过、预测准确并不自动等于优化有效。
- **实验公平。** 比较不同方法时固定合法变换集合、目标硬件和搜索/测量预算；显式统计编译与搜索开销。
- **语义正确。** 只执行现有编译器支持的合法变换，每个候选都必须通过正确性测试；浮点误差容忍度必须明示。
- **机制通用。** 首个可用原型可在一种 CPU 上建立，随后通过 ARM、RISC-V 或 Ascend NPU 检验跨平台适应性。
- **先小后大。** 先验证跨层级信息的增量价值，再决定是否需要专用 Attention、持续状态更新网络或强化学习。

## 已验证的本机入口

```bash
bash scripts/phase0.sh         # 自动准备本地 MLIR，编译、校验并采集 24 条候选
bash scripts/phase0.sh m0      # 仅运行最小链路验收
```

实际环境、产物位置、限制见 [运行指南](docs/RUNNING.md)，阶段结果见 [开发记录](docs/PROGRESS.md)。

M0–M3 首轮已完成：三类工作负载、每类两个 Shape、每个 Case 四条合法路径；两轮正式采集均 24/24 正确、各有 360 个计时样本。见 [首批证据报告](experiments/reports/phase0-initial.md) 和 [简明结果汇总](experiments/reports/phase0-summary.json)。这批证据支持路径性能差异，尚未验证历史 IR 的预测价值；M0–M3 未训练模型。

M4 已完成真实的 linalg 顺序 × affine 动作实验，含三个缓存容量规格及独立 C++ 性能参考。两轮各 21/21 正确，共 630 个计时样本；中、大尺寸观察到 tile16 收益随高层选择反转。见 [M4 结果解读](experiments/reports/m4-findings.md)、[完整交互报告](experiments/reports/m4-interaction.md) 及 [IR/函数代码样例](experiments/reports/m4-lineage.md)。区别仍保留在当前 affine IR，尚不证明历史状态有额外价值。

```bash
bash scripts/phase0.sh m4      # 两轮跨阶段实验、C++ 参考、默认等价检查及报告
```

> **给本地 Agent：** 从 [docs/AGENT_START.md](docs/AGENT_START.md) 开始，阅读相关实验协议后分阶段执行。不要跳过 baseline 与数据质量检查直接训练模型。

## Phase 1 首轮：M5 前瞻优化决策

M4 确认了高层循环顺序与后续 affine 变换之间可重复的性能交互，但仍未证明跨阶段历史信息的独立价值。M5 基于受控候选池研究高层贪心决策的损失，再使用少量程序实例检验有限预算的跨阶段前瞻策略。详见 [M4 审查与 M5 实验任务](docs/M5_PLAN.md)。

M5 规则实验已完成：两家族、十个输入实例，两轮各 60/60 正确，共 1,800 个计时样本。同样四次候选评估的离线前瞻在三个新 matmul 实例上避免了贪心损失，归约简单策略已近最优。见 [结果解读](experiments/reports/m5-interpretation.md)、[完整报告](experiments/reports/m5-findings.md) 和 [冻结协议](docs/M5_PROTOCOL.md)。没有训练模型或声称在线优化收益。

```bash
bash scripts/phase0.sh m5-a    # 仅分析已提交 M4 数据
bash scripts/phase0.sh m5      # 两轮新池、固定预算对照与报告
```

## M6：可学习的跨阶段优化决策

M5 已验证固定四候选的均衡探针覆盖在部分 Matmul Shape 上优于阶段贪心，但它并不是自适应前瞻或学习模型。下一步扩展合法优化空间，比较随机、静态、反馈式与小型可学习价值模型，并据此决定是否引入跨层级 Attention/持续状态。实施要求见 [M5 审查与 M6 规划](docs/M6_PLAN.md)。

M6.0–M6.2 pilot 已完成：三个模板、20个输入、每输入12条候选，两批各240/240正确，共4,800个计时样本。透明 ridge 在小预算下改善部分决策，但 matmul 的强固定覆盖与模型持平，融合链简单规则已达池内最优；8个训练输入/两个训练模板不足以触发 M6.3，未开发神经网络。见 [首轮解读与下一步](experiments/reports/m6-interpretation.md)、[完整报告](experiments/reports/m6-findings.md)、[冻结协议](docs/M6_PROTOCOL.md) 和 [冻结模型](experiments/reports/m6-model.json)。

```bash
bash scripts/phase0.sh m6-a    # 仅读取 M5 数据，穷举覆盖与随机参考
bash scripts/phase0.sh m6      # 两批扩展池、线性模型、盲测策略与成本报告
```

## M7：全局计算图上下文与跨层级表示

M6 表明小型 Ridge 可以改善部分候选选择，但强固定覆盖在 Matmul 上表现相当，而简单融合链的默认实现已经接近最优。为了检验真正的编译架构价值，M7 优先研究**保持局部计算一致、改变周围消费者与融合/物化上下文**时，跨阶段最优决策是否变化；随后对照局部 IR、完整当前 IR 与跨层历史的信息价值。见 [M6 审查与 M7 研究方案](docs/M7_PLAN.md)。本阶段不预设需要 Transformer。

M7.0–M7.2 试点已完成：三种上下文、两个规模、六条路径，两批各36/36正确、共1,080个计时。大规模观察到融合后tile32在单消费者与归约复用间的方向反转，但固定none仍优于信息查表，上游历史未改变选路；没有进入神经架构阶段。动作作用全图，不能把该信号宣称为仅局部目标变换的因果证据。见 [首轮解读](experiments/reports/m7-interpretation.md)、[完整报告](experiments/reports/m7-findings.md) 和 [冻结协议](docs/M7_PROTOCOL.md)。

```bash
bash scripts/phase0.sh m7
```

## M7 审查后的研究重定位

M7 的上下文反转来自整图 Pass，严格局部匹配场景尚无独立选路收益。继续扩大相同场景的数据不足以证明跨层级神经架构。研究焦点转向**影响生产者与消费者接口的联合编译决策**，优先检查中间布局、物化/融合与后续执行代价之间的真实耦合。见 [研究重评估与下一项可行性验证](docs/RESEARCH_REASSESSMENT.md)。下一步仅从 M8.0 的工具链合法性调查开始，不自动启动模型训练。

## 大型真实程序 Benchmark 环境

当前优先准备LLVM test-suite、Zstd、SQLite（含speedtest1）与LLVM/Clang，固定官方版本，源码/大构建放在/mnt/data/chp/irloom-benchmarks，经忽略的benchmark-data软链接访问。默认复用已安装Clang/LLVM18.1.3，仅构建小型LLVM组件，避免整套bootstrap成本；不处理SPEC CPU2026。见[运行与来源说明](docs/REAL_BENCHMARKS.md)。

```bash
bash scripts/benchmarks.sh
```

单入口自动获取缓存、增量构建、正确执行和-O3基线测量，并验证ThinLTO/PGO工具能力；本阶段不开发新的编译优化算法或神经网络。

环境验收已完成：四项目六项工作负载，两次完整流程均通过；源码/大构建等约4.24GiB留在外部目录，缓存复跑约28s。ThinLTO与PGO基础能力通过，不代表整项目性能对照。详见[初始验收结果](experiments/reports/real-programs-initial.md)。

## 真实程序：下一轮性能基线与优化机会审计

官方源码与六项 smoke/环境验收已完成，但目前 LLVM/Clang 未构建完整被优化程序，正式真实负载的代表性与优化配置对照仍需加强。下一轮优先扩充 Zstd、SQLite 和官方 LLVM MultiSource 的执行负载，对至少两个项目建立 O3-native、ThinLTO-native（以及可行的 PGO+ThinLTO）强基线，再调查真实热点和跨函数决策。执行方案见 [docs/REAL_PROGRAM_NEXT.md](docs/REAL_PROGRAM_NEXT.md)。

R1–R3已实跑：Zstd/SQLite四种完整程序编译配置，加上两个官方MultiSource工作负载，共14组合、140个正式样本均通过；并完成gprof/PGO/内联记录审计。运行 `bash scripts/benchmarks.sh all --suite formal` 可复现完整流程。结果与后续最小A/B问题见[正式基线和审计报告](experiments/reports/real-programs-baselines.md)；这些是强基线与候选证据，尚未验证新的全局优化收益。

## 真实程序首轮热点验证

R1–R3 的正式强基线和调用审计已完成：Zstd 的增量 ThinLTO/PGO 收益较小，SQLite 的 PGO+ThinLTO 对 main 负载的提升较明显，但这不是新编译优化算法的收益。当前优先执行两个最小因果实验：**Zstd 热调用链上的内联 A/B** 和 **SQLite 训练覆盖差异的 PGO A/B**，确认成熟编译器是否仍存在值得学习模型协调的性能机会。详见 [docs/REAL_OPTIMIZATION_AB.md](docs/REAL_OPTIMIZATION_AB.md)。

两项最小因果A/B也已完成：Zstd强制内联在PGO下对测试压缩语料观察到小幅收益，SQLite改变PGO训练分布对新索引负载观察到收益；两者均未证明神经编译器架构贡献。审查结论和仅建议的最后一项混合PGO强基线见[docs/REAL_AB_REVIEW.md](docs/REAL_AB_REVIEW.md)。
