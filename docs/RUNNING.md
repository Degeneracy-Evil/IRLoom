# Phase 0 本机运行指南

M0–M5 使用 Python 标准库，M6 的透明线性模型增加 NumPy（由 uv.lock 复现），均通过 `uv` 运行。所有生成产物放在 `tmp/`，日志与原始 JSONL 放在 `logs/`；这些目录均被 Git 忽略。不执行系统安装，不计算哈希，不写单元测试。

## 当前验证环境与 M0 命令

2026-10-18：Ubuntu 24.04 amd64，Intel Xeon 6986P-C，LLVM / Clang / MLIR 18.1.3。系统已有 `/usr/lib/llvm-18/bin` 中的 LLVM 与 Clang，没有 MLIR。以下命令已实际执行：

```bash
bash scripts/phase0.sh setup
bash scripts/phase0.sh probe > logs/probe.json
bash scripts/phase0.sh m0
```

`setup` 将 Ubuntu 官方 `mlir-18-tools_18.1.3-1ubuntu1_amd64.deb` 解包至 `tmp/toolchain/`，幂等检查已有工具，不使用 sudo 或 apt install。系统配置的 apt 镜像下载返回 502，改用 archive.ubuntu.com 后成功。此包在当前已有 LLVM 18 的主机上可直接运行；不承诺适用于其他发行版。其他机器请提供匹配版本的 LLVM/MLIR，并用环境变量 `MLIR_OPT`、`MLIR_TRANSLATE`、`CLANG`、`LLC`、`LLVM_CONFIG` 指向工具。默认顺序为项目本地工具、PATH、系统 LLVM 18。

M0 读取 `benchmarks/matmul.mlir`，保存 `tmp/m0/{high,affine,transformed,scf,llvm_dialect}.mlir`、`kernel.ll`、`kernel.o` 和 `runner`。完整编译命令及输出见 `logs/m0/*.log`，元数据与计时记录见 `logs/m0/records.jsonl`。以本机 `mlir-opt --help` 核实 Pass，未使用假设性参数。

测量先独立校验所有输出元素（double 精度 C 参考实现，`rtol=atol=1e-4`，禁用 fast-math 与浮点收缩），固定到当前允许的第一个 CPU。3 次预热后按 2 的幂校准批量次数，保证校准批次至少 10 ms；记录 10 个批次的每次 kernel 延迟（ms）。计时不包含进程启动、输入分配、参考计算与校验，包含 kernel 自身输出初始化；重复访问同一组输入，属于缓存热态测量。编译耗时独立记录。CPU 频率、邻核负载不可控，PMU 未采集。

用户本轮指令优先于初始协议：**不计算任何 hash，不新增单元测试**。记录用快照路径、父路径、有序命令与 Git commit 建立关联，因此尚不满足原协议的密码学指纹要求。运行实际编译、数值正确性及计时作为必要验收。

## M1 统一入口

```bash
bash scripts/phase0.sh collect --config experiments/m1.json
bash scripts/phase0.sh compile --config experiments/m1.json
bash scripts/phase0.sh run --manifest logs/20261018T093009758842Z/manifest.json
bash scripts/phase0.sh summarize --records logs/20261018T092943494129Z/records.jsonl
```

上述路径是已执行的本地记录。自行运行时用 `compile` / `collect` 输出的新路径替换时间戳。默认操作为 `collect`。每轮新建 `tmp/<UTC时间戳>/` 和 `logs/<UTC时间戳>/`，不覆盖旧实验；无需手动修复或清理目录即可反复运行。配置为 JSON，无第三方依赖。

`compile` 写入可供 `run` 使用的 `manifest.json`；`collect` 一次完成编译、校验、测量和汇总。每条记录包含构建命令及其耗时、快照父子关系、编译主机和测量主机、独立复制的 C runner 源码、候选参数、输入与原始计时。`run.json` 保存配置和总墙钟开销，`schedule.json` 保存实际采样顺序，`summary.json` 保存 median/MAD/IQR、范围和相对 default 的加速比。失败记录留在 JSONL，入口返回非零，失败不参加性能排名。

各候选先用 seeds 1/7/19 逐元素校验，再分别校准；同一 Case 最终统一取所有候选所需的最大批次次数。15 轮中每轮随机打乱候选顺序（种子固定），所有候选有同样的有效计时预算。校准、正确性、编译及执行命令均保留。随机测量仍无法排除频率漂移和共享机器负载。

`ir_changed` 指 transformed 快照相对 default 的文本变化；`duplicate_of` 用完整可执行文件的直接字节比较识别重复，不计算 hash。不同字节不必然等于不同机器指令，解释时还须看代码与后续分析。

## M2 完整工作负载

```bash
bash scripts/phase0.sh
# 等价于：
bash scripts/phase0.sh collect --config experiments/phase0.json
```

默认配置为三类 × 两个 Shape × 四条候选，共 24 个。每类使用 default、affine tile4、tile16、innermost unroll4，lowering 和后端优化参数统一；所有 Pass 在本机 LLVM 18 的 help 与实际编译中验证。程序生成与 C 参考分别位于 `benchmarks/workloads.py` 和 `benchmarks/runner.c`。配置仅改变静态 Shape 与合法 Pass；不在候选间手写不同 kernel。

- matmul：M,N,K = 64,64,64 / 127,65,93；linalg.matmul + 输出清零。
- row reduction：M,N = 128,256 / 127,259；linalg.generic + 输出清零，输出为 M×1。
- 2D five-point stencil：内部 M,N = 128,256 / 127,259；输入含一圈 halo，不更新边界；输出等于中心和四个邻居之和乘 0.2。输入从 affine 循环开始，没有单独的高层 stencil dialect。

所有尺寸均为 f32，独立 double 参考在候选执行前计算；校验 seeds 0/1/7/19，seed 0 为零输入，其他为确定性正负数。输出预填 NaN 检测未完整写入；输入输出前后添加哨兵，计时后再次检查输出及哨兵。没有允许浮点重关联或 fma contraction。哨兵不能替代完整的内存安全证明。

另已实际用临时 JSON 对三类 1×1 最小边界输入运行四条候选，12/12 成功。配置格式与实际记录字段见 [DATA_SCHEMA.md](DATA_SCHEMA.md)。`code_duplicate_of` 直接比较目标文件的标准化反汇编（含字节和重定位），不计算 hash。最小尺寸的四条路径均收敛为相同机器代码，主矩阵 24 条候选未发现此类重复。

包装脚本在编译所需的 MLIR 不存在时自动执行本地 setup，已有工具则跳过。初次运行仍需网络下载约 43 MiB；uv 环境也由包装入口自动创建。

## M3 证据报告

以下命令已运行，输入来自干净 M2 提交的两轮完整实验：

```bash
bash scripts/phase0.sh report --records logs/20261018T093503802387Z/records.jsonl logs/20261018T093526501912Z/records.jsonl
```

输出到 `experiments/reports/`：`phase0-initial.md` 为报告，`phase0-summary.json` 为简明对照表，`phase0-initial.json` 为原始计时、输入、配置、工具/机器元数据、编译命令与快照路径的审计摘要。重复 report 会确定性重建这些派生文件，无需重新编译。它依赖每轮 JSONL 同目录的 `run.json`，适用于 collect 的完整实验输出。

报告保留 median、MAD、IQR、范围、相对 default 加速比和同轮候选延迟比的探索性 bootstrap 区间。原始 JSONL/日志/快照/二进制仍只在本机保留。现有三家族六规格的数据不足以评估历史特征、未见程序泛化或在线搜索；不运行模型。

## M4 跨阶段因子实验

```bash
bash scripts/phase0.sh m4
```

自动完成三种 Shape 的两轮独立重编译与测量（每轮 6 个 A×B cell + 1 个 C++ 参考，共 21 个候选），校验原默认代码等价，导出交互与来源样例报告。设计、缓存规模和参考的适用边界见 [M4_PROTOCOL.md](M4_PROTOCOL.md)。可用 `CLANGXX` 覆盖 clang++，`LLVM_OBJDUMP` 覆盖反汇编工具。

已验证的单独报告重建命令（不用重新测量）：

```bash
bash scripts/phase0.sh m4-report --records logs/20261018T101246162932Z/records.jsonl logs/20261018T101331335316Z/records.jsonl --workflow logs/m4-20261018T101246123107Z/workflow.json
```

`m4-interaction.json` 和两份自动生成的 Markdown 为派生结果；人工解读单独保存在 `experiments/reports/m4-findings.md`，重建统计报告不会覆盖它。

## M5-A 只读数据分析

```bash
bash scripts/phase0.sh m5-a
```

只读取仓库内已提交的 M4 JSON，无需 MLIR 或重新测量；输出 `experiments/reports/m5-a.md` 与 `m5-a.json`。离线规则及成本限制见报告，不把 oracle 当作在线优化器。

## M5 新家族与固定预算对照

```bash
bash scripts/phase0.sh m5
```

使用 `experiments/m5.json`，每批六十条受控候选，共两批；先完成全部正确性与固定次数计时，再生成 M5-A 与 M5-B/C/D 报告。规则、信息边界、严格调用预算和墙钟费用的区别见 [M5_PROTOCOL.md](M5_PROTOCOL.md)。预览完整工作流已用两类共十二条候选的临时配置跑通，两批均通过；旧 M1 的自适应校准入口回归仍通过。

正式完整单入口也已通过，两轮日志路径如下。仅重建统计/成本/来源报告可执行：

```bash
bash scripts/phase0.sh m5-report --records logs/20261018T151218131929Z/records.jsonl logs/20261018T151325422312Z/records.jsonl
```

`m5-findings.md`、`m5-results.json`、`m5-reduction-lineage.md` 为派生结果；`m5-interpretation.md` 是人工解读，不被重建命令覆盖。策略选择/成本/统计由原始数据确定；Python 离线记账耗时会在每次报告运行中重新测量，因此该字段允许变化。

## M6.0 只读固定覆盖与随机参考

```bash
bash scripts/phase0.sh m6-a
```

只读取已提交 M5 JSON，输出 m6-coverage.md/json，不重跑基准或修改 M5 实验。穷举十五种四候选子集，并对 k=2/4/6 运行一百个种子的离线随机/分层随机参考。

## M6 扩展池与透明模型 pilot

```bash
bash scripts/phase0.sh m6
```

单入口按冻结的 [M6 协议](M6_PROTOCOL.md) 完成两轮独立编译、校验、固定次数计时，随后仅使用 train/validation 拟合并保存 ridge 模型，再做 blind 离线策略回放和报告。12条候选/实例，20个输入，其中逐元素融合链整模板留出。线程环境在包装入口统一设为1。配置、随机种子、20个透明特征、反馈分配规则和神经阶段门槛在正式采集前固定。

`m6-report --records <第一轮records.jsonl> <第二轮records.jsonl>` 可只重建报告；模型训练和离线推理计时时钟会变化，权重与策略选择应一致。旧 M5 统计不被重建。完整链路功能试跑使用 tmp/ 下的独立配置，输出到 tmp/m6-preview，未接触正式 blind 标签。

正式单入口已完成，两批各240/240通过。已验证从以下原始记录重建结果；函数级重建预览在 tmp/m6-rebuilt，权重、全部种子分数数组、选路与费用相同：

```bash
bash scripts/phase0.sh m6-report --records logs/20261018T171911048006Z/records.jsonl logs/20261018T172045377274Z/records.jsonl
```

派生报告为 m6-findings.md / m6-results.json / m6-model.json / m6-fusion-lineage.md；人工 m6-interpretation.md 不被覆盖。完整成本、短批测量噪声、模型外推失败和停止神经阶段的原因见人工解读。
