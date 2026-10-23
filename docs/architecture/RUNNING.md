# V1 架构运行记录

## 当前入口：V1.3.1训练验收

```bash
bash scripts/architecture.sh --training-review all --name my-v131
```

读取既有 `benchmark-data/architecture/pilots/v13-stage/dataset.json` 及其引用产物；不重新采标签。配置是 `experiments/architecture/training-v1-3-1.json`，`--training-config "$PWD/path.json"` 可指定同协议配置副本；旧权重隔离审计固定seed23/20步/V1.3数据，不是任意训练接口。`audit/train/all` 分别执行数值审计、六模型训练、训练及少量真实编译重放；`all` 包含主权重/快照验收。新数值夹具也由原 `--objective-audit` 入口执行。与旧入口共用工具链，自动增量准备前端。

主checkpoint固定20次Adam更新，保存0/1/5/10/20步。A/B先用旧有限margin隔离选模，再替换共享根margin，最后独立训练无连续记忆版本。若A在既有PC1训练组仍未正确排序全部强对，自动执行唯一的200步单组诊断，诊断不读取验证/测试，也不部署。没有新标签、测试选模或历史搜索大矩阵。

结果位于 `benchmark-data/architecture/training-reviews/NAME`：`data-audit.json`、各模型快照/曲线、`training.json`、`acceptance.json`、`replay.json`。重放每模型仅PC1 S0和matrix第三个S1前缀，在已标注候选中选择，保留完整候选proposal以说明范围；五次原生计时只是正确执行的诊断，不做新加速结论。日志在 `logs/architecture`。同名完整结果幂等复用，不用缓存命中耗时覆盖原成本；中断的训练使用新名字。

验收与限制见 [V1.3.1报告](../../experiments/reports/architecture-v1-3-training-review.md)。以下V1.3及更早入口作为历史保留。

## V1.3历史入口

V1.3 的内容摘要、角色读出和有限 margin 对照见 [新报告](../../experiments/reports/architecture-v1-3-pilot.md)。一条命令运行新配方：

```sh
bash scripts/architecture.sh --v13 all --name my-v13
```

结构与数值验收分别为 `--context-audit`、`--objective-audit`，均用同一包装脚本；后者的真实数据检查读取既有 `v12-stage` 缓存。它们不是性能 benchmark。可用 `--v13 collect/train/evaluate/measure --name my-v13` 分阶段检查，或 `--v13-config "$PWD/...json"` 指定独立配方。默认配置为 `experiments/architecture/pilot-v1-3.json`，完整缓存同名复用，新试验使用新名字。

hierarchical-v3 的顺序为局部 typed 图、自底向上内容汇总、call/module 交互、原 A/B、角色读出。每阶段共享摘要，summary/candidate-read 是 encoding/network 中的子计时，不能重复求和。structured-v2、旧 softplus 和纯区间目标均保留。新标签按动作类型/函数/循环深度/callee 的固定种子分层，最多25个产物/程序；标签冻结后六轮配对交错根默认与候选。PC1/lists 训练、matrix 验证、Perlin 为唯一新测试族；DES/Puzzle 仅诊断。

```sh
bash scripts/architecture.sh --context-audit --name my-v13-context
bash scripts/architecture.sh --objective-audit --name my-v13-objective
bash scripts/architecture.sh --model B --encoder-version hierarchical-v3 --state-audit --multi-function
bash scripts/architecture.sh --program perlin --model A --weights "$PWD/benchmark-data/architecture/pilots/my-v13/A-s23.pt"
```

Perlin 使用上游已有 SMALL_PROBLEM_SIZE 宏，不改算法；结构可分辨不等于已学会性能，实际监督与验证限制见报告。下文的 V1/V1.1/V1.2 记录保留供追溯。

V1.1 当前修订见 [新报告](../../experiments/reports/architecture-v1-1-pilot.md)。`--encoder-version structured-v2` 为新 CLI 的默认表示，保留 `legacy-v1` 开关；旧 checkpoint 未声明版本时继续使用旧表示。`bash scripts/architecture.sh --structure-audit` 执行真实 MLIR 语义结构夹具和来源使用审计。以下 V1 pilot 记录原样保留。

本目录的新入口独立于历史 RealBench/PGO 工具。依赖下载、SDK、构建、IR、权重和原始报告放在 `benchmark-data/architecture/`（本机链接到 `/mnt/data/chp/irloom-benchmarks`），命令日志在 `logs/architecture/`，临时文件在 `tmp/`。源码与锁文件提交，产物不提交。N0–N5 首轮已完成；正式统计、成本与限制见 [结果报告](../../experiments/reports/architecture-v1-pilot.md)。

## 依赖与来源

Ubuntu 官方 `archive.ubuntu.com/ubuntu/pool/universe/l/llvm-toolchain-18/` 的 `libmlir-18`、`libmlir-18-dev`、`mlir-18-tools`，精确包版本 `18.1.3-1ubuntu1`，仅用 `dpkg-deb -x` 解包到缓存，不安装系统包。与本机 LLVM/Clang SDK `1:18.1.3-1ubuntu1` 匹配；许可证为 Apache-2.0 WITH LLVM-exception，发行包包含版权清单。需要现有 `clang++-18`、LLVM 18.1.3 开发包、CMake、Make、curl、dpkg-deb、uv。脚本以 2 个构建任务编译单个 bridge，不构建 LLVM 全树。

PyTorch 使用官方 CPU wheel 源 `https://download.pytorch.org/whl/cpu`，BSD-3-Clause，实际依赖由 `experiments/architecture/uv.lock` 锁定（本次 torch 2.14.1+cpu）。独立 uv 环境不会改变历史实验环境。完整硬件、编译命令、版本与工作树状态写入每次运行的 `environment.json`。

## 最短入口

```sh
bash scripts/architecture.sh
bash scripts/architecture.sh --context transpose --size 32
bash scripts/architecture.sh --context row_reduce --size 32
bash scripts/architecture.sh --audit
```

入口幂等准备依赖与增量构建；每次运行自动创建独立目录。可用 `--name NAME` 指定新运行名，已有目录会拒绝覆盖。阶段 manifest 是 `compiler/pipelines/architecture-v1.json`。S0 在 tensor/linalg 语义尚存时决策，S1 是 bufferization 后的 affine 循环，S2 是 scf/memref 观察点。S1 支持定点二倍展开，不宣称已经支持 tiling/vectorization。

每个阶段保存 IR 与结构图；固定后端为 Clang `-O3 -march=native -ffp-contract=off`，无 LTO/PGO。复用历史 C 数值运行器以保持独立 oracle；新增编译器接口为 C++。3 个输入种子逐元素比较、检查 NaN 和缓冲区哨兵；5 轮计时，每轮 2000 次调用，固定一个允许的 CPU。未控制频率和邻居负载，小图计时只验证管线，不是科学性能结论。

## 阶段进展

- N0：实现三个暂停/继续边界、数据记录契约、C++ 结构图及固定默认路径。`n0-default-2` 通过 3 个种子的独立数值检查和 5 轮测量（中位数 0.000978 ms/调用；仅作 smoke）。初次运行发现 LLVM JSON 对短生命周期 StringRef 的借用，改为持有字符串副本后通过。
- N1：`n1-audit` 在 S0 执行目标融合、S1 执行目标循环展开，均实际改变 IR；输出通过 3 个种子的独立检查和测量。非法目标被拒绝；在真实变换之后注入失败，父 IR 不变；恢复检查点重放得到逐字节相同的 IR（不计算 hash）。失败也扣预算。动作内用临时来源属性记录受控 clone/merge 的操作级对应；属性在落盘前移除，值/块及一般 lowering 细节保持 unknown。下一阶段重新枚举候选，旧目标不能直接复用。
- N2：A 的三个独立稀疏 Transformer（128 hidden、4 heads）、共享结构编码/动作头、可微锚点对齐已进入真实编译。`n2-a-fit-2` 选择 fuse/unroll2/advance 并通过数值检查；`n2-a-replay` 的动作和浮点分数完全一致。参数量 484740；25 步单轨迹集成拟合 loss 10.4421 → 0.8327，单独由末端损失计算的首阶段 hidden 梯度 L1=3.69435。该拟合主要是实际动作行为克隆，不能视为 N4 训练或泛化结果。`--no-history` 可关闭连续隐藏连接，仍保留所有方法共同可见的离散事件。
- N3：B 使用共享稀疏图 Block 和 GRU，显式存储 module/function/region 记录，未知来源区域刷新，删除记录不会残留。`n3-b-fit` 的 25 步集成拟合 loss 3.9547 → 0.01454，末端到首阶段 hidden 梯度 L1=0.26743；319108 参数。`n3-b-state` 的两条多函数编译分支分别融合/保留，交错继续后均数值正确，检查点恢复重放一致，独立张量副本验证通过。`n3-regression-reduce` 进一步验证具有共享生产者和 reduction 的多函数图，融合/展开、错误恢复与数值检查通过；`n3-b-final` 在补充结构属性特征后再次通过拟合、分支交错、RNG 恢复和完整编译。
- B 的 `--incremental` 仅缓存已知且输入完全未变的基础节点编码；未知节点仍重编码，图提取和稀疏 attention 仍完整执行。验收逐输出比较完整/增量结果（rtol 1e-5、atol 1e-6），单独保存两者耗时、复用节点和失效记录数，不宣称已实现完整增量编译器或性能优势。
- N4：`pilot-v2` 按 LLU=train、PC1=validation、trie=test 重新采集完整程序轨迹，前两族各 default/balanced/两个固定随机路径，测试族在训练前仅采默认基线。A/B/current-only/B-capacity 各固定 20 epoch，使用实测 continuation 相对 log latency、pairwise 排序与资源辅助损失，按验证 loss 选择权重。测试族没有参与训练或 epoch 选择。7 个单路径方法（含断连/重置消融、原配与容量匹配）均完成编译和独立输出检查。
- 本轮 A/B 参数为 484868/319236；B hidden=160 为 486084，与 A 相差 0.251%。v2 训练 loss 首末分别 A 4.8971→0.5774、B 1.6980→0.3444；四套训练合计 57.117 秒，标签采集 126.669 秒。验证 loss 后期上升，不能将训练 loss 下降当作泛化证据。LLVM C 前端新增 inline 动作后比 N2/N3 多一行 action embedding；旧三动作权重加载时保留原行，新行按记录种子初始化。
- N5：`comparison-v2` 完成所有方法/程序的共同 beam；共享预算 beam=2、最多 8 个终端、24 次动作尝试、180 秒总期限，子进程 timeout 按剩余期限收紧。实际各组 2 个正确终端、5 或 8 次尝试；失败不退预算，失败变换、lowering 和错误终端不参加排名。搜索选优后另做 5 次确认测量，最终使用 `report-confirmed.json`，包含确认成本的最大累计工作时间 90.241 秒。本次确认在搜索批次结束后执行；累计工作不含中间人工处理空闲。统一入口现已直接集成即时确认。
- 最终验收：`final-state-audit` 验证成功动作使旧观察失效、拒绝旧目标并扣预算，来源账本随检查点保存/恢复，分支、RNG 和完整/增量推理检查仍通过。`final-replay-A`、`final-replay-B`、`final-profile-B-capacity`、`final-profile-current` 在新进程重放 v2 权重，动作和分数与原记录精确一致；冷加载和五次相同 S0 输入的稳态推理另存 `model.json`。`final-beam-integrated` 用冻结 A 对 PC1 完成 8 次尝试、2 个终端、15 次带计时调用（含 5 次即时确认），完整输出正确，总计 3.699 秒。
- 性能结论：A 在验证 PC1 上的约 1.094× 原生时间比在关闭连续连接后仍然存在；唯一留出族 trie 未显示收益。当前不能归因于跨阶段记忆，三程序几何均值不代表留出泛化。训练和留出族都缺少 S0 非 advance 动作覆盖；下一步扩展程序族、输入、动作覆盖和独立重复。

神经入口（路径相对仓库；示例使用绝对路径以避免工作目录歧义）：

```sh
bash scripts/architecture.sh --model A --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-a-fit
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt"
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt" --no-history
bash scripts/architecture.sh --model B --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-b-fit
bash scripts/architecture.sh --model B --weights "$PWD/benchmark-data/architecture/runs/my-b-fit/weights.pt" --state-audit --multi-function --incremental
bash scripts/architecture.sh --model B --state-audit --multi-function
bash scripts/architecture.sh --pilot --name my-pilot
bash scripts/architecture.sh --compare-pilot "$PWD/benchmark-data/architecture/pilots/my-pilot" --name my-comparison
```

复用已完成的 v2 模型（无需重新训练）：

```sh
bash scripts/architecture.sh --program trie --model A --weights "$PWD/benchmark-data/architecture/pilots/pilot-v2/A.pt" --profile-inference
bash scripts/architecture.sh --program trie --model B --weights "$PWD/benchmark-data/architecture/pilots/pilot-v2/B.pt"
```

`--pilot` 会采集、训练并执行冻结单路径评估，`--compare-pilot` 执行共用 beam 和独立确认。请为新一轮使用新名称；旧结果不覆盖。`current_only` 是单独训练的无连续隐藏历史基线，`A_no_history`/`B_reset` 为冻结模型消融；这些方法仍共享离散动作事件特征。

每次神经运行保存 `weights.pt`、`model.json`、各阶段候选分数与实际 transition。图注意力只计算结构边、反向边、自环和线性数量的摘要边，没有构造 N×N attention 矩阵。所有节点都编码；超预算时拒绝，不静默截断程序。

## 真实程序导入

官方 Polygeist 固定为 `77c04bb2a7a2406ca9480bcc9e729b07d2c8d077`，其 LLVM 子模块为 `26eb4285b56edd8c897642078d91f16ff0fd3472`（LLVM 18.0.0git）。cgeist、polygeist-opt、mlir-translate 和独立 bridge 已在同一源码构建中通过；没有与 18.1.3 SDK 混链。Clang 18.1.3 仍作各方法共同的固定 native 后端。原生小图与 C 程序分别使用两个 manifest。

`bash scripts/prepare-polygeist.sh --build` 下载固定官方 LLVM 归档，只解包 llvm/clang/mlir/cmake/third-party；仅构建 X86 的 cgeist、polygeist-opt、mlir-opt、mlir-translate 和 bridge，默认 8 个任务，可用 `IRLOOM_FRONTEND_JOBS` 调整。禁用 CUDA/ROCm/Polymer、测试及 benchmark 构建，不安装到系统。许可证均为 Apache-2.0 WITH LLVM-exception，原始许可证保留在源码缓存。该构建比 bridge 大得多，首次需要时间；后续复用归档和对象文件。构建日志为 `logs/architecture/polygeist-build.log`。统一入口发现前端缺失或 bridge 更新时也会自动调用此准备过程。

```sh
bash scripts/architecture.sh --program pc1
bash scripts/architecture.sh --program pc1 --audit
bash scripts/architecture.sh --program llu
bash scripts/architecture.sh --program trie
```

程序适配器复用既有 `benchmark-data/sources/llvm-test-suite-18.1.3` 官方源码缓存（提交 `72a946a42cbce06afef68b2326ff6c3869d09799`），不会再次下载；该缓存是完整程序入口的前置依赖，来源准备见 [真实程序指南](../REAL_BENCHMARKS.md)。

上述三个完整原始程序均已通过独立 Clang 输出比较和测量：`import-pc1-default`、`import-llu-default`、`import-trie-default-2`。PC1 的 `import-pc1-audit` 还通过定点内联、循环展开、非法动作拒绝和失败恢复。该适配器明确没有 tensor/linalg 层，S0 是有合法性接口检查的调用内联，S1 是 affine 展开（包含有 cleanup 的动态界），S2 观察后固定 lowering。作用于完整源程序，不抽取或复制 kernel；这些仍是规模有限的真实测试程序，不能代表 SQLite/Clang 的全项目覆盖。

trie 的 empty-index LLVM GEP 暴露了固定开发版 MLIR 的自定义打印/解析不一致；末端改用 `--mlir-print-op-generic` 后可往返并通过原始程序输出检查，没有文本替换 IR。各阶段使用真实结构 API。

测量修正：`pilot-v1` 的外层 subprocess wall time 混入了常驻 PyTorch 大进程的 fork/捕获开销，不能用于短程序性能结论。保留这些原始记录作诊断，正式结果改用 v2 计时并重新采集/训练。新增 C++ `irloom-time` 在独立轻量父进程中计量原始程序的 fork/exec/wait、输出和峰值 RSS；Python 驱动启动及日志仍计入总优化成本，另存 `driver_wall_ms`。`timer-calibration` 的 trie 原生中位时间约 0.924 ms，外层约 6.8 ms，说明修正是必要的。两者不可混作相同指标；原生启动/I/O 仍在完整程序时间内，短程序也仍受系统噪声影响。

保留的失败记录：RC4 的部分二维数组初始化触发 `ValueCategory::store` 断言，两种其他 ABI 选项也失败；bmm 同类数组断言；anagram 在第 347 行出现 `arith.shli` 类型不一致；sim 拒绝旧式局部声明；dijkstra 触发 IntegerType cast 断言；MD5 默认输出 `ca42…` 与独立 Clang 的 `acb2…` 不一致，排除。均未关闭 verifier 或更改源程序来冒充支持。许可证、来源、输入和程序族分配记录于 `experiments/architecture/programs.json`；正式 pilot 配置在采集神经轨迹前锁定于 `pilot.json`，不支持的导入会明确排除。

工程问题：第一次构建期间编辑正在执行的 shell 脚本，导致其末尾解析失败并覆盖当时的进度日志；工具构建产物保留，随后固定脚本并完整重跑增量构建成功。以后不在脚本运行期间修改它。CMake 的 Python 工具通过项目 uv 包装器调用。

## V1.1：组内训练与冻结交错对照

正式配方为 `pilot-v1-1-canonical.json`；先前未规范 DLTI 输入的版本保留作诊断，不能混入正式结果。规范化版本排除无序重复的 `dlti.dl_spec` 打印，保留规范 `llvm.data_layout`。

精确复现历史 V1 训练配方需使用原报告对应的 Git 提交；当前训练代码包含 V1.1 的分组、重复产物和 near-tie 修正。旧表示开关及冻结权重仍可用于当前输入上的推理和同新数据对照。

首轮 V1.1 使用独立配置和新输出名，保留旧结果。复现三步均通过同一包装入口自动准备依赖和增量构建：

```sh
bash scripts/architecture.sh --pilot --pilot-config "$PWD/experiments/architecture/pilot-v1-1-canonical.json" --name my-v11
bash scripts/architecture.sh --compare-pilot "$PWD/benchmark-data/architecture/pilots/my-v11" --pilot-config "$PWD/experiments/architecture/pilot-v1-1-canonical.json" --name my-v11-beam
bash scripts/architecture.sh --interleave-pilot "$PWD/benchmark-data/architecture/pilots/my-v11" --search-report "$PWD/benchmark-data/architecture/comparisons/my-v11-beam/report.json" --name my-v11-interleaved
```

三个阶段分别是采集/训练/冻结单路径、相同预算 beam/选优后确认、冻结产物交错比较。可省略 `--search-report` 只比较单路径。已有名字拒绝覆盖；成功后再次启动用新名字，失败细节保留在对应日志中。`--pilot-config` 使用绝对路径，因为 uv 运行目录为 `experiments/architecture`。

新训练按程序/输入/机器/编译语义分组，只在同组比较相对性能；组内先平均、程序间等权。强排序跳过相同 IR/目标文件和预定义 near-tie。`legacy_A/B` 在同一新数据上独立训练旧表示，`current_only`/`B_current_only` 也是独立训练；`A_no_history`/`B_reset` 是冻结权重测试时干预，不能混称为相同消融。所有方法仍看当前 IR 与离散动作事件。

最后一步使用已冻结的独立目录产物，加入原始源码 native Clang `-O3` 的可执行文件。每程序 10 轮，同一组随机顺序与反向顺序配对；每轮重新计量默认产物。记录原始样本、完整输出检查、顺序、每轮比值、直接字节比较的产物等价组，以及 IR/目标文件/可执行文件相对默认的差异标记，不计算 hash。分 train/validation/test 汇总，不使用早期标签采集时的 default 时间作为最终性能基准。

当前正式配方含六个完整原始程序、一个留出 backtracking 族（Queens/Puzzle），均为规模有限的测试程序。不同程序存在 Stanford suite 的共同历史来源；这不构成大型应用或大型 C++ 泛化验证。实际覆盖、near-tie/重复数量和尚无记忆增量收益的边界见 [V1.1 报告](../../experiments/reports/architecture-v1-1-pilot.md)。

统一入口另外写入 `logs/architecture/entry-*.json`：`started_ns` 到 `ready_ns` 是依赖准备与增量构建，`wall_ns` 是完整入口的总时间，`exit_code` 保留失败状态。原始批次只记录了 driver 内部工作，这个限制已写入报告；之后复现可把包装启动成本一并纳入。

## V1.2：阶段价值学习与共同默认保护

一条命令完成匹配标签采集、训练、冻结 raw/guarded 编译和交错测量：

```sh
bash scripts/architecture.sh --v12 all --name my-v12
```

同名、同协议再次执行复用已完成的 dataset、checkpoint、evaluation 和最终计时，不重新训练/下载，也不覆盖原始成本。新实验或改配方使用新名称；配置为 `experiments/architecture/pilot-v1-2.json`，可用 `--v12-config "$PWD/...json"` 明确指定。部分阶段失败时完整阶段的缓存仍保留，未完整写出的模型/运行会明确报错，不自动删除产物。

分阶段检查仍使用同一入口：

```sh
bash scripts/architecture.sh --v12 audit --name my-frozen-v11-audit
bash scripts/architecture.sh --v12 guard --program puzzle --weights "$PWD/benchmark-data/architecture/pilots/pilot-v1-1-canonical/A.pt" --name my-paid-guard
bash scripts/architecture.sh --v12 collect --name my-v12
bash scripts/architecture.sh --v12 train --name my-v12
bash scripts/architecture.sh --v12 evaluate --name my-v12
bash scripts/architecture.sh --v12 measure --name my-v12
```

`audit` 只读取本机已有冻结 V1.1 记录；其它阶段复用既有固定工具链和源码缓存。`collect` 在同一 Session 检查点分叉动作，实际完成并比较全程序；q 的完成策略固定为后续全 advance，用根默认延迟作统一归一化参照。q 输出没有概率/置信度含义。`raw` 评估直接提议；`guarded` 的 3 个完整候选席位包含 1 个计费默认席位，另外最多 2 个提议，18 次尝试、180 秒；失败和确认也计费。全部搜索使用相同合法动作和 near-tie 规则。

新模型文件位于 `benchmark-data/architecture/pilots/NAME/A-s23.pt` 等目录。可直接检查冻结模型的完整编译：

```sh
bash scripts/architecture.sh --program des --model A --weights "$PWD/benchmark-data/architecture/pilots/my-v12/A-s23.pt"
```

新完整 3DES 测试使用原源码 `1` 参数运行自带 Monte Carlo 正确性检查。Queens/Puzzle 只作回归诊断。模型选择若验证候选不可分辨，使用预先固定最后一轮，并明确记为证据不足。详细覆盖、控制器与记忆收益的区别、原始计时位置见 [V1.2 报告](../../experiments/reports/architecture-v1-2-pilot.md)。

若交错测量中断且已产生样本，保留旧样本，可用同一入口对冻结目录换一个测量名重测，例如 `bash scripts/architecture.sh --interleave-pilot "$PWD/benchmark-data/architecture/pilots/my-v12/frozen" --name my-v12-remeasure`；无需重新采集标签或训练。
