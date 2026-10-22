# V1 架构运行记录

本目录的新入口独立于历史 RealBench/PGO 工具。依赖下载、SDK、构建、IR、权重和原始报告放在 `benchmark-data/architecture/`（本机链接到 `/mnt/data/chp/irloom-benchmarks`），命令日志在 `logs/architecture/`，临时文件在 `tmp/`。源码与锁文件提交，产物不提交。

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

入口幂等准备依赖与增量构建；每次运行自动创建独立目录。可用 `--name NAME` 指定新运行名，已有目录会拒绝覆盖。阶段 manifest 是 `compiler/pipelines/architecture-v1.json`。S0 在 tensor/linalg 语义尚存时决策，S1 是 bufferization 后的 affine 循环，S2 是 scf/memref 观察点。S1 首版拟支持定点展开，不宣称已经支持 tiling/vectorization。

每个阶段保存 IR 与结构图；固定后端为 Clang `-O3 -march=native -ffp-contract=off`，无 LTO/PGO。复用历史 C 数值运行器以保持独立 oracle；新增编译器接口为 C++。3 个输入种子逐元素比较、检查 NaN 和缓冲区哨兵；5 轮计时，每轮 2000 次调用，固定一个允许的 CPU。未控制频率和邻居负载，小图计时只验证管线，不是科学性能结论。

## 阶段进展

- N0：实现三个暂停/继续边界、数据记录契约、C++ 结构图及固定默认路径。`n0-default-2` 通过 3 个种子的独立数值检查和 5 轮测量（中位数 0.000978 ms/调用；仅作 smoke）。初次运行发现 LLVM JSON 对短生命周期 StringRef 的借用，改为持有字符串副本后通过。
- N1：`n1-audit` 在 S0 执行目标融合、S1 执行目标循环展开，均实际改变 IR；输出通过 3 个种子的独立检查和测量。非法目标被拒绝；在真实变换之后注入失败，父 IR 不变；恢复检查点重放得到逐字节相同的 IR（不计算 hash）。失败也扣预算。动作内用临时来源属性记录受控 clone/merge 的操作级对应；属性在落盘前移除，值/块及一般 lowering 细节保持 unknown。下一阶段重新枚举候选，旧目标不能直接复用。
- N2：A 的三个独立稀疏 Transformer（128 hidden、4 heads）、共享结构编码/动作头、可微锚点对齐已进入真实编译。`n2-a-fit-2` 选择 fuse/unroll2/advance 并通过数值检查；`n2-a-replay` 的动作和浮点分数完全一致。参数量 484740；25 步单轨迹集成拟合 loss 10.4421 → 0.8327，单独由末端损失计算的首阶段 hidden 梯度 L1=3.69435。该拟合主要是实际动作行为克隆，不能视为 N4 训练或泛化结果。`--no-history` 可关闭连续隐藏连接，仍保留所有方法共同可见的离散事件。
- N3：B 使用共享稀疏图 Block 和 GRU，显式存储 module/function/region 记录，未知来源区域刷新，删除记录不会残留。`n3-b-fit` 的 25 步集成拟合 loss 3.9547 → 0.01454，末端到首阶段 hidden 梯度 L1=0.26743；319108 参数。`n3-b-state` 的两条多函数编译分支分别融合/保留，交错继续后均数值正确，检查点恢复重放一致，独立张量副本验证通过。`n3-regression-reduce` 进一步验证具有共享生产者和 reduction 的多函数图，融合/展开、错误恢复与数值检查通过；`n3-b-final` 在补充结构属性特征后再次通过拟合、分支交错、RNG 恢复和完整编译。
- B 的 `--incremental` 仅缓存已知且输入完全未变的基础节点编码；未知节点仍重编码，图提取和稀疏 attention 仍完整执行。验收逐输出比较完整/增量结果（rtol 1e-5、atol 1e-6），单独保存两者耗时、复用节点和失效记录数，不宣称已实现完整增量编译器或性能优势。
- N4：`pilot-v1` 按 LLU=train、PC1=validation、trie=test 采集完整程序轨迹，前两族各 default/balanced/两个固定随机路径，测试族在训练前仅采默认基线。A/B/current-only/B-capacity 各固定 20 epoch，使用实测 continuation 相对 log latency、pairwise 排序与资源辅助损失，按验证 loss 选择权重。测试族没有参与训练或 epoch 选择。7 个单路径方法（含断连/重置消融、原配与容量匹配）均完成编译和独立输出检查。`n4-replay-A` 从冻结权重重放留出程序，动作与分数完全一致。
- 本轮 A/B 参数为 484868/319236；B hidden=160 为 486084，与 A 相差 0.251%。训练 loss 首末分别 A 4.6122→0.5765、B 1.7689→0.3498，训练耗时约 14 秒/模型；验证 loss 后期上升，不能将训练 loss 下降当作泛化证据。LLVM C 前端新增 inline 动作后比 N2/N3 多一行 action embedding；旧三动作权重加载时保留原行，新行按记录种子初始化。
- N5：共同 beam 已实现，`n5-first-beam` 对 PC1 验证 8 次动作试执行、2 个正确终端；完整对照运行待完成。共享预算 beam=2、最多 8 个终端、24 次动作尝试、180 秒总期限；子进程 timeout 按剩余期限收紧。失败不退预算，失败变换、lowering 和错误终端不参加排名。

神经入口（路径相对仓库；示例使用绝对路径以避免工作目录歧义）：

```sh
bash scripts/architecture.sh --model A --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-a-fit
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt"
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt" --no-history
bash scripts/architecture.sh --model B --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-b-fit
bash scripts/architecture.sh --model B --weights "$PWD/benchmark-data/architecture/runs/my-b-fit/weights.pt" --state-audit --multi-function --incremental
bash scripts/architecture.sh --pilot --name my-pilot
bash scripts/architecture.sh --compare-pilot "$PWD/benchmark-data/architecture/pilots/my-pilot" --name my-comparison
```

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

上述三个完整原始程序均已通过独立 Clang 输出比较和测量：`import-pc1-default`、`import-llu-default`、`import-trie-default-2`。PC1 的 `import-pc1-audit` 还通过定点内联、循环展开、非法动作拒绝和失败恢复。该适配器明确没有 tensor/linalg 层，S0 是有合法性接口检查的调用内联，S1 是 affine 展开（包含有 cleanup 的动态界），S2 观察后固定 lowering。作用于完整源程序，不抽取或复制 kernel；这些仍是规模有限的真实测试程序，不能代表 SQLite/Clang 的全项目覆盖。

trie 的 empty-index LLVM GEP 暴露了固定开发版 MLIR 的自定义打印/解析不一致；末端改用 `--mlir-print-op-generic` 后可往返并通过原始程序输出检查，没有文本替换 IR。各阶段使用真实结构 API；完整程序计时为进程 wall time，包含启动、捕获和日志开销，特别小的程序受此开销影响。

保留的失败记录：RC4 的部分二维数组初始化触发 `ValueCategory::store` 断言，两种其他 ABI 选项也失败；bmm 同类数组断言；anagram 在第 347 行出现 `arith.shli` 类型不一致；sim 拒绝旧式局部声明；dijkstra 触发 IntegerType cast 断言；MD5 默认输出 `ca42…` 与独立 Clang 的 `acb2…` 不一致，排除。均未关闭 verifier 或更改源程序来冒充支持。许可证、来源、输入和程序族分配记录于 `experiments/architecture/programs.json`；正式 pilot 配置在采集神经轨迹前锁定于 `pilot.json`，不支持的导入会明确排除。

工程问题：第一次构建期间编辑正在执行的 shell 脚本，导致其末尾解析失败并覆盖当时的进度日志；工具构建产物保留，随后固定脚本并完整重跑增量构建成功。以后不在脚本运行期间修改它。CMake 的 Python 工具通过项目 uv 包装器调用。
