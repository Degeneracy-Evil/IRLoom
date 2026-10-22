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
- N4/N5 尚未完成：当前权重仅经过小图集成拟合，没有按真实完整程序族划分的训练/留出评价、共同有界搜索或容量匹配正式结论。

神经入口（路径相对仓库；示例使用绝对路径以避免工作目录歧义）：

```sh
bash scripts/architecture.sh --model A --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-a-fit
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt"
bash scripts/architecture.sh --model A --weights "$PWD/benchmark-data/architecture/runs/my-a-fit/weights.pt" --no-history
bash scripts/architecture.sh --model B --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json" --name my-b-fit
bash scripts/architecture.sh --model B --weights "$PWD/benchmark-data/architecture/runs/my-b-fit/weights.pt" --state-audit --multi-function --incremental
```

每次神经运行保存 `weights.pt`、`model.json`、各阶段候选分数与实际 transition。图注意力只计算结构边、反向边、自环和线性数量的摘要边，没有构造 N×N attention 矩阵。所有节点都编码；超预算时拒绝，不静默截断程序。

## 真实程序导入

已查询官方 Polygeist 仓库，固定 `77c04bb2a7a2406ca9480bcc9e729b07d2c8d077`，其 LLVM 子模块为 `26eb4285b56edd8c897642078d91f16ff0fd3472`（LLVM 18.0.0git）。官方没有发布预编译 release。该开发版不能直接当作 MLIR 18.1.3 SDK 的匹配前端；独立源码接入正在准备。当前尚无完整 C/C++ 程序经过此 V1 入口，不把旧原生基线计为 A/B 结果。

`bash scripts/prepare-polygeist.sh --build` 下载固定官方 LLVM 归档，只解包 llvm/clang/mlir/cmake/third-party；仅构建 X86 的 cgeist、mlir-opt、mlir-translate，默认 8 个任务，可用 `IRLOOM_FRONTEND_JOBS` 调整。禁用 CUDA/ROCm/Polymer、测试及 benchmark 构建，不安装到系统。许可证均为 Apache-2.0 WITH LLVM-exception，原始许可证保留在源码缓存。该构建比 bridge 大得多，首次需要时间；后续复用归档和对象文件。构建日志为 `logs/architecture/polygeist-build.log`。
