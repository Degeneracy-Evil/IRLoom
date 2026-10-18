# Phase 0 本机运行指南

实现只依赖 Python 标准库，通过 `uv` 运行。所有生成产物放在 `tmp/`，日志与原始 JSONL 放在 `logs/`；这些目录均被 Git 忽略。不执行系统安装，不计算哈希，不写单元测试。

## 当前验证环境与 M0 命令

2026-10-18：Ubuntu 24.04 amd64，Intel Xeon 6986P-C，LLVM / Clang / MLIR 18.1.3。系统已有 `/usr/lib/llvm-18/bin` 中的 LLVM 与 Clang，没有 MLIR。以下命令已实际执行：

```bash
bash scripts/phase0.sh setup
bash scripts/phase0.sh probe > logs/probe.json
bash scripts/phase0.sh
```

`setup` 将 Ubuntu 官方 `mlir-18-tools_18.1.3-1ubuntu1_amd64.deb` 解包至 `tmp/toolchain/`，幂等检查已有工具，不使用 sudo 或 apt install。系统配置的 apt 镜像下载返回 502，改用 archive.ubuntu.com 后成功。此包在当前已有 LLVM 18 的主机上可直接运行；不承诺适用于其他发行版。其他机器请提供匹配版本的 LLVM/MLIR，并用环境变量 `MLIR_OPT`、`MLIR_TRANSLATE`、`CLANG`、`LLC`、`LLVM_CONFIG` 指向工具。默认顺序为项目本地工具、PATH、系统 LLVM 18。

M0 读取 `benchmarks/matmul.mlir`，保存 `tmp/m0/{high,affine,transformed,scf,llvm_dialect}.mlir`、`kernel.ll`、`kernel.o` 和 `runner`。完整编译命令及输出见 `logs/m0/*.log`，元数据与计时记录见 `logs/m0/records.jsonl`。以本机 `mlir-opt --help` 核实 Pass，未使用假设性参数。

测量先独立校验所有输出元素（double 精度 C 参考实现，`rtol=atol=1e-4`，禁用 fast-math 与浮点收缩），固定到当前允许的第一个 CPU。3 次预热后按 2 的幂校准批量次数，保证校准批次至少 10 ms；记录 10 个批次的每次 kernel 延迟（ms）。计时不包含进程启动、输入分配、参考计算与校验，包含 kernel 自身输出初始化；重复访问同一组输入，属于缓存热态测量。编译耗时独立记录。CPU 频率、邻核负载不可控，PMU 未采集。

用户本轮指令优先于初始协议：**不计算任何 hash，不新增单元测试**。记录用快照路径、父路径、有序命令与 Git commit 建立关联，因此尚不满足原协议的密码学指纹要求。运行实际编译、数值正确性及计时作为必要验收。
