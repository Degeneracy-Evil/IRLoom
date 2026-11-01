# LoomTrace V1：一次有界的大程序接入尝试

2026-11-01，基线 `c5e1da2`。按 [LoomTrace V1 第5节](../../docs/architecture/LOOMTRACE_V1_PLAN.md) 独立尝试一个更大完整程序，不阻塞 H1–H3。结论为 **固定 Polygeist 前端导入不支持，未获得有效图或可执行产物**。没有把该程序加入轨迹训练/验证，没有依据速度更换程序。

选定现有 pinned LLVM test-suite 的 `MultiSource/Benchmarks/MiBench/automotive-susan/susan.c`。这是完整单 TU 的 SUSAN 图像处理程序，含 main、图像读写、平滑、边缘和角点检测等多函数/多区域代码，2125 行、59824 字节，明显大于既有 71–607 行源码；没有复制小算子或删减原功能。

来源为 LLVM test-suite commit `72a946a42cbce06afef68b2326ff6c3869d09799` 的 [原文件](https://github.com/llvm/llvm-test-suite/blob/72a946a42cbce06afef68b2326ff6c3869d09799/MultiSource/Benchmarks/MiBench/automotive-susan/susan.c)，复用既有 `llvm-test-suite-18.1.3` 缓存，未下载新依赖。原头部和同目录 LICENSE 记载 Crown Copyright、research purposes only、保留原头部且不可出售；源码与许可证均留原缓存，不提交复制品。原 CMake 的运行路径是 `input_large.pgm /dev/null -s`；本次预登记将输出改为项目缓存中的 PGM 文件，以便比较完整文件字节，同时登记同目录 `input_small.pgm` 作正确性检查。未改算法、输入图像或浮点门槛，也未使用官方 CMake 的输出 hash 机制。

实际调用匹配 SDK 的 `cgeist`，原前端参数 `-S -O0 -raise-scf-to-affine -std=gnu89 -resource-dir=/usr/lib/llvm-18/lib/clang/18`。Session 显式设置 `cleanup-positional-v1`，预期 bridge 为同版本私有修复产物；但导入阶段先失败，**没有实际调用 bridge 或执行变换**，因此不声称该修复已覆盖此大程序。

```sh
PYTHONPATH="$PWD" bash scripts/architecture-python.sh -m irloom.neural.workload_probe --name loomtrace-v1-susan
```

独立入口复用 Session、检查点、结构定位、lowering 和完整输出函数，不改主线 recipe/模型。完整同名失败结果缓存复用；使用新名字会重现同一固定程序，最多 default 和 first-joint 两个终端。本次只做一次正式源码导入，未继续尝试其它参数或另选程序。后续分支代码用于导入支持时的完整 PGM/stdout 校验，本次没有执行，不能据此声称已通过。

实际失败命令：

```sh
benchmark-data/architecture/frontends/build/bin/cgeist \
  benchmark-data/sources/llvm-test-suite-18.1.3/MultiSource/Benchmarks/MiBench/automotive-susan/susan.c \
  -S -O0 -raise-scf-to-affine -std=gnu89 \
  -resource-dir=/usr/lib/llvm-18/lib/clang/18 \
  -o benchmark-data/architecture/runs/loomtrace-v1-susan-root/imported.mlir
```

`cgeist` 返回 5，verifier 在 `susan.c:2059:7` 报告 `func.call` 第2号 operand 期望 `f32`、实际 `f64`，调用目标为 `susan_smoothing`。源码该函数使用旧式 K&R 参数声明，`dt` 为 float，调用处传入 `dt`；错误与导入时的默认参数提升/函数签名类型处理一致。这里仅定位到具体导入类型冲突，没有进一步修复或证明前端内部的完整根因。未关闭 verifier、注入 prototype、改源声明、修改 flags 或只取某个子函数绕过。

| 检查项 | 实际结果 |
| --- | --- |
| 固定源码尝试 | 1 |
| 成功完整轨迹 / 链接终端 | 0 / 0 |
| 有效图节点/函数/区域规模 | 未获得，不能从被拒的 IR dump 冒充有效图 |
| S0 inline / S1 unroll2 合法机会 | 未到合法观察，未知 |
| 完整 PGM/stdout 正确性 | 未执行，无结论 |
| 不同最终实现 / 性能测量 | 未获得 / 0 次 |
| 已记录前端命令时间 | 1.219150 秒（版本查询及导入） |
| probe wall | 1.352669 秒（含环境与错误记录） |

证据在 `benchmark-data/architecture/workload-probes/loomtrace-v1-susan/{protocol,result}.json`，原错误和被拒 IR dump 在 `logs/architecture/loomtrace-v1-susan-root/001-import.log`。所有临时/大文件在现有项目缓存，未计算 hash、未新建测试工程或重建 SDK。该接入负结果保留，主线训练继续；本轮不再硬攻前端或据此自动启动新的 workload 搜索。
