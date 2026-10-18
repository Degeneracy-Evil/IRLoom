# M4：两个真实决策点的受控交互

沿用 [PHASE0_REVIEW.md](PHASE0_REVIEW.md) 的规划与既有正确性、计时规范。只研究一个 matmul 程序家族；不增加模型、训练任务或新编译器 Pass。

## 因素与合法性

- A0：命名 linalg.matmul → generalize → identity interchange `[0,1,2]`，循环顺序 i-j-k。
- A1：同样 generalize → interchange `[0,2,1]`，循环顺序 i-k-j。
- B0：affine 层无额外变换。
- B1：`--affine-loop-tile=tile-size=16`。
- B2：`--affine-loop-unroll=unroll-factor=4`。

两份显式 Transform schedule 位于 `compiler/schedules/`，通过本机 MLIR 18 的 preload library + transform interpreter 执行。语法先参考 LLVM 18 官方的 [generalize](https://github.com/llvm/llvm-project/blob/release/18.x/mlir/test/Dialect/Linalg/transform-op-generalize.mlir) 与 [interchange](https://github.com/llvm/llvm-project/blob/release/18.x/mlir/test/Dialect/Linalg/transform-op-interchange.mlir) 测试，再在本机逐项编译执行验证。Interchange 必须作用于 generic，不能直接作用于命名 matmul。

每个 C[i,j] 的 k 递增累加顺序保持不变，不允许 fast-math、重关联或浮点收缩。B 的 unroll 始终针对当前最内层循环，因此 A0 下展开 k，A1 下展开 j；这是同一编译动作依赖当前 IR 的真实效果，不能误称为语义上同一个轴。

完整路径为 high → high_decision → affine（B 决策点）→ transformed → scf → LLVM dialect → LLVM IR → native。高层输入、schedule、高层变换后 IR、B 前后快照和有序命令分别保存。A0 与原 Phase 0 默认路径另编译并直接比较反汇编，避免 generalize 本身改变基线。

## Shape 与 Cache

配置为 `experiments/m4.json`，每个 Shape 测 2×3 个因子 cell，加一个独立 C++ 参考。

| M,N,K | A+B+C f32 工作集 | 本机容量关系 |
| --- | ---: | --- |
| 32,32,32 | 12 KiB | 小于 L1d 48 KiB |
| 127,65,93 | 约 102 KiB | 超过 L1d，小于 L2 2 MiB；含非整齐边界 |
| 1024,256,256 | 2304 KiB | 超过每核 L2 2 MiB，小于共享 L3 |

Cache 容量从所用逻辑 CPU 的 sysfs 元数据采集。工作集不含 double 参考、哨兵与代码；只用于声明容量跨界，不作为实际 miss 或带宽证据。无 cache flush，继续重复复用同一输入/输出；大尺寸属于稳态复用而非冷缓存流式测试。所有 kernel 每次重置 C，重置包含在延迟中。

## C++ 性能参考

`benchmarks/matmul_reference.cpp` 是普通 row-major i-k-j 三重循环，以 `clang++ -O3 -march=native -ffp-contract=off` 单独编译，然后与同一个 C 计时运行器链接。不使用 BLAS、restrict、额外线程、手工 SIMD 或不同数据布局。没有 LTO；MLIR 与 C++ 都调用同一 ABI、同一输入和同一输出校验。

它是较现实的常规实现参考，不是独立数值真值（真值仍由 double C 参考计算），也不代表成熟自动调优框架的最优性能。它不进入 A×B 的交互计算。C 与 C++ 共用 `benchmarks/memref.h`，复制到各编译产物目录，避免 ABI 定义漂移。

## 两批测量和交互

每批重新编译全部 21 个候选；固定输入 seed=1，正确性 seeds=0/1/7/19。分别用 order_seed=1、17 随机交错候选；改变测量次序不改变输入。每候选 15 个原生批次样本，3 次预热，每个 Shape 取所有候选校准次数的最大值以统一批量次数；编译、正确性、校准、采样和总墙钟分别留痕。

定义 `D = log[T(A1,B)/T(A1,B0)] - log[T(A0,B)/T(A0,B0)]`，T 是各 cell 的批次内中位延迟。按同一测量轮次联合重采样四个 cell，2000 次、seed=1，重算中位数与 D；报告探索性 95% 区间。只有两批 D 区间同向排除零，才记为本次数据中重复观察到交互；低层收益的符号反转还要求两个 A 的 log-effect 区间分别完全位于零两侧。

这不是严格独立机器实验：两批分别启动、重编译且次序不同，但同主机连续执行，仍有共享负载、频率、ASLR 和时间相关性，未做多重比较校正。保留原始样本、median/MAD/IQR/范围、编译耗时；不把狭窄区间当作通用结论。

## 单入口与验收

```bash
bash scripts/phase0.sh m4
```

单入口依次完成两批 collect、原默认路径代码对照和报告。每批日志独立写入 `logs/<timestamp>/`，产物写入 `tmp/<timestamp>/`；总流程与原默认验证命令另记在 `logs/m4-<timestamp>/workflow.json`。报告输出到 `experiments/reports/m4-interaction.md`、`m4-interaction.json` 和 `m4-lineage.md`。重复执行无需修复脚本，不删除旧产物；派生报告按当前两批结果重建。

实现预验收已完成 21/21 编译、小尺寸 7/7 的两批真实测量、三个 Shape 原默认反汇编相同及 M0 真实回归。正式数据必须从提交后的干净代码采集。

报告应区分：观察到路径交互；当前 affine IR 中是否仍显式包含循环/访问区别；历史是否有额外预测价值。只做前两者的证据分析，不声称已验证第三者。代表性 IR diff 与函数反汇编只提供定性解释，暂不开发模型或细粒度 Operation 追踪。继续遵守不计算 hash、不写单元测试的约束。
