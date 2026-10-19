# RealBench Phase 1：正式负载与强基线

承接原始环境验收，执行docs/REAL_PROGRAM_NEXT.md。R1新增不重复的源码压缩语料、固定种子结构化记录、SQLite size200、官方Bullet和tramp3d-v4；旧smoke及报告保留。后续R2隔离强编译配置，R3调查有证据的热点/调用决策；不开发神经模型或新优化算法。

本报告按里程碑追加实际结果。运行命令与负载约束见docs/REAL_BENCHMARKS.md。

## R1：接入验收通过

完整入口20261019T164349181907Z（UTC，首次构建）通过5项负载，随后20261019T164632936166Z验证tramp3d的20进程批次。每项一次预热与一个测量样本，仅作功能/时间尺度验收；R2再做至少10次交错测量。

| 工作负载 | 进程墙钟秒 | 原生计算/输入规模 |
| --- | ---: | --- |
| Zstd software-eval | 4.226 | 约2.1MiB确定性源码tar，无重复填充；压缩/解压各约2秒 |
| Zstd records-eval | 4.359 | 约33MiB、300000条结构化记录；压缩/解压各约2秒 |
| SQLite main size200 | 4.530 | 内存数据库；文件模式只校验size10 |
| Bullet | 2.274 | 官方lit exec_time=2.1321秒 |
| tramp3d-v4 ×20 | 1.851 | 20个独立进程，官方参数与每个进程的参考校验保持，exec_time合计1.6304秒 |

首次构建命令耗时：Zstd9.40秒、SQLite34.52秒、两个MultiSource目标合计25.93秒，另计配置。没有新下载。规模审计：Zstd可执行含987个已定义文本/弱符号，程序前端11个TU/10216行（不包括静态链接压缩库）；SQLite amalgamation262899行、1951文本/弱符号；Bullet121个TU/40820行、2107符号；tramp3d1个TU/56089行、1210符号。符号数含编译器/运行时辅助函数，不能当作精确源码函数数。

失败记录20261019T164604787824Z：lit的%t自带.tmp后缀，与timeit多RUN文件命名不同；改为%T/生成test文件名-i.out后20项参考比较通过。未修改上游源码。机器可读记录：[real-programs-r1.json](real-programs-r1.json)。下一步R2隔离native、ThinLTO及PGO训练/评测，记录完整成本、代码大小与交错样本。

## R2：实现与成本归档规则

新增完整Zstd/SQLite的o3、o3-native、thinlto-native、pgo-thinlto-native，独立目录、固定输入/功能开关、全部正确性验证后交错采样。训练Zstd只读独立源码文件与不同种子记录；SQLite训练orm,cte，评估main。详细复现命令及隔离规则见运行指南。源码和大型产物继续留在外部缓存。

开发失败保留：20261019T164834444601Z在SQLite configure时因conftest main与训练main结构不同、将profile告警升级为错误而停止；20261019T165253099613Z在LLD18不支持等号式opt-remarks选项时停止。分别保留编译器诊断、改用分离参数后重跑。SQLite shell有额外功能宏且未参与训练，PGO构建可见shell.c无profile及少量函数体不匹配告警；不得将其描述为全shell覆盖的PGO。性能评价对象是speedtest1，shell只作数据正确性辅助。

## R2：十轮正式结果

干净提交2f1a57e，UTC日志20261019T165654773769Z；14个负载/配置组合，每组一次预热+10次独立样本，共140个测量样本，全部通过。完整工作流709.76秒，含PGO-use按profile批次新建目录的构建；不是缓存纯执行时间。

| 工作负载 | 编译变体 | 主指标 median ± MAD | 相对O3-native |
| --- | --- | --- | ---: |
| zstd-software-eval / compression_MB_s | o3 | 364.000 ± 0.200 MB/s | -2.43% |
| zstd-software-eval / decompression_MB_s | o3 | 1595.400 ± 1.700 MB/s | +0.95% |
| zstd-software-eval / compression_MB_s | o3-native | 373.050 ± 0.500 MB/s | +0.00% |
| zstd-software-eval / decompression_MB_s | o3-native | 1580.350 ± 1.000 MB/s | +0.00% |
| zstd-software-eval / compression_MB_s | thinlto-native | 374.450 ± 1.050 MB/s | +0.38% |
| zstd-software-eval / decompression_MB_s | thinlto-native | 1581.600 ± 1.100 MB/s | +0.08% |
| zstd-software-eval / compression_MB_s | pgo-thinlto-native | 375.150 ± 0.300 MB/s | +0.56% |
| zstd-software-eval / decompression_MB_s | pgo-thinlto-native | 1604.150 ± 1.650 MB/s | +1.51% |
| zstd-records-eval / compression_MB_s | o3 | 313.200 ± 1.350 MB/s | -2.75% |
| zstd-records-eval / decompression_MB_s | o3 | 1519.250 ± 4.000 MB/s | +0.55% |
| zstd-records-eval / compression_MB_s | o3-native | 322.050 ± 2.000 MB/s | +0.00% |
| zstd-records-eval / decompression_MB_s | o3-native | 1510.950 ± 10.300 MB/s | +0.00% |
| zstd-records-eval / compression_MB_s | thinlto-native | 322.600 ± 1.750 MB/s | +0.17% |
| zstd-records-eval / decompression_MB_s | thinlto-native | 1514.450 ± 3.300 MB/s | +0.23% |
| zstd-records-eval / compression_MB_s | pgo-thinlto-native | 323.100 ± 1.800 MB/s | +0.33% |
| zstd-records-eval / decompression_MB_s | pgo-thinlto-native | 1520.200 ± 0.950 MB/s | +0.61% |
| sqlite-main-memdb-size200 / compute_seconds | o3 | 4.526 ± 0.009 s | +2.95% |
| sqlite-main-memdb-size200 / compute_seconds | o3-native | 4.659 ± 0.062 s | +0.00% |
| sqlite-main-memdb-size200 / compute_seconds | thinlto-native | 4.599 ± 0.009 s | +1.32% |
| sqlite-main-memdb-size200 / compute_seconds | pgo-thinlto-native | 4.263 ± 0.008 s | +9.30% |
| bullet / exec_seconds | o3 | 2.127 ± 0.001 s | — |
| tramp3d-v4 / exec_seconds | o3 | 1.625 ± 0.001 s | — |

相对值为median之比：吞吐candidate/native−1，延迟native/candidate−1，正数表示该指标更好；不是成对显著性检验。Zstd固定时长内部循环的进程墙钟只作执行成本，不能用其墙钟比替代吞吐比。MultiSource为O3参考，未做变体加速对照。

| 完整程序 | 变体 | 文件字节 | llvm-size text | 本次增量构建秒 |
| --- | --- | ---: | ---: | ---: |
| zstd | o3 | 1063064 | 988617 | 0.072 |
| zstd | o3-native | 2145120 | 1036378 | 0.079 |
| zstd | thinlto-native | 2147056 | 1015695 | 0.075 |
| zstd | pgo-thinlto-native | 1502656 | 732077 | 15.820 |
| sqlite | o3 | 1787520 | 1654888 | 0.015 |
| sqlite | o3-native | 3527280 | 1698293 | 0.015 |
| sqlite | thinlto-native | 3644672 | 1712124 | 0.014 |
| sqlite | pgo-thinlto-native | 2737992 | 1392150 | 59.866 |

上述构建成本含configure/build命令，不含随后size/nm记录；已有对象缓存命中不能当作干净重建成本。首次观察的构建成本（不同开发步骤可能部分复用对象）及失败时间完整归档在JSON的cost_observations。llvm-size text包含只读数据等段，并非纯机器指令字节。

| PGO项目 | 本次训练缓存 | 原始训练+merge秒 | 训练profile |
| --- | --- | ---: | --- |
| zstd | True | 5.282 | /home/chen/develop/IRLoom/benchmark-data/builds/zstd/1.5.7/clang-18.1.3/formal-pgo-train/training-v1/20261019T165302470538Z/training.profdata |
| sqlite | True | 0.732 | /home/chen/develop/IRLoom/benchmark-data/builds/sqlite/3.50.4/clang-18.1.3/formal-pgo-train/training-v1/20261019T164834444601Z/training.profdata |

验收数据核查通过：全部命令exit0、140个测量样本、训练/评测源码文件集合无交集且记录种子不同、PGO-use目录与profile批次匹配、原始二进制大小匹配、官方参考PASS且无产物hash指标。没有新增单元测试。原始样本、环境、完整运行命令、编译/链接参数、profile和成本来源见[real-programs-baselines.json](real-programs-baselines.json)。

这批结果支持两点有限结论：native ISA并非所有指标都更好（Zstd解压、SQLite主负载有退化）；ThinLTO在已很强的O3-native上增量较小，SQLite的PGO+ThinLTO在固定未训练main输入上更快，但仍是单机器单批次证据。Zstd结构化记录的0.2%–0.6%差异与波动同量级，不作为稳健加速结论。PGO的llvm-size text比O3-native分别减少约29%（Zstd）和18%（SQLite）；文件字节还受优化记录关联的调试信息等影响，不可直接当作指令缓存占用。不能仅凭PGO结果把原因归结为某一个内联决策。
