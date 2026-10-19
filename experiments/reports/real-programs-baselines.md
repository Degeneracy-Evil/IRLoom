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
