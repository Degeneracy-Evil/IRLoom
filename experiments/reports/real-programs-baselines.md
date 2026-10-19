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

硬件为Intel Xeon 6986P-C，240个逻辑CPU、SMT2；Linux 6.8.0-55/glibc2.39，Clang/LLVM与本地LLD均为18.1.3。执行固定CPU0，sibling120未预留；默认构建jobs2，每个LLD内部并行2。可移植组为-O3 -DNDEBUG，native组增加-march=native，不启用fast-math；完整CPU/ISA/cache/工具版本与实际环境见JSON。

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

## R3：热点与调用决策审计完成

使用已有GNU gprof，未安装工具或修改perf权限；本机无perf命令，perf_event_paranoid=4。独立的-pg/-fno-omit-frame-pointer native构建验证后采样，另以PGO训练程序采集**仅供审计**的评测计数，绝不合入原训练profile。Zstd记录包含最终ThinLTO阶段；SQLite通过Make展开真实speedtest1参数，单独编译核心TU，避开shell/核心并行编译覆盖默认opt.yaml的问题。PGO核心诊断启用-Werror=profile-instr-out-of-date并通过，不能把shell的少量profile告警套用到speedtest1核心。

修正后接入日志20261019T171253028973Z，94.74秒，含SQLite三个核心诊断编译26.39/13.23/18.79秒。首次gprof程序构建成本分别为Zstd15.95秒、SQLite42.37秒（20261019T170927765410Z）。干净提交ba60f5e复跑20261019T171540350635Z，37.00秒；所有命令通过，源码/构建/训练/诊断记录缓存命中，无下载、无重新配置或核心诊断重编译。正式基线产物大小检查仍一致，根目录无gmon.out/default.profraw。原始调用图与抽样、评测计数及诊断记录留在logs/外部缓存，选取的调用图片段、具体决策和命令归档于[real-programs-audit.json](real-programs-audit.json)。

| 工作负载 | 干净复跑gprof主要自采样占比 | 动态调用证据 |
| --- | --- | --- |
| Zstd software-eval | decompressSequences 44.87%，compressBlock_doubleFast 38.90%，encodeSequences 5.25% | compressBlock_internal→buildSeqStore→doubleFast，两个调用边各6246次 |
| Zstd records-eval | decompressSequences 46.32%，compressBlock_doubleFast 36.82%，HUF fast asm loop 6.40% | 同一压缩调用链，每条边5439次 |
| SQLite main size200 | VdbeExec 25.99%，BtreeIndexMoveto 12.50%，BtreeTableMoveto 6.25% | VdbeExec→BtreeIndexMoveto 3325350次，callee总计4121076次 |

gprof每个样本0.01秒，百分比只是在可归属程序文本中的自采样比例，不能当作含libc/内核的完整应用时间；_init等归属不明项不作为优化目标。不同复跑百分比有数个百分点变化，主热点位置一致。Zstd内部按时间迭代，调用总数随机器速度/采样开销变化，不用绝对次数跨语料排名。gprof调用图的子时间还依赖传播估计；PGO的高频MEM_read/bit操作包括已内联函数，频度不等于真实调用开销。

### 候选1：Zstd热调用链的内联与全程序代码预算

可复核位置是固定1.5.7的lib/compress/zstd_compress.c:4399，compressBlock_internal调用buildSeqStore；后者经函数指针进入另一个TU中的doubleFast。动态调用图确认该链承担主要压缩工作，而非仅凭函数名称猜测热点。具体inline记录如下：

| 配置 | 阶段 | 决策 | Cost / Threshold | 训练Hotness |
| --- | --- | --- | --- | ---: |
| O3-native | 前端 | TooCostly | 625 / 625 | 无 |
| ThinLTO-native | 最终LTO | TooCostly | 660 / 625 | 无 |
| PGO+ThinLTO-native | 最终LTO | TooCostly | 630 / 625 | 4757 |

PGO/ThinLTO已知道热度并仍选择保留边界；这不是“编译器不知道跨函数信息”的证据。buildSeqStore自身几乎没有自采样，单纯省掉几千次call很可能无意义。值得核查的是：内联能否让调用方上下文进一步消除分支或约束压缩器分派，并在全程序代码预算下优于既有选择；反面可能是复制大的helper并恶化局部性。

**下一次最小A/B：** 在独立临时源码副本中只改变buildSeqStore的内联属性（基线、强制inline、显式noinline），固定PGO+ThinLTO及训练输入，对其三个源码调用位置逐一核查行为，记录源码diff。对两种现有评测语料分别做10次交错吞吐、逐字节往返和text大小比较；若profile CFG不匹配，则各变体用相同训练负载重新采集并计费。先观察常量传播/间接调用/代码复制是否实际改变，再谈全局策略。该A/B尚未执行；若仅增大代码、效果落入噪声，就关闭此候选。暂不把该问题推广为历史状态或神经模型的必要性。

### 候选2：SQLite opcode调用上下文与PGO训练覆盖

VdbeExec→BtreeIndexMoveto在main评测中有3325350次动态调用。精确speedtest1核心前端记录中，这类边在O3/native ThinLTO下的成本有640/625、625/625等，PGO下记录为65/45、Hotness=0。原源码src/vdbe.c的Seek与Found等不同opcode在多个位置调用同一callee；本轮记录未精确对应每个opcode调用点，不能把累计3325350次直接分配给某一条remark。

这里更直接的问题是训练orm,cte与评测main的覆盖差异：某些评测很忙的路径在训练中冷，现有PGO仍将其按冷路径处理。PGO整体已经改善main性能，并不意味着每条边都获益。SQLite核心又是一个amalgamation TU，编译器原本已可见大部分函数，当前没有充分证据支持“缺少跨模块可见性”或“需要比PGO更复杂的模型”。

**下一次最小A/B先审计覆盖：** 先从仅审计的评测profile/IR对应到具体opcode调用点；再在相同训练执行预算下，对比既有orm,cte训练与预先冻结、不同种子的索引查询训练，保持编译选项不变。使用新的预先留出SQL评测集判断覆盖改进是否稳定；当前main已被观察，只能作诊断pilot，不能再次当作未见泛化集。只有代表性训练后仍有明确残留调用决策，再考虑对单一callee做属性A/B。该候选目前是强基线的覆盖限制，不是已确认的编译器漏优化。

## 阶段边界与下一步

R1、R2、R3要求的正式负载、两个完整项目强基线、独立PGO闭环及至少两个程序的热点/调用证据已建立。现有数据支持上面的有限后续问题，**没有验证某个新全局优化能带来收益**；下一轮应先完成最小合法A/B和训练覆盖核查，不启动神经网络、MLIR扩展或全LLVM bootstrap。

实际全部缓存约5.6GiB，含开发阶段产物与诊断记录；本轮没有新下载，未删除旧缓存。源码/大型构建均经benchmark-data软链留在/mnt/data/chp/irloom-benchmarks，临时对象经./tmp，运行/剖析日志在./logs。复现完整流程：bash scripts/benchmarks.sh all --suite formal；仅重测用run，仅审计用audit。旧smoke入口、首轮报告及M0–M7/M8研究记录保持独立。

限制保留：单CPU亲和但未预留SMT sibling，频率、邻核负载、系统页缓存不受控；正式性能只有一批十轮，不是跨机器/生产数据泛化证据。SQLite只对小文件进行已知结果/完整性及跨变体dump校验，大输入依赖程序自身执行检查，未运行全量上游测试。PGO当前为Clang前端插桩，Zstd计时包含固定内置benchmark循环；LLVM/Clang仍是源码/组件能力检查，并未构建完整被优化编译器。两项MultiSource本轮只有O3参考，SQLite最终LTO决策未单独采集，审计不能越过这些证据边界。

最终兼容验收：旧smoke SQLite入口20261019T172041502021Z及formal单变体入口20261019T172109721568Z均通过，后者确认summary.md展示原生TOTAL及median/MAD。它们只是入口兼容检查，未加入上述140个正式样本。
