# Zstd 单函数内联因果实验

固定协议见 [REAL_AB_PROTOCOL.md](../../docs/REAL_AB_PROTOCOL.md)。干净提交 `636b5b2` 执行两批各十轮，六种生产配置、两种语料共 240 个正式进程样本，24 次预热丢弃。日志 `logs/benchmarks/20261019T175257936558Z`；原始样本、配对比例、构建与训练成本见同名 JSON。全部命令成功，两种语料在六种配置均通过逐字节压缩往返。

固定官方 Zstd 1.5.7 / Clang、LLD 18.1.3，CPU 0。只在隔离源码副本修改 `ZSTD_buildSeqStore` 一处声明属性；三组各自重新训练，训练输入、级别和预算一致，评测输入未合并到训练。该 helper 与直接调用者同 TU，不能据此宣称跨模块优化。旧 R1–R3 数据和报告保持原样。

## 最终代码与成本

原始/noinline 在最终 zstd_compress 后端模块均保留三处调用，helper 符号为 ThinLTO 1941 字节、PGO+ThinLTO 1395 字节。always_inline 在源码 4363/4399/4512 三处均有 AlwaysInline 记录，最终程序无 helper 符号、该模块无 helper 调用；保存后端 IR、所选函数汇编与克隆符号。此属性实验同时改变三处调用，不能归因于单个调用点。

| 属性 | 构建 | 构建墙钟 s | text 字节 | 文件字节 |
| --- | --- | ---: | ---: | ---: |
| original | thinlto-native | 20.291 | 1015759 | 2153024 |
| original | pgo-train | 20.252 | 1164305 | 2677336 |
| original | pgo-thinlto-native | 15.905 | 731725 | 1505496 |
| always-inline | thinlto-native | 20.822 | 1019839 | 2162184 |
| always-inline | pgo-train | 20.586 | 1169593 | 2688536 |
| always-inline | pgo-thinlto-native | 15.813 | 734093 | 1511688 |
| noinline | thinlto-native | 20.585 | 1015759 | 2153088 |
| noinline | pgo-train | 20.423 | 1164305 | 2677336 |
| noinline | pgo-thinlto-native | 15.870 | 731725 | 1505560 |

首次三套构建均独立执行，包含生成器、编译与链接；本次正式测量复用缓存。每套 PGO 使用相同两项 `-b3 -i1` 训练工作；墙钟与逐命令 user/system CPU 均保留。

| 属性 | 训练及合并墙钟 s |
| --- | ---: |
| original | 4.277 |
| always-inline | 4.273 |
| noinline | 4.275 |

## 两批正式性能

表中为同轮配对吞吐比的中位数百分比；正数更快，括号为配对比例 MAD。每单元十个样本，B1/B2 独立列出。压缩与解压分别比较，不平均；完整原始 MB/s 的 median/MAD 及每轮值见 JSON。

| 输入 | 上下文 | 属性 | 压缩 B1 % (MAD) | 压缩 B2 % (MAD) | 解压 B1 % (MAD) | 解压 B2 % (MAD) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| zstd-software-eval | thinlto-native | always-inline | -0.228 (0.468) | +0.227 (0.496) | +0.006 (0.126) | +0.117 (0.054) |
| zstd-software-eval | thinlto-native | noinline | -0.480 (0.243) | +0.121 (0.428) | -0.025 (0.051) | -0.133 (0.098) |
| zstd-software-eval | pgo-thinlto-native | always-inline | +0.698 (0.200) | +0.977 (0.213) | -0.174 (0.353) | -0.165 (0.068) |
| zstd-software-eval | pgo-thinlto-native | noinline | -0.067 (0.134) | +0.054 (0.335) | -0.059 (0.354) | +0.009 (0.146) |
| zstd-records-eval | thinlto-native | always-inline | -0.322 (0.438) | +0.677 (0.968) | -0.047 (0.130) | -0.036 (0.142) |
| zstd-records-eval | thinlto-native | noinline | +0.015 (0.904) | +0.170 (0.846) | +0.023 (0.067) | -0.063 (0.116) |
| zstd-records-eval | pgo-thinlto-native | always-inline | +0.672 (0.735) | +0.832 (0.528) | +0.301 (0.102) | +0.161 (0.118) |
| zstd-records-eval | pgo-thinlto-native | noinline | -0.345 (0.790) | -0.359 (0.351) | -0.083 (0.198) | -0.102 (0.138) |

## 解释与限制

PGO+ThinLTO 的 always_inline 压缩在两语料两批均为约 +0.7%–1.0%，同时 text 增加 2368 字节（约 0.32%）；源码语料的配对 MAD 小于差异，记录语料波动较大。ThinLTO 无 PGO 时两种语料均跨批变号，因此未得到统一稳定的内联规则。解压方向也随语料变化。这是负载相关的小信号，保留复核价值，尚不足以证明可推广收益或新的全局神经架构必要性，本轮停止扩展此候选。

这是对一个局部内联候选的受控结果，不是新算法收益。noinline 与原始版保留同样调用仍有小幅吞吐差异，说明布局、其它上下文决策及系统波动必须考虑。保持原评测语料，未据结果扩大函数搜索。

两批交错、轮次旋转并反转起始顺序；固定单核亲和但未预留 SMT sibling、锁频或控制邻核/缓存，差异不作显著性结论，配对统计也不是置信区间。解压结果可反映布局及波动，不能归因于压缩 helper 的直接计算变化。训练预算按相同原生工作单位与时间参数冻结，非精确相同 CPU 周期。

构建接入日志 `20261019T174414997996Z`；`20261019T174733285149Z` 暴露跨文件系统 rename，改为复制后 `20261019T174849641429Z` 接入通过。接入样本不纳入上表。后续按预冻结协议执行 SQLite 覆盖与训练配方对照，不开发神经模型。

复现：`bash scripts/benchmarks.sh all --suite zstd-ab`。默认复用隔离源码/构建/训练缓存，再做两批十轮并保存新日志；重训必须使用新的版本化命名空间，不覆盖本次 profile。
