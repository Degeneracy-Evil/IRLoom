# 真实程序因果 A/B 冻结协议 v1

执行用户新增的REAL_OPTIMIZATION_AB.md。只验证一个Zstd函数和一个SQLite训练覆盖问题；不改动R1–R3源码缓存、产物或报告，不扫描其它函数/Pass，不开发模型。

## Zstd：在测量前冻结

Zstd1.5.7/Clang+LLD18.1.3，原功能选项不变。三组只改变lib/compress/zstd_compress.c中ZSTD_buildSeqStore声明：原始、always_inline、noinline。该函数与compressBlock_internal同TU，其后间接调用的匹配器位于其它TU，不能将目标属性改动本身称为跨TU内联。补丁在experiments/patches，工作副本在benchmark-data/sources/zstd-1.5.7-inline-ab-v1/，产物在builds/zstd/1.5.7/clang-18.1.3/inline-ab-v1/处理组/编译上下文。

每组各有ThinLTO-native、全新PGO生成、PGO+ThinLTO-native产物。三个PGO profile分别由对应插桩程序用同样的formal-v1 software-train/records-train采集，固定-b3 -i1 --single-thread，各语料压缩/解压同样的计时预算；内部迭代数可随速度变化，不能称为相同调用次数。禁止借用旧profile；记录生成构建、采集、合并、应用的真实CPU/墙钟成本及缓存复用。

评测固定已有software-eval/records-eval，两批、每批每组一次预热+10次独立进程，-b3 -i2 --single-thread。每轮旋转六个处理/上下文组合，第二批反转起始顺序；执行CPU0，线程1，频率、SMT sibling、邻核和系统缓存不隔离。分别报告压缩/解压MB/s的median/MAD、逐轮成对比例和批间变化，不平均掩盖退化。两种语料与训练文件/种子隔离；它们是已有诊断输入，不是新增blind泛化证明。

每组先逐字节往返，收集最终LTO inline记录、优化后bitcode/IR、目标/调用方实际符号与汇编、总text及相关符号体积。只有最终无剩余helper调用且源调用点有AlwaysInline证据，才视为强制inline生效；noinline须保留实际调用。源中三个调用点可因上下文展开为更多后端调用，不混淆源位置与clone。只从完整应用吞吐判断结果：负结果或噪声级收益都如实记录，不继续搜索函数。

入口：`bash scripts/benchmarks.sh all --suite zstd-ab`；可用build只准备构建。单轮接入用`--batches 1 --repetitions 1`，不进入正式结果。

## SQLite：评测/训练配方在首次运行前冻结

P0保留官方speedtest1 orm,cte训练类型；P1采用可审计、固定种子的SQL索引查询训练。两者对同一版本、相同功能宏的SQLite核心插桩构建采集，各分配60个进程CPU秒，完整工作单元结束时停止，允许不超过2%的完成粒度差异并公开实际CPU秒数，超过则不作等预算结论。P0以原有size40完整工作单元重复，P1以固定seed的独立数据集/查询循环；不使用旧评测main参与profile合并。

新增C++ SQLite API工作负载保持相同程序/编译标志，定义如下：表t(k INTEGER,v INTEGER)，k不是rowid，单独建立唯一索引；N=100003；v=3*k+seed%97。固定查询seed、数据规模、查询次数及类型：训练P1 seed=42001；新评测point seed=52001、2000000次精确索引查找；新评测range seed=62003、200000次长度32的索引范围聚合。输入由标准mt19937原始输出取模生成，查询预期count/sum用整数闭式公式独立计算，每次检查，禁止hash。两个评测此前未执行，协议冻结后不依据P0/P1差异改种子/规模；旧main只能辅助，不称blind。

原始SQL源码、输入/CPU预算、profile覆盖和实际构建命令都保存。用Clang coverage映射将VdbeExec中BtreeIndexMoveto调用语句对应到源行/区域计数，并区分训练P0、P1及独立审计评测profile；评测profile绝不用于重建正式PGO应用。补充最终核心IR/汇编与前端/最终LTO记录，注明哪些调用点已内联或保留。评测两批、每批一次预热+10次交错独立运行，记录内部计算时间、墙钟、正确性、代码体积、构建/训练费用。

若P1的额外覆盖已被常规PGO有效利用，结论属于训练配方；若有回退或无可分辨差异，同样记录。没有残留因果收益和跨对象交互证据就不升级到模型/全局架构。实现中任何协议偏离先记录，不能悄悄替换评测输入。
