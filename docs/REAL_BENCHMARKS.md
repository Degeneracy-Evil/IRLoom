# 大型真实程序 Benchmark 环境

本轮优先环境准备与-O3基线验证，不推进M8、不开发新优化算法或模型，不处理SPEC CPU 2026。固定来源见experiments/real-programs.json。

## 存储与来源

仓库benchmark-data是被Git忽略的软链接，指向/mnt/data/chp/irloom-benchmarks。downloads保留官方归档，sources为固定版本源码，builds为分项目/工具链/配置的增量构建，install为本地工具，data为固定输入，tmp为临时目录，并经仓库tmp/real-benchmarks软链接访问，以保留./tmp约定。日志始终保存在仓库logs/benchmarks/。源码、归档、二进制与大构建均不提交；首次实际下载434.31 MiB，其中test-suite约336.30 MiB。避免下载LLVM完整历史，使用官方拆分源码。

| 项目 | 固定版本/提交 | 官方来源 | 许可证/声明 |
| --- | --- | --- | --- |
| LLVM test-suite | llvmorg-18.1.3 / 72a946a42cbce06afef68b2326ff6c3869d09799 | https://github.com/llvm/llvm-test-suite | Apache-2.0 WITH LLVM-exception；内置程序各自许可证 |
| Zstd | 1.5.7 / f8745da6ff1ad1e7bab384bd1f9d742439278e99 | https://github.com/facebook/zstd | BSD-3-Clause OR GPL-2.0-only |
| SQLite | 3.50.4，2025-07-30 SOURCE_ID固定在配置中 | https://www.sqlite.org/2025/sqlite-src-3500400.zip | 核心代码Public Domain，构建工具另见声明 |
| LLVM/Clang源码 | 18.1.3，llvm/clang/cmake/third-party拆分包 | https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.3 | Apache-2.0 WITH LLVM-exception及各第三方声明 |
| 本地LLD | Ubuntu lld-18 18.1.3-1ubuntu1 amd64官方deb | https://archive.ubuntu.com/ubuntu/pool/universe/l/llvm-toolchain-18/ | LLVM许可与Ubuntu打包声明 |

源码中的LICENSE/COPYING等文件原样保留，具体路径进入环境记录。7zip是test-suite中较旧的内置版本，其License.txt声明LGPL-2.1-or-later，RAR部分另附unRAR限制；anagram还有Raymond Chen的保留修改说明要求，见Ptrdist/LICENSE.TXT。这里不把顶层LLVM许可证冒充所有内置程序的唯一许可。

## 依赖与成本控制

复用Ubuntu已有LLVM/Clang18.1.3、llvm-profdata、compiler-rt profile runtime、CMake3.28.3、GNU Make4.3、Tcl、C/C++系统头文件与链接库，以及现有uv Python环境。需要git/curl、tar/xz/zip支持、dpkg-deb；不执行系统安装。未安装Ninja，使用Make生成器即可。CMake内部Python通过scripts/benchmark-python.sh的uv包装调用。lit直接使用固定LLVM源码中的18.1.3版本，新增Python依赖为psutil（BSD-3-Clause，https://github.com/giampaolo/psutil ，uv.lock固定），用于60秒测试超时控制。

默认最多两个构建进程。LLVM/Clang源码已准备，主编译器复用系统安装；先配置LLVM+Clang并仅编译小型LLVMDemangle库，避免默认全量bootstrap/安装LLVM。源代码版本与Ubuntu编译器打包修订分别记录，不冒称为自己从源码构建的Clang。PGO/ThinLTO先做工具能力验证，后续真实项目对照须分开构建目录及训练/测试输入。

## 获取与环境记录

```bash
bash scripts/benchmarks.sh fetch
bash scripts/benchmarks.sh probe
```

均为幂等操作：完整归档缓存后不重复下载，源码marker校验URL/版本/提交身份；临时下载只有成功后才转成正式缓存。按用户约定不计算产物hash，使用固定官方来源/版本、归档字节数和元数据记录关联，因此没有额外提供密码学完整性校验。SQLite ZIP的执行位由ZIP原始属性恢复，避免configure助手缺权限；不修改上游源码文本。

fetch/probe已实跑，第二次获取全部命中缓存，环境记录在logs/benchmarks/<时间戳>/environment.json；保存CPU/ISA/cache/内存/亲和、频率控制边界、工具绝对路径与版本、源码来源与许可。构建/执行入口如下，运行时会自动做增量构建。

## 统一入口

```bash
bash scripts/benchmarks.sh                         # 默认全部项目，jobs=2，每项一次预热+三次测量
bash scripts/benchmarks.sh build                   # 只准备源码/配置并增量构建
bash scripts/benchmarks.sh run --project sqlite    # 单项目也自动保证产物已构建
bash scripts/benchmarks.sh all --project zstd sqlite --repetitions 5
```

可选项目为llvm、zstd、sqlite、test-suite；--jobs限制1..8，默认2。所有入口在仓库内执行，Python经uv，无--python和直接pip调用。fetch/probe/run的新日志路径会打印，summary.md与results.json包含状态、逐次样本、median/MAD、工作负载内部指标、完整命令和构建/执行耗时；失败保留日志并返回非零。

构建缓存位于benchmark-data/builds/<项目>/<固定版本>/clang-18.1.3/o3。配置和compile_commands.json记录实际编译选项，SQLite使用原生Makefile并保存configure设置；改动SQLite编译选项不得悄悄复用既有o3目录。所有基线统一-O3 -DNDEBUG，不加-march=native/fast-math；源代码和CMake缓存可复用，源码版本与工具链/变体目录分开。

## 已接入的工作负载与校验

| 项目 | 内容 | 校验/测量方式 |
| --- | --- | --- |
| LLVM/Clang | 官方LLVM的LLVMDemangle源码库与C++驱动；Clang编译官方ItaniumDemangle.cpp单个TU | API已知结果检查；内部循环秒数与进程墙钟；编译吞吐仅记录单TU墙钟 |
| Zstd1.5.7 | level3，单线程，8MiB固定源码文本语料 | 压缩/解压后直接逐字节相同；内置-b3 -i1吞吐与进程墙钟 |
| SQLite3.50.4 | sqlite3 shell、speedtest1 main；正式基线--memdb --size25 | 版本/SOURCE_ID匹配，事务/聚合/JSON/FTS5/RTree已知结果；小文件工作负载后integrity_check；内部TOTAL与进程墙钟 |
| LLVM test-suite | 内置p7zip9.20的7zip-benchmark；Ptrdist/anagram | 7zip b 1 -mmt=1 -md=18内置检查，约6MiB需求；anagram通过原版lit/timeit/fpcmp比较官方参考输出，保留exec_time |

Zstd语料由该固定版本lib/**/*.c按路径排序、拼接、重复后截取8MiB，生成说明随data缓存保存；只是环境基线，不代表Silesia/生产数据。SQLite内存模式减少磁盘噪声，不能当作持久化数据库性能；文件模式只做较小完整性检查，不运行全量上游测试。7zip显式限制线程和字典，不直接使用可能扩大并发/内存的上游默认参数。

lit必须从MultiSource顶层配置，以保留上游traditional_output继承；仅构建/运行上表两个target，未验证整个test-suite。BENCHMARKING_ONLY排除UnitTests，未编写新单元测试。生成的lit.site.cfg移除默认二进制hash指标并把执行/日志目录指到本次logs下，保留原官方fpcmp数值/文本参考检查；上游源码文本不做补丁。

## ThinLTO/PGO准备与限制

全入口的llvm项目会做能力探针：C++驱动以-flto=thin配合本地LLD链接并执行；再以-fprofile-instr-generate产生独立临时profraw、llvm-profdata merge合并、-fprofile-instr-use重新构建并执行。ThinLTO缓存和profile均在外部专用目录，链接并行不超过2。这里仅证明Clang前端插桩PGO与ThinLTO基础设施可用，LLVMDemangle静态库仍是O3 native；**没有做整项目ThinLTO/PGO性能对照，也未源码bootstrap Clang**。

后续正式对照须为o3/thinlto/pgo-generate/pgo-use建立独立目录，固定相同项目功能开关/输入，划分PGO训练与未见评测输入，记录标签获取与重建成本；不能拿能力探针数据当正式训练集或优化收益。

运行进程固定到允许的第一个CPU，线程环境为1；一次预热后三次独立进程采样，记录median/MAD与原始样本。频率、邻核共享负载与OS缓存不可控，没有PMU；进程墙钟包含启动，lit墙钟还包含Python/验证开销，应优先查看其原生exec_time。当前目的是环境验收，不发布编译优化加速结论。工具内部CMake/configure诊断随外部build缓存保留，我们的命令/结果与lit运行日志均在仓库logs/。

## 正式入口验收结果

两次完整入口均通过，四个项目共六项工作负载，每项一次预热后三个样本；第二次没有新增下载或重新configure。首次94.36s含部分新构建，缓存复跑28.23s，均非优化加速比较。结果与硬件/源/构建/测量摘要见[首轮验收报告](../experiments/reports/real-programs-initial.md)。实际命令环境见environment.json的benchmark_environment；host是复用的通用机器/工具探针。外部目录当前约4.24GiB（含开发缓存），Git只提交入口、配置、文档和小型结果摘要。

## RealBench Phase 1：正式负载

新方案见[REAL_PROGRAM_NEXT.md](REAL_PROGRAM_NEXT.md)。旧入口默认仍为smoke；新增正式负载入口复用相同源码缓存与工具链，不覆盖初始报告：

```bash
bash scripts/benchmarks.sh all --suite formal --repetitions 1  # 接入验收，不能用于性能结论
bash scripts/benchmarks.sh all --suite formal --repetitions 10 # 一次预热、十次独立测量
```

formal默认选择Zstd、SQLite、test-suite；LLVM/Clang能力检查继续使用smoke入口。新缓存formal-o3与旧o3隔离。

| 正式负载 | 输入/计算 | 校验与限制 |
| --- | --- | --- |
| Zstd software-eval | 固定1.5.7的lib下.c/.h按路径排序，索引非4倍数的文件打包为确定性USTAR；无重复填充 | level3单线程逐字节压缩往返；-b3 -i2分别报告压缩和解压吞吐；训练只用剩余文件 |
| Zstd records-eval | 300000条JSONL，seed=20261021，时间戳/租户/状态/数值/96-bit token | 固定生成配方随数据缓存；训练seed=20261020且30000条；属于合成结构化数据，不能当作生产流量 |
| SQLite main-memdb-size200 | 官方speedtest1 main，相比旧size25放大8倍；内存数据库 | 版本及SOURCE_ID、小文件main size10的integrity_check及非空表；不使用会计算校验hash的--verify；磁盘只做功能校验 |
| Bullet | 官方bullet target；7组物理模拟各100帧 | 官方完整文本参考输出，经lit/fpcmp；源码带zlib式许可声明 |
| tramp3d-v4 | 官方tramp3d-v4 target；模板数值程序，32³网格10步 | 官方参考输出，绝对误差1e-7；README声明FreePOOMA为BSD3，其余Public Domain |

数据与配方在benchmark-data/data/formal-v1。两项MultiSource保留官方输入/参数/验证命令；不修改上游源码。mafft虽有源码，但其上游参考验证需要HashProgramOutput，暂不纳入本轮“不计算hash”的集合。7zip/anagram继续保留旧smoke。正式样本的进程墙钟、Zstd内部吞吐、SQLite内部TOTAL、lit原生exec_time分别记录，不能混用。

R1接入已实跑通过；tramp3d单次过短，formal入口生成20个官方RUN及逐次VERIFY的批次test，仅写外部构建目录，lit的exec_time为20次原生用户CPU时间之和（不是墙钟）。首轮失败的输出路径问题及修复见[阶段报告](../experiments/reports/real-programs-baselines.md)。R1规模/时间/原始样本归档在real-programs-r1.json，不能从单样本推断优化收益。

## R2：强编译配置与独立PGO

formal默认对Zstd和SQLite构建四个完整程序变体：o3（可移植）、o3-native（-march=native）、thinlto-native（native加-flto=thin/本地LLD）、pgo-thinlto-native（同一ThinLTO配置加-fprofile-instr-use）。test-suite的两项数值负载本轮保留O3参考，避免扩大构建矩阵。正式入口默认10次测量；--variants可选择子集：

```bash
bash scripts/benchmarks.sh all --suite formal
bash scripts/benchmarks.sh all --suite formal --project zstd sqlite --variants o3-native thinlto-native --repetitions 10
bash scripts/benchmarks.sh build --suite formal --project zstd --variants pgo-thinlto-native
```

缓存目录为builds/项目/版本/clang-18.1.3/formal-变体。完整静态压缩库与Zstd程序均按变体构建；SQLite使用同一amalgamation与speedtest1、相同功能选项。ThinLTO归档使用llvm-ar/llvm-ranlib，每个链接器内部并行最多2（SQLite两个目标可由Make并发）；native变体保留inline优化记录（.opt.yaml），编译/链接命令、二进制文件大小、llvm-size和符号数进入每次metadata.builds。CMake的compile_commands.json及link.txt保存在各缓存中，SQLite原生Make命令保存在日志。配置诊断开关不会当作优化收益。

PGO先独立构建formal-pgo-train（native+ThinLTO+Clang前端插桩），训练Zstd时只读取software-train与records-train，训练SQLite时只执行orm,cte size40；评测仍为software-eval/records-eval和main size200。训练语料的文件集合/记录种子与评测隔离；SQLite是不同testset而非较小的main前缀。不同测试仍共享实现与数据分布，不能据此宣称生产泛化。

训练profraw/profile保存在各项目pgo-train/training-v1/时间戳；每次新采集都用新目录，合并后仅对应项目使用。训练缓存marker核对训练配方和二进制mtime/大小（不计算hash），匹配时复用profile并明确记录cache_hit；训练二进制重建后重新采集，旧profile不删除、不混入。PGO训练编译、采集、合并、最终重建成本分别可追溯；metadata.profiles保留首次训练成本，后续缓存命中不能冒充重新付费。PGO布局/策略只是这组固定训练输入的结果，不保证加速。

每个正式负载一次预热后10轮，各轮旋转变体次序，单CPU串行运行。保存全部样本、进程墙钟及原生指标的median/MAD；Zstd以压缩/解压MB/s分别比较，SQLite以TOTAL秒、MultiSource以lit exec_time比较（其为timeit用户CPU秒）。所有变体先做Zstd逐字节往返、SQLite版本/已知SQL/文件完整性及跨变体数据库dump比较，官方MultiSource继续逐次参考校验。SMT sibling未预留，频率/邻核负载/页缓存不受控；不能仅凭小幅median差异宣称编译器优化有效。

PGO-use目录再按训练profile的采集时间戳隔离，重新训练不会悄悄复用包含旧-fprofile-instr-use路径的SQLite对象。SQLite configure的conftest main与训练程序不是同一函数体，Clang可能报告profile不匹配；保留原始诊断而不把探测程序告警升级为构建错误，正式编译日志仍须审查。Zstd的ThinLTO内联记录通过LLD单独输出至各变体lto-inline.yaml及其thin分片。

## R3：热点/调用与优化记录审计

formal的all入口在计时结束后执行审计；run只做正式采样。也可单独复用缓存：

```bash
bash scripts/benchmarks.sh audit --suite formal --project zstd sqlite
bash scripts/benchmarks.sh run --suite formal --repetitions 10 # 只测量，不运行审计
```

本机没有perf命令，perf_event_paranoid=4，不改系统配置或安装系统包。复用已有GNU binutils的gprof：另外构建formal-audit-gprof（-O3 -march=native -pg -fno-omit-frame-pointer），验证后对两种Zstd输入和SQLite main采集用户态采样与动态调用弧。GMON_OUT_PREFIX将原始二进制采样文件放在本次logs目录，flat/call graph完整文本也随命令保存。这些插桩构建改变调用/内联成本，不能加入正式性能样本，也不能把库外时间或未插桩库调用当作已完整覆盖。

若已准备PGO，则以formal-pgo-train对评测输入另行采集evaluation-only.profdata，只用于审计函数/基本块执行计数；目录在本次tmp内，与固定训练profile完全隔离，不更新最终PGO-use二进制。计数代表频度而非CPU周期。最后汇总各正式native配置的Clang前端及Zstd LLD ThinLTO内联记录，完整调用边记录留在logs/inline-项目-变体.json；前端与LTO记录通过stage区分，不能直接相加成独立优化次数。

审计结果写metadata.audit，保留工具版本、限制、热点样本、调用图、PGO计数及优化记录路径。下一步候选必须同时核查热度、实际调用边、现有PGO/ThinLTO是否已经处理、合法A/B改动与完整输出校验；看到热点或Missed remark不等于编译器漏优化。

SQLite上游Make会为shell与speedtest1的两个sqlite3.c编译写入同一个sqlite3.opt.yaml，原文件无法可靠归属。审计改用make -n -W读取真实speedtest1编译配方，去除链接输入及harness源文件，保持全部核心编译宏/优化参数，仅为sqlite3.c生成独立诊断对象与明确命名的记录。对象在本次tmp，记录缓存于对应变体audit-remarks；marker核对配置、源码及二进制mtime。此操作不覆盖正式二进制，相关成本进入事件日志；PGO核心诊断额外将profile函数体不匹配升级为错误，防止误用shell的profile解释评测目标。SQLite记录仅代表前端，不能冒充其最终ThinLTO后端决策。

configure及正确性运行也可能触发gprof输出，统一GMON_OUT_PREFIX预设在本次tmp，只有正式审计样本覆盖到logs前缀，防止gmon.out污染仓库。summary.md同时给出原生主指标和剖析概要，results.json仍为完整证据来源。

R1–R3已完成实际验收：14个负载/配置组合各10个样本全部通过；两种Zstd分布、SQLite放大输入、Bullet/tramp3d官方参考有效。强基线结果、构建/训练/剖析成本、数据核查、两个下一步问题与限制见[正式报告](../experiments/reports/real-programs-baselines.md)，原始摘要见同名JSON及real-programs-audit.json。R3干净提交缓存复跑约37秒，不代表完整十轮性能流程耗时。

## 函数属性与PGO训练覆盖A/B

新增任务见REAL_OPTIMIZATION_AB.md；测量前冻结的处理组、输入、预算与判定规则见[REAL_AB_PROTOCOL.md](REAL_AB_PROTOCOL.md)。首先执行单函数Zstd内联实验，保持原始缓存/历史报告不变：

```bash
bash scripts/benchmarks.sh all --suite zstd-ab
```

默认两批，每批各一次预热+10次测量，三种函数属性×两种强编译上下文×两种语料；每个属性重新采集自己的兼容PGO profile。工作副本、构建、profile、保存的最终LTO bitcode均在外部版本化目录，原始样本/最终IR/汇编在logs，临时对象在tmp。只构建不测量用build；接入验收可用--batches 1 --repetitions 1，不作正式收益结论。SQLite的种子/查询量和等预算规则已预先冻结，Zstd正式结果完成后才执行其覆盖A/B。

Zstd正式结果见[独立报告](../experiments/reports/real-zstd-inline-ab.md)。随后执行SQLite独立PGO覆盖对照：

```bash
bash scripts/benchmarks.sh all --suite sqlite-ab
```

P0为官方speedtest1的orm,cte size40完整工作单位，P1为固定种子的索引点查询；各至少60进程CPU秒，末次工作单位量化差异必须不超过1.2秒。生成新SQLite amalgamation并复用官方核心编译配方，两组相同功能宏、native ThinLTO及C++查询驱动；PGO仅应用于SQLite核心。新point/range输入的种子、规模、参考答案及训练隔离在冻结协议中定义。每次运行先核对版本、索引计划、数据库完整性和每条SQL整数答案，再完成两批各十轮；报告完整进程墙钟与查询循环耗时，两者均包含必要的答案核查。

源码调用点覆盖来自llvm-cov，保存训练及单独评测的region计数、前端/最终内联记录、最终LTO IR与所选汇编。评测profile只做审计，永不合并到训练。`build --suite sqlite-ab`只准备独立构建及训练缓存；缓存复用配置、源码/驱动mtime和profile路径，配方改变应使用新命名空间。`--batches 1 --repetitions 1`仅作接入验收。全量源码、构建、profile与bitcode留在benchmark-data外部缓存，不提交Git。

两项A/B均已完成两批十轮，分别240与80个正式样本；结果、费用及限制见[Zstd报告](../experiments/reports/real-zstd-inline-ab.md)和[SQLite报告](../experiments/reports/real-sqlite-pgo-ab.md)。SQLite新索引训练对当前新输入表现较好，但第二批有系统漂移；报告使用同轮配对统计并保留全部离群样本，不把未测负载的性能当作已知。
