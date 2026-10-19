# 大型真实程序 Benchmark 环境

本轮优先环境准备与-O3基线验证，不推进M8、不开发新优化算法或模型，不处理SPEC CPU 2026。固定来源见experiments/real-programs.json。

## 存储与来源

仓库benchmark-data是被Git忽略的软链接，指向/mnt/data/chp/irloom-benchmarks。downloads保留官方归档，sources为固定版本源码，builds为分项目/工具链/配置的增量构建，install为本地工具，data为固定输入，tmp为临时目录。日志始终保存在仓库logs/benchmarks/。源码、归档、二进制与大构建均不提交；首次实际下载434.31 MiB，其中test-suite约336.30 MiB。避免下载LLVM完整历史，使用官方拆分源码。

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
