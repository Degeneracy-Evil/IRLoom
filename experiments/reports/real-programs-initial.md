# 真实程序 Benchmark 环境首轮验收

本轮完成官方固定来源、外部缓存、受限构建和-O3执行测量；不开发新优化算法或神经网络。完整复现见[运行指南](../../docs/REAL_BENCHMARKS.md)，来源与选项见[固定配置](../real-programs.json)，原始环境/样本/命令见[审计摘要](real-programs-initial.json)。

## 部署范围

- LLVM test-suite：固定llvmorg-18.1.3 / 72a946a42cbce06afef68b2326ff6c3869d09799，验证7zip-benchmark和Ptrdist/anagram，不声称完整suite全部通过。
- Zstd1.5.7：固定f8745da6ff1ad1e7bab384bd1f9d742439278e99，源码编译-O3，单线程8MiB文本语料，压缩/解压逐字节一致。
- SQLite3.50.4：完整官方源码，含speedtest1；版本/SOURCE_ID精确匹配。SQL事务回滚、聚合、JSON/FTS5/RTree已知结果通过；文件speedtest后integrity_check和非空表检查通过。正式测量main/--memdb/--size25。
- LLVM/Clang18.1.3：复用Ubuntu现有编译器；已下载官方llvm/clang/cmake/third-party拆分源码并成功配置LLVM+Clang，只构建LLVMDemangle及其C++工作负载；额外测量Clang编译官方ItaniumDemangle.cpp一个TU。没有全量bootstrap Clang。
- 本地LLD18.1.3-1ubuntu1从Ubuntu官方deb解包，不安装到系统；ThinLTO链接/执行和Clang前端PGO生成→合并→使用→执行均通过。探针只作用于驱动，静态库仍是O3，不能当作整项目优化对照。

## 两次完整入口验收

- `logs/benchmarks/20261019T145513159552Z`：干净提交`d72893c`，六项工作负载各一次预热+三次测量，全部通过；完整流程94.36s。
- `logs/benchmarks/20261019T145759896060Z`：干净提交`15ad677`，六项工作负载各一次预热+三次测量，全部通过；完整流程28.23s。

第二次没有curl下载事件，没有重新configure或重链接未变的LLVM驱动；官方项目走增量构建。首次流程包含部分新版本化构建，复跑时长差主要来自缓存，不能解释为代码优化加速。

| 工作负载 | 首轮进程墙钟中位s | 缓存复跑中位s | 复跑MAD s |
| --- | ---: | ---: | ---: |
| llvm-demangle | 0.024581 | 0.024667 | 0.000031 |
| clang-compile-llvm-tu | 1.899776 | 1.895812 | 0.000178 |
| zstd-b3 | 2.090808 | 2.080325 | 0.000907 |
| sqlite-speedtest1 | 0.395026 | 0.394846 | 0.000641 |
| test-suite-7zip | 1.574461 | 1.574685 | 0.001688 |
| test-suite-anagram | 0.589105 | 0.587119 | 0.000200 |

复跑Zstd原生吞吐中位：压缩400.0 MB/s，解压1759.2 MB/s；SQLite内部TOTAL中位0.393s；anagram官方timeit记录的CPU exec_time中位0.474600s。进程墙钟包含启动，lit还含Python和参考输出验证，不应与原生指标混用。

## 环境与存储

Ubuntu24.04/x86_64，Intel Xeon6986P-C；Clang/LLVM18.1.3、CMake3.28.3、GNU Make4.3、glibc2.39、Tcl8.6.14、Python3.12.3。优化输入为-O3 -DNDEBUG，不加-march=native或fast-math，项目本身的功能开关保留并记录在configure/compile_commands中。默认build jobs=2；运行亲和CPU0，单线程/单worker设置固定；频率、共享负载、OS缓存不可控，无PMU。

大文件位于/mnt/data/chp/irloom-benchmarks，通过仓库忽略的benchmark-data访问。归档合计434.31MiB，源码约3.5GiB；包含开发与正式缓存的实际分目录空间如下。未删除开发缓存，也没有系统安装。

| 目录 | 占用MiB |
| --- | ---: |
| sources | 3574.96 |
| downloads | 434.32 |
| builds | 276.79 |
| data | 16.00 |
| tmp | 37.19 |
| install | 4.98 |

## 处理的问题与限制

- SQLite ZIP需要恢复原始POSIX执行位；已集成到获取入口。本版speedtest1内存参数为--memdb。
- LLVM test-suite从MultiSource顶层配置以继承traditional_output；只构建两个指定target。关闭默认artifact hash模块，保持官方fpcmp参考验证；运行输出与test.log重定向到仓库logs，生成配置变更不修改上游源码。
- 通过uv添加psutil7.2.2用于lit每测试60秒超时；lit本身直接使用固定LLVM18.1.3源码版本，不额外安装另一套lit。
- 早期配置/参数探查失败保留在logs；最终两次完整入口全部通过。未增加单元测试，未额外计算产物hash。
- 只验证列出的子集：Zstd语料由固定源码文本构造，SQLite正式为内存模式，7zip采用256KiB字典和一次迭代。没有生产数据代表性或优化收益结论。
- 下一阶段先为-O3/ThinLTO/PGO建立独立构建配置，冻结相同功能开关及更有代表性的输入，划分PGO训练/评测，再比较编译/训练成本与最终性能；当前能力探针不能复用为正式收益证据。

## 手动复现

```bash
bash scripts/benchmarks.sh
```

可按--project选择单项目；每次新建日志和临时目录，已有源与构建缓存自动复用。源码、归档、二进制和大型build均未进入Git。
