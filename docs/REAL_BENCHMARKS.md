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

复用Ubuntu已有LLVM/Clang18.1.3、llvm-profdata、compiler-rt profile runtime、CMake3.28.3、GNU Make4.3、Tcl、C/C++系统头文件与链接库，以及现有uv Python环境。需要git/curl、tar/xz/zip支持、dpkg-deb；不执行系统安装。未安装Ninja，使用Make生成器即可。CMake内部Python通过scripts/benchmark-python.sh的uv包装调用。

默认最多两个构建进程。LLVM/Clang源码已准备，主编译器复用系统安装；先配置LLVM+Clang并仅编译小型LLVMDemangle库，避免默认全量bootstrap/安装LLVM。源代码版本与Ubuntu编译器打包修订分别记录，不冒称为自己从源码构建的Clang。PGO/ThinLTO先做工具能力验证，后续真实项目对照须分开构建目录及训练/测试输入。

## 获取与环境记录

```bash
bash scripts/benchmarks.sh fetch
bash scripts/benchmarks.sh probe
```

均为幂等操作：完整归档缓存后不重复下载，源码marker校验URL/版本/提交身份；临时下载只有成功后才转成正式缓存。按用户约定不计算产物hash，使用固定官方来源/版本、归档字节数和元数据记录关联，因此没有额外提供密码学完整性校验。SQLite ZIP的执行位由ZIP原始属性恢复，避免configure助手缺权限；不修改上游源码文本。

fetch/probe已实跑，第二次获取全部命中缓存，环境记录在logs/benchmarks/<时间戳>/environment.json；保存CPU/ISA/cache/内存/亲和、频率控制边界、工具绝对路径与版本、源码来源与许可。完整运行入口与结果将在构建/执行验收后补充。
