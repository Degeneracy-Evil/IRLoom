# Campaign C1–C2：cleanup carry 修复与既有候选分析

2026-10-24，基于 `7362444` 的审查要求。没有训练、调参、扩展搜索或重跑性能矩阵。旧 campaign、权重、标签、SDK、失败产物和报告保留。机器摘要见 [JSON](architecture-campaign-correctness.json)，完整证据在 `benchmark-data/architecture/correctness-v1/c1-review/`。

## 失败在哪一步发生

原 3 次 raw 失败和 6 个 guarded 内部错误候选，去重后是两个 S0 inline 位置、同一个 S1 `main` reduction 的 unroll2、S2 advance。两个 LLVM 文本对应一个目标文件字节类。重新从各自快照确认注册动作及结构路径，没有跨路径复用节点编号。

旧编译器的两类路径分别比较 default、inline-only、unroll-only、combined，输入为正整数 `1,2,3,9,10,11,20,21,300`。default 和 inline-only 全部正确；unroll-only 和 combined 出错。该循环在未内联时已经存在，单独展开对照成立。错误首见于 affine unroll 为主循环创建 cleanup loop 的状态接续，早于 LLVM lowering。

固定 LLVM `26eb4285b56edd8c897642078d91f16ff0fd3472` 的 `mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp` 中，`generateCleanupLoopForUnroll` 遍历主循环结果、cleanup 结果及 **live initializer range**，逐项调用 `replaceUsesOfWith(init, result)`。当多个 carry 共享同一个初值时，该调用同时改写多个 operand；下一轮又读取已经改变的初值。初值同时充当 bound 时还会误改 bound。

Spectral Norm 最终 reduction 的两个 f64 carry 都以 zero 初始化。旧 cleanup 接到 `[vBv,vBv]`，应接 `[vv,vBv]`；偶数 N 的 cleanup 为零次循环，最后比值成为 1，N=10 输出 `1.000000000`，独立 C oracle 为 `1.271844019`。verifier 接受该 IR，因为类型和结构仍合法。

修复先重定向循环外结果，再按 **operand 位置** 将每个 cleanup carry 连接到对应主循环结果，不按值全局替换。没有程序名、seed 或节点黑名单，没有改源算法、模型、动作注册、verifier、oracle、O3 或数值门槛。

## 独立复现与工具链隔离

[最小 MLIR](../../compiler/diagnostics/affine-cleanup.mlir) 和 [C++ runner](../../compiler/diagnostics/affine-cleanup-runner.cpp) 不依赖原程序、前端、模型或来源属性。包含共享初值、不同初值、初值与 bound 别名、常量奇/偶 trip count。原官方 `polygeist-opt --affine-loop-unroll=unroll-factor=2` 也能复现共享初值错误，将原因收敛到固定 SDK utility，而非 IRLoom bridge 的来源记录。

例如 N=2，预期五项 `2004 3006 4 3006 4008`；旧 unroll 得到 `4004 3006 6 5006 4008`；修复后恢复预期。旧/新 default 及新 unroll 均通过全部九个输入。旧 malformed LLVM 交给 O0 后端仍错，原 C 的 O0/O3 oracle 全部一致；O0 仅用于定位，正式回归保持 O3。

新 revision `cleanup-positional-v1` 使用同一 LLVM/Polygeist SDK（Polygeist `77c04bb2a7a2406ca9480bcc9e729b07d2c8d077`）。[构建入口](../../scripts/prepare-correctness.sh) 复制并修补单个 LoopUtils.cpp，复用原 bridge 对象和静态库，私有对象先于 archives 链接；link map 断言旧 `LoopUtils.cpp.o` 未被提取。无需下载或全量重建，旧 SDK 源码、bridge 和 archive 不覆盖。新源码、对象、binary、构建记录在 `benchmark-data/architecture/compilers/cleanup-positional-v1/`，许可仍为 Apache-2.0 WITH LLVM-exception。

## 有限回归结果

| 检查 | 实际结果 |
| --- | --- |
| 两类失败路径 × 四种固定模式 × 九输入 | 修复后 72/72 通过 |
| 原 32 条完整提议的匹配动作 | 全部正确，目标文件与原历史产物逐字节一致 |
| 原 B-s23、B_current_only-s23、A-s47 raw 调用 | 冻结权重通过原检查及九输入扩展检查 |
| PC1/Matrix/Puzzle/Perm：default 与已测 unroll | 八条路径均通过 recipe 检查；七个对象与历史相同，Puzzle 改变 |
| guarded 私有 proposal 人为替换错误输出 | 两个错误候选被结构化标记 wrong_output；默认回退、确认通过 |
| guarded 注入执行失败 | 失败记录、回退和默认确认通过 |
| 旧编译器固定错误轨迹 | 拒绝输出，失败 JSON 保留 0 正确终端、3 次尝试、实际 stdout 与正成本 |

故障注入最初暴露 Session.fork 固定创建基类，导致注入子类未被继承；现改为保留实际类型，并重新完成真实错误产物检查。首次未注入运行保留，不计入最终拒绝证据。所有失败/编译/候选/默认/确认成本仍进入共同账本。通常无正确终端的 raw 现在也写 `guided-failure.json`，便于保留失败信息。

## 273 条旧标签的复用边界

通过匹配 SDK 的结构 API 审查 37 个 S1 前缀，按结构位置匹配实际循环，检查共享 init、bound 别名和实际 cleanup 需求：260 条不受此次机制修复影响；13 条需重新验证/采集；未定 0。后者为 Puzzle 训练 7 条、DES 验证 6 条。仅重编译这 13 条原路径，原输入完整输出全部正确，但 13 个可执行文件及其 `.text` 均改变，全部需要 **新性能标签**。本轮没有测量或将旧耗时分配给新产物。

训练 S1 的 113 强对中，65 对至少一侧为改变的 Puzzle 候选（三个前缀分别 19、27、19）；其余训练 S1 48 对、训练 S0 24 对、验证 S1 33 对不触及这些改写候选。DES 的六条改变记录不构成旧强排序对。这是旧监督的适用范围审计，不是对新二进制排序的预测。260 条“机制不受影响”也不等于跨版本/跨硬件的普遍有效性证明。旧标签与结论均不倒改。

## C2：实际候选空间

只读原有标签及第 200 次更新时的预测；不调用模型重新选路，不增加测量。按同程序/前缀 advance 参照和原分辨门槛分类，273 条记录中可执行文件未变 140、产物改变但无可分辨收益 107、已测正确有益 8、有害 18。50 个前缀组中仅 5 组存在已测有益候选，5 组所有可执行文件相同。MLIR/LLVM 文本、对象、ELF、ELF `.text` 均做直接字节类比较，不计算 hash；文本不同不代表语义不同，`.text` 相同也不能排除 relocation/data 差异。

| 程序 | 家族 | split | 前缀组 | 记录 | 单一可执行类的组 | 有益组 | 有害组 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PC1 | pc1 | train | 4 | 15 | 0 | 1 | 0 |
| Lists | linked-list | train | 4 | 16 | 1 | 0 | 0 |
| BubbleSort | sorting | train | 4 | 14 | 0 | 0 | 0 |
| Towers | towers | train | 4 | 19 | 0 | 0 | 0 |
| Queens | backtracking | train | 3 | 10 | 0 | 0 | 0 |
| Puzzle | backtracking | train | 4 | 36 | 0 | 3 | 3 |
| HeapSort | sorting | train | 3 | 6 | 1 | 0 | 0 |
| QuickSort | sorting | train | 4 | 14 | 0 | 0 | 0 |
| Perm | perm | train | 4 | 18 | 1 | 0 | 3 |
| ReedSolomon | reed_solomon | train | 4 | 38 | 0 | 1 | 0 |
| Matrix | matrix | validation | 4 | 36 | 1 | 0 | 3 |
| Perlin | perlin | validation | 4 | 16 | 1 | 0 | 3 |
| DES | des | validation | 4 | 35 | 0 | 0 | 0 |

有益组的已测最佳/advance 比为 PC1 S0 1.0890×、ReedSolomon S0 1.0441×、Puzzle 三个 S1 前缀约 1.1320–1.1328×。这些仅为历史候选集合中的最佳，非全动作 oracle；尤其 Puzzle 修复后的性能尚未测量，不能沿用该加速。

下表 direction 为先组内强对准确率、再按指定单位等权平均；没有强对的单位不填 1。beneficial selection / harmful rejection 取模型在已测候选集合内的选择，分母仅为具有对应候选的组，再按程序宏平均。不能当成全池 proposal 的成功率。

| 模型 | train方向/前缀 | train方向/程序 | train方向/家族 | train有益选择/程序 | validation方向/程序 | validation有害拒绝/程序 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-s23 | .8384 | .7091 | .8239 | .6667 | .4481 | .5000 |
| B-s23 | .8960 | .8770 | .8563 | .6667 | .7296 | .8333 |
| current_only-s23 | .9295 | .8971 | .8751 | .6667 | .4481 | .5000 |
| B_current_only-s23 | .9495 | .9091 | .8864 | .6667 | .7815 | .8333 |
| A-s47 | .9664 | .9435 | .9300 | 1.0000 | .4481 | .5000 |
| B-s47 | .8794 | .8791 | .8640 | .6667 | .4481 | .5000 |
| current_only-s47 | .8907 | .8071 | .7595 | .3333 | .4481 | .5000 |
| B_current_only-s47 | .9011 | .8861 | .8676 | .6667 | .4481 | .5000 |

训练分母为 38 前缀/10 程序/7 家族，其中有强对 9/5/4，有益 5/3/3，有害 6/2/2；验证为 12/3/3，其中有强对和有害均 6/2/2，无有益组（选择率应记 null）。训练已测有害拒绝率各模型均为 1。完整 JSON 保留每前缀预测、漏选标记、full proposal 是否已测、容差后的 log regret，以及程序/家族明细。未测 proposal 不标成有害；历史 raw 语义错误另记 C1，不能当作低耗时收益。S0 无前阶段连续状态，不能把 S0 的有益选择归于记忆。A-s47 的训练宏平均较好并未形成稳定验证优势，也没有新的记忆收益结论。

## 复现、限制与后续

在保留原 campaign 和匹配 SDK 构建对象的机器，从仓库根目录执行：

```sh
bash scripts/architecture.sh --correctness-audit all --name c1-review
```

已完成同名结果直接复用；新名字执行一套独立正确性审计。细分 `isolate|minimal|regress|analyze` 用于定位，`regress` 应在同名 `isolate` 后执行。日志在 `logs/architecture`，临时目录在 `tmp`，大型证据在外部缓存。部分中断时不强行覆盖已有 session，应保留现场、使用新名字复现。详细工具命令和输入输出保留在各 JSON events 与日志。

新修复的正常完整程序运行需显式指定 `--compiler-revision cleanup-positional-v1`。默认 legacy 保留历史协议，不能把默认命令误认为已迁移；新正确性审计自动准备隔离修复。完整 C1 同名重复已校验不产生新 run、不改变历史 JSON/权重及旧 bridge/pass，直接比较字节，无 hash。

下一步应先按新 revision 独立采集上述 13 条受影响标签及匹配参照，保持原完成策略、测量与分组协议；然后再决定训练/架构对照。本轮不自动启动该新活动。覆盖仍限于已支持完整程序和有限输入，未证明所有变换语义等价；没有新的工业规模或 blind 泛化结论。
