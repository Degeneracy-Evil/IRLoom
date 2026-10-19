# SQLite Pmix 冻结协议 v1（2026-10-20）

执行 REAL_AB_REVIEW.md，仅补一个传统混合 PGO 基线，完成后停止 SQLite 局部试验，不再搜索权重/函数属性。旧报告及全部旧缓存保持不变。

## 处理与费用

固定 SQLite 3.50.4、Clang/LLD 18.1.3，沿用 pgo-ab-v1 的核心功能宏、native ThinLTO、同一 C++ API 驱动。P0/P1 直接复用此前独立 profile 和生产核心对象。Pmix 仅将两份 profile 以整数权重 **1:1** 合并，不根据任何本轮评测选权，不探索其它权重。

依据：P0/P1 原训练约各60进程CPU秒，1:1 保留这两个同等历史采集窗口内的原始动态计数。它代表历史窗口的计数叠加，**不保证两种负载相同 counter mass，也不是 CPU 时间归一化**。先用本机 llvm-profdata show 记录两类总函数计数/总块计数和详细摘要，merge --weighted-input=1,... --num-threads=2 --failure-mode=any；核查核心函数 profile 结构兼容及各块计数相加，正式核心启用 profile 不匹配为错误。LLVM 文档：https://llvm.org/docs/CommandGuide/llvm-profdata.html ，本机18的help/实际执行优先。

P0/P1 各约60秒历史训练；Pmix **使用约120秒历史训练数据**，本轮新增训练0秒，另计实际合并、构建及验证费用。不能称同预算优势。harness main 来自不同程序，profile 中可有多个main结构；PGO只应用于相同核心，绝不应用于主程序。

## 评测与正确性（首次执行前冻结）

四个家族权重各1/4：

1. point：既有C++驱动，N=100003，seed=72007，2000000次查询。
2. range：同驱动，N=100003，seed=82009，200000次长度32范围查询。
3. orm：官方 speedtest1 --memdb --size 60 --testset orm。
4. cte：官方 speedtest1 --memdb --size 60 --testset cte。

新种子及size60未用于合并定权；训练后不再更新profile。point/range模式及ORM/CTE家族此前已观察，CTE部分固定拼图与size40训练相同，故**整体仅作预冻结诊断性比较**，不声称新blind或跨家族泛化。旧point/range种子不重新当作新样本。

point/range继续逐条整数oracle、数据库完整性、索引计划、版本/SOURCE_ID检查。官方ORM/CTE另建仅供验证的step记录harness，将所有sqlite3_step状态及每行各列类型/原始值写至tmp，不计算hash，逐字节比较三组与同配方未优化(-O0、无PGO/LTO)参考核心的完整结果流。原版harness独立链接同一生产核心对象进行正式计时，不启用--verify（上游会计算hash），不把结果流I/O纳入性能样本。记录验证重链接与最终生产链接上下文不同的局限；运行错误必须中止。参考核对不是新单元测试。

## 测量、聚合与停止

CPU0串行，两批，每配置/家族每批预热一次并测10个独立进程（240正式样本、24预热）。每轮旋转P0/P1/Pmix，第二批反转起始顺序。报告原生TOTAL/查询循环及完整进程墙钟median/MAD、同轮配对比、异常/漂移，不删离群值。主要统计为**完整进程墙钟**。

每轮/每批，以P0为参考，对四家族延迟比取等权几何均值；先逐轮聚合再报告median/MAD，不能拿独立median之比代替配对统计。报告各家族最差退化及其跨批变化。诊断oracle下界为每轮每家族三个已测配置的最短墙钟组合（不是真实可部署单程序），用于度量Pmix到三配置观测前沿的距离；存在事后选择偏差，不能当新算法。

若两批Pmix的几何均值相对该oracle延迟差均不超过3%，称“本小矩阵接近观测前沿”；3%为预冻结描述阈值，非统计显著性。否则报告残留权衡，不追加调权或属性试验。无论结果如何，本轮仅交付报告及停止判断；未证明决策交互和传统工具无法解决前，不升级神经架构。

沿用未锁频/SMT sibling未预留/邻核与缓存不受控限制。新产物在外部pgo-mix-v1目录，日志logs、临时文件tmp；旧数据不改。统一入口：`bash scripts/benchmarks.sh all --suite sqlite-mix`，接入单轮不并入正式统计。
