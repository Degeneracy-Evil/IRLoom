# M5-A：既有 M4 候选池的离线决策分析

只读取已提交的 m4-interaction.json；不重新运行编译或基准。全部结果是离线描述，不是在线优化器收益。

贪心策略先查询两个 A 的 B=none，再固定该 A、查询另两个 B，共四个唯一候选。全池 oracle 查询全部六个，仅为延迟下界，不进入同预算排名。

| Case | 轮 | 每个 A 的最优下游 | 贪心最终候选 | 全池最优 | 贪心/全池 | 超过描述性噪声阈值 |
| --- | ---: | --- | --- | --- | ---: | --- |
| matmul-1024x256x256 | 1 | ijk: ijk-tile16 (6.521942 ms); ikj: ikj-none (5.647583 ms) | ikj-none | ikj-none | 1.0000× | 否 |
| matmul-127x65x93 | 1 | ijk: ijk-tile16 (0.061774 ms); ikj: ikj-none (0.076537 ms) | ikj-none | ijk-tile16 | 1.2390× | 是 |
| matmul-32x32x32 | 1 | ijk: ijk-tile16 (0.001942 ms); ikj: ikj-tile16 (0.010123 ms) | ijk-tile16 | ijk-tile16 | 1.0000× | 否 |
| matmul-1024x256x256 | 2 | ijk: ijk-tile16 (6.726766 ms); ikj: ikj-none (5.708394 ms) | ikj-none | ikj-none | 1.0000× | 否 |
| matmul-127x65x93 | 2 | ijk: ijk-tile16 (0.063474 ms); ikj: ikj-none (0.078140 ms) | ikj-none | ijk-tile16 | 1.2311× | 是 |
| matmul-32x32x32 | 2 | ijk: ijk-tile16 (0.001998 ms); ikj: ikj-tile16 (0.010262 ms) | ijk-tile16 | ijk-tile16 | 1.0000× | 否 |

阈值为 max(全池最优中位数的 1%, 贪心与最优两个 MAD 和的 3 倍)，仅用于描述性筛选。M4 的交互 D 与决策损失不同：大尺寸也有强交互，但 greedy 已选到全池最优。

## 冻结的简单静态规则

仅从决策前已知的程序结构和 Shape：matmul 若 N≤32 且 K≤32 保留 ijk，否则选连续 j 的 ikj；行归约保留连续归约维最内层的 ij。零测量的直接实现先用 B=none。该规则用 M4 开发，不能把这三个 Shape 当成未见测试。

在 M4 上，规则为小矩阵选 ijk、大矩阵选 ikj，能选中高层分支；但非整齐 127×65×93 选择 ikj，错过 ijk+tile16，不能预见其后续潜力。它也不会识别小矩阵的轻微 tiling 收益。

M5 后续固定四次候选评估：static 在静态选中的 A 查询全部三个 B，再查询另一 A 的 none；greedy 先测两个 none，再测选中 A 的另两个 B；lookahead 测两个 A 各自的 none 和 tile16。所有选择只用 query 暴露过的延迟；全池最优只在事后计算 regret。规则与 probe 集合在新增数据前冻结，不调参。

## 已知成本（每条编译与计时分开）

| Case | 候选 | 轮 | 编译 s | 15 批计时覆盖的原生时间 s |
| --- | --- | ---: | ---: | ---: |
| matmul-32x32x32 | default | 1 | 0.2248 | 0.2531 |
| matmul-32x32x32 | ijk-tile16 | 1 | 0.2231 | 0.2387 |
| matmul-32x32x32 | ijk-unroll4 | 1 | 0.2250 | 0.2807 |
| matmul-32x32x32 | ikj-none | 1 | 0.2075 | 1.6206 |
| matmul-32x32x32 | ikj-tile16 | 1 | 0.2175 | 1.2440 |
| matmul-32x32x32 | ikj-unroll4 | 1 | 0.2234 | 1.6420 |
| matmul-127x65x93 | default | 1 | 0.2066 | 1.2084 |
| matmul-127x65x93 | ijk-tile16 | 1 | 0.2853 | 0.2373 |
| matmul-127x65x93 | ijk-unroll4 | 1 | 0.2114 | 1.1547 |
| matmul-127x65x93 | ikj-none | 1 | 0.2189 | 0.2940 |
| matmul-127x65x93 | ikj-tile16 | 1 | 0.2538 | 0.9030 |
| matmul-127x65x93 | ikj-unroll4 | 1 | 0.2176 | 0.4135 |
| matmul-1024x256x256 | default | 1 | 0.1962 | 1.9889 |
| matmul-1024x256x256 | ijk-tile16 | 1 | 0.2167 | 0.1960 |
| matmul-1024x256x256 | ijk-unroll4 | 1 | 0.2088 | 2.0014 |
| matmul-1024x256x256 | ikj-none | 1 | 0.2157 | 0.1694 |
| matmul-1024x256x256 | ikj-tile16 | 1 | 0.2157 | 0.6840 |
| matmul-1024x256x256 | ikj-unroll4 | 1 | 0.2339 | 0.2475 |
| matmul-32x32x32 | default | 2 | 0.2210 | 0.2524 |
| matmul-32x32x32 | ijk-tile16 | 2 | 0.2203 | 0.2428 |
| matmul-32x32x32 | ijk-unroll4 | 2 | 0.2226 | 0.2845 |
| matmul-32x32x32 | ikj-none | 2 | 0.2053 | 1.6638 |
| matmul-32x32x32 | ikj-tile16 | 2 | 0.2162 | 1.2613 |
| matmul-32x32x32 | ikj-unroll4 | 2 | 0.2228 | 1.6849 |
| matmul-127x65x93 | default | 2 | 0.2016 | 1.2281 |
| matmul-127x65x93 | ijk-tile16 | 2 | 0.2890 | 0.2411 |
| matmul-127x65x93 | ijk-unroll4 | 2 | 0.2131 | 1.1809 |
| matmul-127x65x93 | ikj-none | 2 | 0.2219 | 0.2976 |
| matmul-127x65x93 | ikj-tile16 | 2 | 0.2567 | 0.9125 |
| matmul-127x65x93 | ikj-unroll4 | 2 | 0.2253 | 0.4179 |
| matmul-1024x256x256 | default | 2 | 0.2010 | 1.9957 |
| matmul-1024x256x256 | ijk-tile16 | 2 | 0.2203 | 0.1997 |
| matmul-1024x256x256 | ijk-unroll4 | 2 | 0.2085 | 2.0046 |
| matmul-1024x256x256 | ikj-none | 2 | 0.2106 | 0.1725 |
| matmul-1024x256x256 | ikj-tile16 | 2 | 0.2125 | 0.6934 |
| matmul-1024x256x256 | ikj-unroll4 | 2 | 0.2342 | 0.2522 |

原生计时覆盖时间 = sum(timings_ms) × iterations / 1000；它不含启动、参考计算、校验、预热或校准。已提交 M4 摘要没有逐候选校准/运行器墙钟明细，不能补造这些值。完整每轮墙钟见 M4 报告；新增 M5 数据将保留按阶段的真实命令耗时，并采用预先固定迭代次数，消除离线模拟读取全池校准次数的依赖。

下一步先建立非 matmul 的真实 A×B 和未见 Shape 数据；仅在存在规划收益时比较规则策略，不训练模型。
