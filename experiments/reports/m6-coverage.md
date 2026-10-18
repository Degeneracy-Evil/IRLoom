# M6.0：M5 固定覆盖与随机预算参考

仅读取已提交 M5 JSON，不编译或运行基准，也不修改 M5 冻结策略及原报告。旧 lookahead 是预先固定的 balanced coverage，反馈只用于最后选最优，不据反馈改变查询。

## 同一固定四候选子集用于一个家族全部 Shape

每个六候选池共有 C(6,4)=15 个子集。先对每个子集计算逐实例 regret 的几何均值，再对十五个结果作算术平均；不是把每个实例事后最佳子集当成可部署策略。

| 家族 | 轮 | 全 Shape 覆盖 oracle 的子集 | 十五子集平均 GM regret |
| --- | ---: | ---: | ---: |
| matmul | 1 | 3/15 | 1.203305× |
| reduction | 1 | 1/15 | 1.060370× |
| matmul | 2 | 3/15 | 1.203555× |
| reduction | 2 | 1/15 | 1.060537× |

## 随机和分层随机：每实例独立抽取唯一候选

每个策略 100 个固定种子 0–99；分层随机均分两个 A。种子分布不是 100 个独立程序。表格为每个种子的家族 GM regret 再取均值及范围；JSON 保留每种子结果、实际记录成本和 seed=0 的查询。

| 家族 | 轮 | k | 策略 | 平均 GM regret | 种子范围 |
| --- | ---: | ---: | --- | ---: | --- |
| matmul | 1 | 2 | random | 1.8625× | [1.0705, 3.1248] |
| matmul | 1 | 2 | stratified_random | 1.8399× | [1.0167, 3.1673] |
| matmul | 1 | 4 | random | 1.1708× | [1.0000, 1.7029] |
| matmul | 1 | 4 | stratified_random | 1.2305× | [1.0000, 1.8955] |
| matmul | 1 | 6 | random | 1.0000× | [1.0000, 1.0000] |
| matmul | 1 | 6 | stratified_random | 1.0000× | [1.0000, 1.0000] |
| reduction | 1 | 2 | random | 1.2713× | [1.0005, 2.0436] |
| reduction | 1 | 2 | stratified_random | 1.1859× | [1.0000, 1.3469] |
| reduction | 1 | 4 | random | 1.0571× | [1.0000, 1.3007] |
| reduction | 1 | 4 | stratified_random | 1.0729× | [1.0000, 1.3013] |
| reduction | 1 | 6 | random | 1.0000× | [1.0000, 1.0000] |
| reduction | 1 | 6 | stratified_random | 1.0000× | [1.0000, 1.0000] |
| matmul | 2 | 2 | random | 1.8628× | [1.0707, 3.1209] |
| matmul | 2 | 2 | stratified_random | 1.8403× | [1.0168, 3.1669] |
| matmul | 2 | 4 | random | 1.1711× | [1.0000, 1.7044] |
| matmul | 2 | 4 | stratified_random | 1.2307× | [1.0000, 1.8959] |
| matmul | 2 | 6 | random | 1.0000× | [1.0000, 1.0000] |
| matmul | 2 | 6 | stratified_random | 1.0000× | [1.0000, 1.0000] |
| reduction | 2 | 2 | random | 1.2720× | [1.0003, 2.0470] |
| reduction | 2 | 2 | stratified_random | 1.1867× | [1.0000, 1.3489] |
| reduction | 2 | 4 | random | 1.0573× | [1.0000, 1.3031] |
| reduction | 2 | 4 | stratified_random | 1.0730× | [1.0000, 1.3035] |
| reduction | 2 | 6 | random | 1.0000× | [1.0000, 1.0000] |
| reduction | 2 | 6 | stratified_random | 1.0000× | [1.0000, 1.0000] |

随机抽取不查看标签；全池数据只用于事后计分。每个 k 对齐候选编译和原生调用预算，记录墙钟离线求和，未实际在线随机运行。穷举后的最优子集只是回顾性覆盖上界，不用于后续盲测选策略。M5 的固定覆盖成功并非反馈式或学习型优化的证据。

下一步冻结 2×6 池、额外融合模板、train/validation/blind 分组和固定/随机/反馈规则，再采集正式标签。
