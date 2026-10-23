# V1.3：共享分层候选上下文与校准阶段价值

规划基线 `dd67c52`；旧工具链、模型、结果保持原样。本报告区分结构/数值验收、训练和真实性能，不把夹具当新 benchmark。

## E1：真实 MLIR 上的候选上下文

`v13-e1-context` 由匹配 Polygeist bridge 解析合法 MLIR：同一模块含头部/签名相同、函数体分别 add/mul 的 f/g，保留全部节点，只改变一个 call→callee 边。对种子17/23/47，旧 structured-v2 的 A/B inline q 差均为0；新 hierarchical-v3 可区分，关联 callee/body→角色→读出的梯度非零。只重命名 SSA、交换等价 DictionaryAttr 打印顺序时输出精确不变。该结果只是结构可达性，不说明应该内联哪个函数或已学会性能。

共享计算顺序固定为 typed SparseBlock 局部处理→沿 contains/defines/argument 归属自底向上内容池化→已解析 callee 与 callsite 交互/模块汇总→原 A 阶段 Block 或 B 共享 Block+GRU→角色化候选读出。CFG 环和递归调用不作为树父关系。inline 读取 callsite/caller/callee+body/region/module，unroll 读取 loop/body/所在函数；缺失角色有显式 mask，fusion 仍仅 smoke。每份函数/区域内容汇总复用，记录 summary 和 candidate-read 耗时，不创建全程序 N×N 注意力；树池化工作量还随结构深度增长，未声称任意深图常数代价。

A/B 骨干与记忆公式不变，B 写入的当前状态现在来自内容汇总；一般 lowering 仍只按既有粗锚点继承。`v13-e1-state` 通过真实分支/恢复/RNG/完整与增量编码输出比较、数值检查；`v13-e1-a-fit` 通过 A 集成拟合、跨阶段梯度和完整编译。旧表示开关及旧 checkpoint 继续保留。

## E2：log 代价目标可同时满足

`v13-e2-objective` 使用实际 Torch 目标复现两标签数学诊断：真间距 log(1.09) 时，旧 softplus 在真 q 处仍有梯度，拟合间距0.734565、隐含比2.084575；有限实测 margin 与纯区间回归均拟合0.0861777、比1.09，真值处 loss/梯度为0。相同产物零代价、near-tie 和原真实训练子集也通过可行性检查。这是数值语义验收，不是编译性能。

新 `calibrated-margin-v1` 使用有限 `relu(max(0,|yi-yj|-mi-mj)-sign(yi-yj)*(qi-qj))²`，保留旧 `softplus-v1` 和 `interval-only-v1`。新实验统一将毫秒分辨率分别经 numerator/denominator 传播到 log 比值区间；不用毫秒阈值直接当 log margin。误差区间仍是冻结分辨率规则，不称统计置信概率。q 语义仍是根默认归一化、后续 all-advance-v1，不是 Q*。

## E3：冻结结构覆盖与最小消融

`pilot-v1-3.json` 在新阶段标签前冻结：固定种子按动作种类、所属函数、循环层次和 callee 分层抽样；小池全覆盖，大池轮转抽样而非固定首尾。每个程序采三个真实 S0 选择产生的 S1 前缀，每观察点最多六动作、包含 advance，每程序最多25个完整标签产物。标签产物冻结后，根默认和所有候选做六轮随机/反向配对交错，根默认共用同一参照；初次编译测量也计费但不冒充最后标签。

训练 PC1/lists，验证新 matrix，Perlin 为预先冻结的新测试族，旧 DES/Puzzle 仅诊断。新增程序均使用原始官方 test-suite 源码，Perlin 用原有 SMALL_PROBLEM_SIZE 宏控制成本，不改算法；来源/许可证见 `programs-v1-3.json`。methcall 前端崩溃和 Treesort 的导入错误保留于 v13-import-1/2，不通过改算法掩盖。没有按模型效果挑程序或调低噪声阈值。

最小矩阵：同旧表示比较旧/新损失(A/B)，同新损失比较旧/分层表示(A/B)，最后在分层输入上比较独立训练无连续记忆 A/B；仅一个预先固定种子23。冻结断连为补充，raw/guarded 控制器与预算沿用 V1.2。新增摘要/读出带来参数差异，不能直接将共同表示收益算连续记忆收益。
