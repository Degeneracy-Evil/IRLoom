# Architecture V1.1：结构表达与训练修订

基线为 `96aac67` 的 V1 首轮审查；旧 pilot 和报告原样保留。本报告区分实现、集成验收和性能证据。

## R1：有序、有类型的结构输入

C++ bridge 通过 MLIR API 导出 operand/result/block argument/CFG successor 及嵌套位置。边格式追加端口；原方向、reverse、self、摘要交换使用不同关系，多重边不合并。`structured-v2` 稀疏 Block 的 attention key 和 value 实际消费关系和位置。节点保留规范类型及有序属性字节，以带位置权重的字节序列池化取代无序平均；不截断长属性，不使用源码行号或符号名字训练。未知属性仍可编码，但这不是所有 dialect 的完整语义解析，也不保证有限向量无碰撞。旧 `legacy-v1` 模式和旧属性子集保留，可加载旧权重，并在新数据上独立训练表示对照。

`v11-r1-structure-2` 实际解析/提取减法、除法、比较和 affine map 夹具：SSA 重命名保持结构；交换 operand 或 affine 结果顺序可区分。未建立单元测试工程。第一版夹具缺少 module 属性 dialect 前缀，被 verifier 正确拒绝，修正夹具后通过。

受控变换后若在同阶段重新观察，可靠的 operation 来源关系组合进入 Observation；一旦经过一般 lowering，仍仅使用 module/function/body 锚点，不把内部对应冒充已知。Observation 单独报告按粒度 known/unknown、账本关系数和 lowering 数；A/B 另报实际读取的节点/记录数，B 报刷新/失效数。检查点同步保存来源游标，避免跨分支误用账本。当前阶段决策后通常立即 lowering，因此细粒度受控映射虽可在同阶段接通，常规跨阶段仍主要是粗锚点；这是已知覆盖限制。

`v11-r1-state` 通过 B 的真实编译分支、张量隔离、RNG、恢复重放、完整/增量输出比较及独立数值检查；`v11-r1-a-fit` 通过 A 的跨阶段梯度和集成拟合。所有 SDK/前端构建复用固定版本缓存，新增代码没有安装系统依赖。

复现：

```sh
bash scripts/architecture.sh --structure-audit
bash scripts/architecture.sh --model B --state-audit --multi-function
bash scripts/architecture.sh --model A --fit-report "$PWD/benchmark-data/architecture/runs/n1-audit/report.json"
```

日志在 `logs/architecture/`，原始图、状态统计和权重在外部 `benchmark-data/architecture/runs/`。R2/R3 的新配置、覆盖与结果在完成后追加；本阶段不声称结构修正带来性能提升。

## R2：冻结的数据与训练配方

`experiments/architecture/pilot-v1-1.json` 在新标签采集前提交为 `2b38f09`。训练使用 LLU（原始 `-i 300` 输入）、PC1、完整 Stanford Bubblesort；验证使用 Towers；Queens 与 Puzzle 同属留出的 backtracking 族，仅计为一个测试族。PC1 是旧实验的验证程序，本轮明确转为训练，绝不称其为留出。完整源文件来自相同固定官方 test-suite，不复制 kernel；Stanford 程序共用历史声明形式，不能将它们视为互不相关的大型项目。

首轮兼容性探测的 Bubblesort/Queens/Towers/Puzzle 均有 S0 inline 与 S1 unroll2 候选，并通过完整原始输出 oracle。ReedSolomon 导入图更大（S0 1627 节点），但默认 stdout 缺乏有用独立结果，未纳入性能结论；fbench S0 无非 advance 候选，暂不选作多阶段主用例。Fhourstones 3.1 完整 SearchGame.c 的导入在 VisitBinaryOperator 触发 `rhs.val` 断言，保留日志 `v11-probe-6`。兼容性筛选依据是导入/动作可达性，不是加速比。

训练去掉单一 `next(...)` 训练族限制，按程序、输入、机器/亲和性、宏与完整编译语义配置分组。每组自己的 default 为相对延迟参照，只在组内排序，各组等权，不按路径数增加权重。失败路径不生成成功标签、缺默认组明确拒绝。排序跳过直接逐字节相同 IR 或目标文件的路径；其余差异须超过预先固定的 max(2% 延迟、0.05 ms、2×MAD) 才生成强胜负标签。未计算产物 hash。重复/near-tie 路径仍提供组内 value/resource 监督，不能冒充新增独立程序或可靠动作偏好。

所有模型使用相同数据、种子和 20 epoch；A/B、旧表示 A/B、独立训练的 current-only/B-current-only、容量匹配 B 同时训练。测试不参与 epoch 选择。损失仍是完整 continuation 的 policy log 概率排序和末端 value/resource 监督，beam 中间分支按 policy 排名，不宣传为已训练的前缀未来价值搜索器。后续覆盖表报告真实合法候选、执行、有效性能成对标签与 action/block 梯度；fusion 仅原生 smoke 覆盖，不属本轮 Polygeist 正式训练动作。

R2 实际完成：22 条采集轨迹、七套模型训练、60 组冻结单路径运行均通过独立输出比较，没有采集失败。标签采集 17.981 秒，训练合计 196.014 秒。训练 3 组、验证 1 组；全部七套冻结 checkpoint 在新进程中复现选中 epoch 的验证损失（误差 <1e-6），inline/unroll2 头梯度均非零，fuse 梯度为零且明确不纳入已训练动作。A 的三个阶段 Block、B 的共享 Block 都有梯度。

| 组 | split | 路径 | S0/S1 有选择观察 | 有效成对标签 | 相同产物跳过 | near-tie 跳过 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| llu_v11 | train | 5 | 0 / 5 | 0 | 1 | 9 |
| pc1 | train | 5 | 5 / 5 | 7 | 1 | 2 |
| bubblesort | train | 5 | 5 / 5 | 0 | 0 | 10 |
| towers | validation | 5 | 5 / 5 | 0 | 1 | 9 |

虽然已实现多程序、两阶段学习，强成对排序目前仅来自 PC1；LLU/Bubblesort/Towers 的可分辨差异不足，其贡献主要为 value/resource 监督。这没有解决广泛动作偏好训练不足的问题，不能把 20 epoch 宣称为足够训练。程序有多个输入组时，先组内平均再按程序等权；本轮每程序只有一个输入组。

更大来源记录：SQLite 3.50.4 完整生成库翻译单元（复用已有官方源码及生成缓存，`SQLITE_THREADSAFE=0`）在 `getMLIRType` 触发 `types.size()` 断言，日志为 `v11-large-sqlite-import`；这只是大型导入诊断，不是完整可执行 Benchmark。没有修改源文件或关闭 verifier。
