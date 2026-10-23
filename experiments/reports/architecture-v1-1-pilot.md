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
