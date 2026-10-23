# V1.3.1：固定训练预算与共享根成对分辨率

基线 `d51185d`；执行计划为 `docs/architecture/V1_3_TRAINING_REVIEW.md`。本轮保留 A/B、hierarchical-v3、all-advance-v1、原始计时和旧权重，仅复用 `v13-stage/dataset.json`。

## 冻结协议

`experiments/architecture/training-v1-3-1.json` 固定 seed23、Adam 学习率0.0003、20次更新；主权重总是第20次更新，保存0/1/5/10/20快照。六个模型依次隔离 fixed-final、共享根 margin、独立无连续记忆训练。旧 earliest-best 仅存诊断副本。

沿用2%/0.05ms/2MAD分辨率和绝对q区间；成对比较使用同一产物规范化时间，根参照消去，`gamma=log(Tslow/(Tfast+delta))`，强对必有正gamma。相同产物和near-tie不产生强排序。未采新标签，不运行旧测试矩阵。

## 初步数值审计

旧训练PC1 S0的8个强对全部保留正有效间隔；旧验证matrix S1的5个强对有效间隔均为0。不可将后者推广为全部训练标签失效。100/97ms公式夹具确认旧目标允许同q零损失，新目标对打平产生方向正确的非零梯度，真q仍为零损失/零梯度。完整逐对数据位于外部缓存 `training-reviews/v131-review/data-audit.json`。

## 复现入口

```bash
bash scripts/architecture.sh --training-review all --name my-v131
```

需要保留既有 `benchmark-data/architecture/pilots/v13-stage/dataset.json` 及其引用的终端产物。包装脚本复用本地工具链；源码、权重和IR在忽略的外部缓存，日志在 `logs/architecture`。可用 `--training-review audit` 或 `train` 单独检查。完整记录按名字缓存，协议不匹配拒绝复用；未完成训练使用新名字，不覆盖旧权重。

训练和少量train/validation真实编译重放正在执行；最终结果、成本与限制将在本报告补全。重放只在已标注候选集合中选路，不是blind或搜索性能比较。
