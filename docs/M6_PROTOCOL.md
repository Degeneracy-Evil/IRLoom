# M6 冻结实验协议（正式标签采集前）

本文件与实现一起提交后，从干净工作区执行正式采集。M6.0 只读分析已经完成；本协议对应 M6.1 和 M6.2 pilot。不得按 blind 结果修改动作、规则、特征、划分或超参数网格。

## 候选与语义

每个 case 固定12个候选：A0/A1 × none/tile4/tile8/tile16/tile32/unroll4。tile 是 affine-loop-tile 的统一 tile-size，unroll 是 affine-loop-unroll 的 unroll-factor=4；完整参数在 experiments/m6.json。所有失败保留，不允许事后删除不利动作。

- matmul：高层 ijk/ikj，沿用 M4/M5 合法调度。
- reduction：高层 ij/ji，沿用 M5 合法调度。
- pointwise_chain：C=A*B+A，两段逐元素 linalg.generic；A0 separate 保持两段，A1 fused 使用 MLIR 18 的 linalg-fuse-elementwise-ops。随后 one-shot-bufferize、buffer-deallocation。实际功能预验收确认高层两段变一段，两路缓冲化后均保留一个 alloc/dealloc 和输出 copy，这些 kernel 内部操作都计时。

功能 pilot 观察过 8×13 融合链；它只验证编译和数值，不参与最终拟合或调参。整模板留出指正式监督标签留出，并非从未接触模板语义或外部保密评审。融合/缓冲化依据 [LLVM release/18.x 融合案例](https://github.com/llvm/llvm-project/blob/release/18.x/mlir/test/Dialect/Linalg/fusion-elementwise-ops.mlir) 与 [one-shot 案例](https://github.com/llvm/llvm-project/blob/release/18.x/mlir/test/Dialect/Bufferization/Transforms/one-shot-bufferize.mlir)，并已用本地18.1.3实际执行。

## 输入与分组

下表是预先列举的有限静态 Shape 表，不声称随机采样总体。所有正式输入组合不同于 M4/M5。case 是独立输入单位；同 case 的全部候选和两次重复始终处于同 split。

| Case | Split | 每计时批 kernel 次数 |
| --- | --- | ---: |
| matmul-40x56x32 | train | 512 |
| matmul-64x64x48 | train | 256 |
| matmul-80x48x80 | train | 256 |
| matmul-96x96x64 | train | 128 |
| matmul-56x80x48 | validation | 256 |
| matmul-112x48x64 | validation | 256 |
| matmul-72x104x40 | blind | 256 |
| matmul-128x40x80 | blind | 256 |
| reduction-48x96 | train | 4096 |
| reduction-96x64 | train | 4096 |
| reduction-64x160 | train | 2048 |
| reduction-160x64 | train | 2048 |
| reduction-80x128 | validation | 2048 |
| reduction-96x192 | validation | 2048 |
| reduction-112x96 | blind | 2048 |
| reduction-160x160 | blind | 1024 |
| pointwise_chain-32x64 | blind | 8192 |
| pointwise_chain-65x97 | blind | 4096 |
| pointwise_chain-128x128 | blind | 2048 |
| pointwise_chain-257x129 | blind | 1024 |

训练8个输入（两个模板各4），验证4个（各2），blind8个：两个已见模板各2个未见 Shape，另加 pointwise_chain 整模板4个。候选标签不是独立程序样本，两批也不是新增训练实例。

## 实际执行与预算

单入口 bash scripts/phase0.sh m6 连续两批独立重编译（order_seed=1/17）；每批240个候选，四个正确性种子0/1/7/19，输入计时seed=1，10轮随机交错计时。沿用独立 double 参考、NaN 检查、哨兵与 rtol=atol=1e-4，禁用重关联/浮点收缩。单 CPU 亲和、单线程、同缓冲热态，频率与共享负载不可控。

预列固定 iterations，禁止用完整池校准帮助策略；min_batch_ms 对这些 case 不生效，也不保证每个快路径批次达到10ms。保留短批样本及噪声，不按结果增加迭代次数。报告中位数/MAD/范围；小于 max(3%, 三倍两候选最大相对MAD) 的成对差异不解释为稳定增益。每次 timing 进程先一次校验调用、3次预热，再计 iterations 次。k=2/4/6 时均 k 次候选编译、4k correctness 进程、10k timing 进程、零校准；原生调用数 k×[4+10×(iterations+4)]。真实编译/进程墙钟和原生计时单列；同调用预算不等于同时间预算。

所有策略均离线回放完整池。query 才公开该候选实测中位数；没有查询的 latency 不能参与决策。oracle 只事后计分。compile_seconds 沿用原 schema，包含 snapshot pipeline、翻译、codegen/link 和审计反汇编墙钟；compiler_processes 计数排除反汇编。反汇编不是决策特征。记录额外决策/推理耗时，训练费用只收一次，不混入每次查询。

## 冻结策略

候选稳定排序为 A0 的 none/tile4/tile8/tile16/tile32/unroll4，然后 A1 同序；所有并列按稳定顺序。k 是下面计划的前 k 个唯一候选。

- fixed_m5：none、tile16、unroll4，各动作先A0再A1，保持旧 balanced coverage。
- fixed_spread：tile8、tile32、none，各动作先A0再A1，独立预设覆盖参考。
- static：优选归约A0，matmul仅 N<=32 且 K<=32 时A0，否则A1；链优选A1。顺序 preferred-none/tile16/unroll4、other-none、preferred-tile8/tile32。规则不在本轮标签上调优。
- random：随机无放回 k 个。stratified_random：每高分支 k/2 个。32个种子0..31，每 case 使用 seed+1000×配置索引；先计算每 seed 跨 case 的 regret GM，再报告均值和范围。种子不算独立程序。
- greedy：先 A0-none/A1-none，再固定默认最快分支，依序 tile16/unroll4/tile8/tile32/tile4。
- adaptive：先两个 none。每次分支分数为该分支已测最小延迟 / (1+0.5/sqrt(该分支已查询数))，选最小且有剩余动作的分支。若该分支已测 tile16 相对 none 改善超过1.05倍，剩余顺序 tile16/tile8/tile32/tile4/unroll4；否则 tile16/unroll4/tile8/tile32/tile4。分支与后续顺序均依赖实际反馈，不把固定均衡覆盖称为自适应。
- ridge：固定模型预测 log latency 排序。ridge_feedback：先两个 none，再以同分支已测残差之和/(1+残差数)修正剩余预测，选最小。只修正当前 case 的预测，不更新权重。

## 透明模型与信息边界

仅 NumPy ridge，没有神经网络依赖。20个预定特征、21个参数含截距，精确顺序和公式见 irloom/value.py。来自静态原始结构/Shape、计划动作、高层预期最内层访问范围和 stride、tile及余数、操作量/字节量/强度、工作量与tile交互，以及相对于目标L1/L2容量的字节量。cache 只来自静态机器信息，同一主机不能识别硬件迁移能力。阶段数和是否归约在训练中缺乏变化，整模板 OOD 只能作为探索性外推。

特征不使用最终 affine/LLVM 快照、反汇编、实际代码重复标签、未来合法路径的实测延迟；无需额外 Lowering，特征编译费用为零，生成特征的时间包含在训练/预测时间中。只有已查询路径可作为反馈特征。

两批同 case/candidate 的20个计时合并取中位数作为训练/验证标签，得到96个训练候选标签，非96个独立输入。训练标准化，常量特征 scale=1，截距不惩罚；目标为 log(ms/kernel)。alpha=[0.1,1,10]，只按验证集 ridge k=4 的逐 case regret GM 选择，平局按网格顺序。选定后仍只拟合 train，不把 validation 合并训练。模型参数/训练日志先保存，之后才读取 blind 延迟进行首次评分；报告 RMSE 仅为辅助，决策 regret 优先。训练秒数、依赖版本、划分、标准化参数与权重导出 m6-model.json，本机详细日志在 logs/。

## M6.3 触发门槛与边界

本轮预定工程门槛为至少20个独立训练输入、至少3个训练模板，且两批均表现出相同预算下超过强无学习基线的稳定决策收益；此门槛只是最低工程筛选，不是统计充分性证明。当前只有8个训练输入/两个训练模板，故不执行 M6.3，不训练 MLP/图网络/Attention，也不声称进行了参数量匹配的神经消融。若静态/随机已逼近 oracle 或线性模型失败，保留负结果。

完整IR、二进制、原始命令仅放 tmp/ 与 logs/；提交可复核原始计时摘要、模型、成本、分布、查询样例和人工解读。不计算hash，不写单元测试；验收使用实际编译、数值检查、两次测量和数据审计。

## 正式验收记录

冻结实现提交 `7f0b1e0`。两批均从干净工作区采集，各240/240正确、2400个计时；完整流程190.03 s。实际审计与解读见 [m6-interpretation.md](../experiments/reports/m6-interpretation.md)。本节对成本文字作实现一致性澄清，未改变策略、模型、划分或数据。
