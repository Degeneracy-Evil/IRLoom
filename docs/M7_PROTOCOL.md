# M7 冻结协议：计算图上下文与信息窗口

本文件与实现/配置一同提交后，才执行正式矩阵。不按正式结果修改上下文、Shape、动作、信息窗口、规则、划分或噪声阈值。M7.0的8×8功能预验收见 [m7-feasibility.md](../experiments/reports/m7-feasibility.md)，另用8/13的小配置验证整个两批报告流程；它们不进入正式标签。

## 假设、局部匹配与动作范围

目标局部区域是D[i,j]=A[i,j]*B[i,j]，f32、行主、A/B只读且互不别名，M×N。原始producer文本由唯一函数target_region生成；严格窗口只含该producer及其输入/输出类型与布局，不包含users、其它算子或全函数alloc。相同Shape的三个上下文具有逐字相同局部输入；输入数值也相同。

- single：C=D+A。
- transpose：Y=D+A；C[i,j]=Y[i,j]+D[j,i]，只接受M=N。
- row_reduce：Y=D+A；Z[i]=sum_j D[i,j]；C[i,j]=Y[i,j]+Z[i]。

每个上下文的六条候选必须等价输出；不同上下文工作量不同，只比较各自候选的相对效应。高层动作separate保持图、fused执行现成linalg-fuse-elementwise-ops。实际MLIR18会保留多消费者D，只融合其它合法边；不能把fused标记当作D已经全内联。bufferization后alloc数分别1→1、2→2、3→2。

低层none/tile8/tile32，沿用affine-loop-tile统一tile-size作用于图中全部可变换循环nest；后续canonicalize/cse和LLVM后端一致。两级动作是完整程序pipeline，**不是只修改D的局部Pass**。上下文若改变最佳路径，首先证明本试点全程序候选的上下文依赖，不能独占归因为仅作用于D的局部因果效应。消费者转置/归约本身的工作也会影响低层选择；如需要隔离这种效应，应另行预注册目标区域限定变换，不能补写为本轮已有证据。

所有临时申请、释放与输出copy均在kernel内，计入延迟。不写自定义Pass，不重写编译后端。

## 正式矩阵与测量

experiments/m7.json：三上下文×两个规模（64×64、513×513）×两高层×三低层，共36条。小规模每计时批4096次kernel，大规模128次。尺寸预列，两个规模用于容量/边界可行性，不声称随机采样总体。两批独立重编译，order_seed=1/17，不重复运行旧M0–M6矩阵或重写其结论。

每条先用0/1/7/19四种子独立double参考校验，NaN检测未写输出、哨兵检测越界，rtol=atol=1e-4；归约保持j递增，禁用重关联/浮点收缩。参考在timer外，每行参考和只算一次。计时输入seed=1，三个预热，再计固定iterations，15轮随机交错。沿用单CPU亲和、单线程、热态缓冲；频率/邻核负载不可控、无PMU。

不执行全池校准；保留不足10ms的短批，不按结果重设迭代。同一case候选相同调用次数，候选编译和进程墙钟另存，不把相同次数解释为相同墙钟预算。

## M7.1：路径效果与噪声

逐case输出六条路径median/MAD/min/max及原始样本。基准separate-none，另外逐低层比较separate/fused、逐高层比较none/tile。可分辨方向必须超过max(3%,三倍两候选最大相对MAD)，并且按共同计时轮索引重采样1000次的探索性95%速度比区间不跨1。上下文反转要求同Shape同一动作比较，在不同上下文得到相反且可分辨方向；两批都重现才作为重复信号。区间不校正多重比较，也不代表跨时段独立实验。

逐Shape给出各context最佳路径，以及在三个context共用同一个动作的事后最小GM regret。它使用全池标签，只是严格匹配局部信息下的描述性共享选择下界，不是可部署策略。赢家不同但差距小于噪声阈值，不声称稳定反转。保持所有候选，不删除不利动作。

## M7.2：同一低层边界的信息比较

高层分支分别固定为separate/fused，决定B时只允许读取high和affine边界，禁止读取transformed/LLVM/反汇编、代码重复标签和测试case未查询延迟。高层选择不由这个低层pilot学习，因此低层oracle也只在固定高层的三个B中计算，不能冒充联合六候选oracle。

L0：第一个外层affine循环nest的9项统计（affine.for/load/store、arith.mulf/addf、memref.alloc/dealloc/copy、反向二维load），加共同可见的Shape/f32/行主/固定机器。窗口不含其它循环或全局分配。separate的该窗口在三个context做SSA重命名后逐字相同；fused-single的局部body已混入consumer，因此此时局部匹配已不成立，必须披露。

L1：L0 + 当前完整affine函数的同样9项统计，看到所有循环、分配/释放和copy。这里是完整IR的廉价统计，不声称统计向量等于完整IR内容。

L2：L1 + 原始high图的四项上游信息：D消费者数、generic数、有无转置consumer、有无归约consumer。来源是实际保存的高层快照，非标签或手工赢家注释。这是在同一affine决策点添加真实上游信息；不用初始高层处当前/历史相同的别名伪装历史消融。

窗口提取器仅面向本试点的已知静态MLIR打印结构，不能声称通用Operation级N:M溯源。九项统计和SSA窗口均在collect测量前保存。高层/affine读取与统计时间记录到context_features.seconds，L1/L2成本包含所需较简单窗口；无需额外编译。必要的固定高层prefix编译包含在每个选中候选的完整compile_seconds中。

透明查表不训练神经网络：先按可见签名匹配训练case，L0→L1→L2逐步细化，未知较丰富签名保留较简单窗口分组；再选择与测试Shape的log维度欧氏距离最小的训练组，按训练低层动作的GM延迟排序取最低（平局none/tile8/tile32）。记录参与匹配的训练case。所有表示可用的训练标签相同，特征差异只改变匹配分组，不借助测试成绩挑表示或动作。

预定两种完整case划分，重复计时始终和case一起：

- leave_shape_out：同一规模的三个完整context全留出，用另一规模的三个训练case；只测试同图新规模，不能宣称未见图泛化。
- leave_context_out：同一种上下文的两个规模全留出，用另两种上下文的四个训练case；新签名回退，不因为没有匹配就偷用留出标签。

两批同case同candidate的30次计时合并作训练中位标签；每个测试case全部候选和重复都不出现在自己的训练集。按各批原始标签分别计分，批次不算独立训练样本。这是六个输入的分组交叉验证pilot，没有外部保密blind，也没有调参步骤。

无学习参考为fixed_none、fixed_tile8、fixed_tile32；廉价current_rule在当前affine存在转置load时选tile32，否则none。规则在正式数据前固定，不按结果修补。所有方法测试时只查询一个B：一次完整候选编译、四次correctness、十五次timing、零calibration，原生调用4+15×(iterations+4)。特征提取、查表/排序开销另外实测，训练候选池标签获取费用也必须披露。为各表示重复拟合/评估的离线操作不能当成真实在线执行，反事实费用不能相加成实际工作流墙钟。

## 交付与停止

单入口bash scripts/phase0.sh m7：两批独立collect、路径相对效应、噪声/上下文对照、信息窗口分组pilot和主要IR差异。m7-report可只用原始JSONL重建。完整IR/二进制/暂存文件放tmp/，日志和完整原始记录放logs/；提交m7-findings/results/lineage及单独人工m7-interpretation。沿用不计算hash、不新增单元测试，实际编译、数值、数据审计与报告重建为验收。

若廉价当前IR统计或规则已解释上下文，或跨Shape效果不稳定，不增加神经网络；即使存在全局上下文增益，也不推出上游历史、Attention、持续状态有效。六个输入不足以评估泛化，先如实交付信号与限制，不通过重复堆Shape凑旧门槛。
