# M4 跨阶段决策交互实验

本实验比较一个 matmul 程序家族中的两个真实编译决策点，不训练模型。

## 受控设计与测量

A：linalg generic 的迭代顺序 ijk / ikj，均由命名 matmul 经 Transform Dialect 泛化及合法 interchange 得到。B：affine 的 none / tile16 / innermost unroll4。每个 Shape 为 2×3 因子设计；C++ ikj 为单独性能参考，不进入交互计算。

A 的两种顺序保持每个 C[i,j] 的 k 递增加法顺序；所有路径禁用 fast-math 与浮点收缩。C++ 参考为普通 row-major i-k-j 三重循环，Clang++ -O3 -march=native -ffp-contract=off，无 BLAS、restrict、手工向量化或不同布局。它共享 ABI、输入、输出清零和计时运行器，独立 double 参考负责正确性，不把它称为自动调优或最先进基线。

每轮独立重新编译，固定相同输入种子，分别使用 order_seed=1/17 随机交错；各候选 15 次计时，先用 seeds 0/1/7/19 校验。校准至少 10 ms，同一 Shape 所有候选与参考使用同一批量次数。3 次预热后重复同一缓冲区；计时含输出清零，排除分配、参考计算与校验。大工作集不会被人为逐次驱逐，属于稳态复用，不是冷缓存或实测带宽。

- 批次1：`logs/20261018T101246162932Z/records.jsonl`；构建 `97138e4dd18242d48e80f6940dff138ee55e199a`；工作区干净；通过 21/21；墙钟 45.12 s；编译子进程合计 4.40 s。
- 批次2：`logs/20261018T101331335316Z/records.jsonl`；构建 `97138e4dd18242d48e80f6940dff138ee55e199a`；工作区干净；通过 21/21；墙钟 45.64 s；编译子进程合计 4.40 s。

目标 cache（sysfs，同一逻辑 CPU）：

```json
[
  {
    "cpu": 0,
    "level": "1",
    "type": "Data",
    "size": "48K",
    "shared_cpu_list": "0,120"
  },
  {
    "cpu": 0,
    "level": "1",
    "type": "Instruction",
    "size": "64K",
    "shared_cpu_list": "0,120"
  },
  {
    "cpu": 0,
    "level": "2",
    "type": "Unified",
    "size": "2048K",
    "shared_cpu_list": "0,120"
  },
  {
    "cpu": 0,
    "level": "3",
    "type": "Unified",
    "size": "516096K",
    "shared_cpu_list": "0-239"
  }
]
```

| Shape (M,N,K) | A+B+C 工作集 |
| --- | ---: |
| 32,32,32 | 12.00 KiB |
| 127,65,93 | 102.00 KiB |
| 1024,256,256 | 2304.00 KiB |

以上只计热态 kernel 的输入输出，不含运行器的 double 校验缓冲区、哨兵和代码；超过 cache 容量仅是静态工作集条件，不能替代 PMU miss 或内存带宽实测。

## 延迟与参考

| Case | 候选 | 轮 | median ms | MAD% | 相对本轮 default | 编译 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| matmul-1024x256x256 | cpp-ikj | 1 | 5.620770 | 0.05 | 11.817× | 0.129 |
| matmul-1024x256x256 | default | 1 | 66.421220 | 0.13 | 1.000× | 0.196 |
| matmul-1024x256x256 | ijk-tile16 | 1 | 6.521942 | 0.28 | 10.184× | 0.217 |
| matmul-1024x256x256 | ijk-unroll4 | 1 | 66.719781 | 0.12 | 0.996× | 0.209 |
| matmul-1024x256x256 | ikj-none | 1 | 5.647583 | 0.12 | 11.761× | 0.216 |
| matmul-1024x256x256 | ikj-tile16 | 1 | 22.788754 | 0.20 | 2.915× | 0.216 |
| matmul-1024x256x256 | ikj-unroll4 | 1 | 8.249410 | 0.11 | 8.052× | 0.234 |
| matmul-127x65x93 | cpp-ikj | 1 | 0.077908 | 0.14 | 4.040× | 0.132 |
| matmul-127x65x93 | default | 1 | 0.314736 | 0.70 | 1.000× | 0.207 |
| matmul-127x65x93 | ijk-tile16 | 1 | 0.061774 | 0.17 | 5.095× | 0.285 |
| matmul-127x65x93 | ijk-unroll4 | 1 | 0.300604 | 0.28 | 1.047× | 0.211 |
| matmul-127x65x93 | ikj-none | 1 | 0.076537 | 0.10 | 4.112× | 0.219 |
| matmul-127x65x93 | ikj-tile16 | 1 | 0.235122 | 0.06 | 1.339× | 0.254 |
| matmul-127x65x93 | ikj-unroll4 | 1 | 0.107409 | 0.23 | 2.930× | 0.218 |
| matmul-32x32x32 | cpp-ikj | 1 | 0.010802 | 0.18 | 0.187× | 0.136 |
| matmul-32x32x32 | default | 1 | 0.002015 | 0.28 | 1.000× | 0.225 |
| matmul-32x32x32 | ijk-tile16 | 1 | 0.001942 | 0.11 | 1.038× | 0.223 |
| matmul-32x32x32 | ijk-unroll4 | 1 | 0.002280 | 0.28 | 0.884× | 0.225 |
| matmul-32x32x32 | ikj-none | 1 | 0.013175 | 0.54 | 0.153× | 0.207 |
| matmul-32x32x32 | ikj-tile16 | 1 | 0.010123 | 0.20 | 0.199× | 0.218 |
| matmul-32x32x32 | ikj-unroll4 | 1 | 0.013352 | 0.23 | 0.151× | 0.223 |
| matmul-1024x256x256 | cpp-ikj | 2 | 5.639424 | 0.17 | 11.801× | 0.130 |
| matmul-1024x256x256 | default | 2 | 66.548163 | 0.08 | 1.000× | 0.201 |
| matmul-1024x256x256 | ijk-tile16 | 2 | 6.726766 | 1.49 | 9.893× | 0.220 |
| matmul-1024x256x256 | ijk-unroll4 | 2 | 66.856390 | 0.26 | 0.995× | 0.209 |
| matmul-1024x256x256 | ikj-none | 2 | 5.708394 | 0.92 | 11.658× | 0.211 |
| matmul-1024x256x256 | ikj-tile16 | 2 | 23.156087 | 0.49 | 2.874× | 0.213 |
| matmul-1024x256x256 | ikj-unroll4 | 2 | 8.474589 | 0.16 | 7.853× | 0.234 |
| matmul-127x65x93 | cpp-ikj | 2 | 0.079329 | 1.18 | 4.029× | 0.135 |
| matmul-127x65x93 | default | 2 | 0.319626 | 0.28 | 1.000× | 0.202 |
| matmul-127x65x93 | ijk-tile16 | 2 | 0.063474 | 0.43 | 5.036× | 0.289 |
| matmul-127x65x93 | ijk-unroll4 | 2 | 0.306482 | 0.42 | 1.043× | 0.213 |
| matmul-127x65x93 | ikj-none | 2 | 0.078140 | 0.56 | 4.090× | 0.222 |
| matmul-127x65x93 | ikj-tile16 | 2 | 0.238186 | 0.23 | 1.342× | 0.257 |
| matmul-127x65x93 | ikj-unroll4 | 2 | 0.109410 | 0.32 | 2.921× | 0.225 |
| matmul-32x32x32 | cpp-ikj | 2 | 0.011012 | 0.83 | 0.184× | 0.133 |
| matmul-32x32x32 | default | 2 | 0.002025 | 0.66 | 1.000× | 0.221 |
| matmul-32x32x32 | ijk-tile16 | 2 | 0.001998 | 0.77 | 1.014× | 0.220 |
| matmul-32x32x32 | ijk-unroll4 | 2 | 0.002324 | 0.41 | 0.872× | 0.223 |
| matmul-32x32x32 | ikj-none | 2 | 0.013349 | 0.70 | 0.152× | 0.205 |
| matmul-32x32x32 | ikj-tile16 | 2 | 0.010262 | 0.30 | 0.197× | 0.216 |
| matmul-32x32x32 | ikj-unroll4 | 2 | 0.013583 | 0.22 | 0.149× | 0.223 |

## 高层选择如何改变低层收益

`D = log[T(ikj,B)/T(ikj,none)] - log[T(ijk,B)/T(ijk,none)]`，T 为批次内中位延迟。D=0 表示在对数延迟尺度上没有交互；D>0 表示 B 在 ikj 下的相对效果更差。下表 speedup 分别以该 A 的 none 为基线，大于 1 为改善。

95% 区间按同一测量轮次联合重采样四个 cell，再重算中位数和 D，2000 次、固定种子。这是探索性区间，未作多重比较校正，不消除共享负载、频率、ASLR 或跨轮依赖。符号反转仅在两个 A 的低层 log-effect 区间分别完全位于零两侧时标记。

| Case | B | 轮 | ijk 下 speedup | ikj 下 speedup | D | 探索性 95% 区间 | 收益符号反转 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| matmul-1024x256x256 | tile16 | 1 | 10.184× | 0.248× | 3.7159 | [3.7097, 3.7193] | 是 |
| matmul-1024x256x256 | unroll4 | 1 | 0.996× | 0.685× | 0.3744 | [0.3699, 0.3765] | 否 |
| matmul-127x65x93 | tile16 | 1 | 5.095× | 0.326× | 2.7506 | [2.7419, 2.7579] | 是 |
| matmul-127x65x93 | unroll4 | 1 | 1.047× | 0.713× | 0.3848 | [0.3754, 0.3942] | 是 |
| matmul-32x32x32 | tile16 | 1 | 1.038× | 1.301× | -0.2266 | [-0.2325, -0.2209] | 否 |
| matmul-32x32x32 | unroll4 | 1 | 0.884× | 0.987× | -0.1100 | [-0.1190, -0.1047] | 否 |
| matmul-1024x256x256 | tile16 | 2 | 9.893× | 0.247× | 3.6922 | [3.6565, 3.7280] | 是 |
| matmul-1024x256x256 | unroll4 | 2 | 0.995× | 0.674× | 0.3905 | [0.3605, 0.4006] | 否 |
| matmul-127x65x93 | tile16 | 2 | 5.036× | 0.328× | 2.7311 | [2.7238, 2.7707] | 是 |
| matmul-127x65x93 | unroll4 | 2 | 1.043× | 0.714× | 0.3786 | [0.3591, 0.4040] | 是 |
| matmul-32x32x32 | tile16 | 2 | 1.014× | 1.301× | -0.2492 | [-0.2670, -0.2209] | 否 |
| matmul-32x32x32 | unroll4 | 2 | 0.872× | 0.983× | -0.1200 | [-0.1402, -0.1058] | 否 |

## 跨批次判读

- matmul-1024x256x256 / tile16：各批次区间同向排除零，观察到重复的路径交互；各批次均出现收益符号反转。
- matmul-1024x256x256 / unroll4：各批次区间同向排除零，观察到重复的路径交互；不宣称重复的收益符号反转。
- matmul-127x65x93 / tile16：各批次区间同向排除零，观察到重复的路径交互；各批次均出现收益符号反转。
- matmul-127x65x93 / unroll4：各批次区间同向排除零，观察到重复的路径交互；各批次均出现收益符号反转。
- matmul-32x32x32 / tile16：各批次区间同向排除零，观察到重复的路径交互；不宣称重复的收益符号反转。
- matmul-32x32x32 / unroll4：各批次区间同向排除零，观察到重复的路径交互；不宣称重复的收益符号反转。

## 当前 IR 能解释什么

B 决策点是 affine 快照；上游 ijk/ikj 的区别仍显式体现在当前循环顺序和 B 的下标访问中。下表直接读取当前 affine IR，不使用真实延迟或未来 LLVM IR。

| Case | A | 最内层 IV | B 矩阵访问 | 内层连续访问 B |
| --- | --- | --- | --- | --- |
| matmul-32x32x32 | ijk | %arg5 | %arg5, %arg4 | 否 |
| matmul-32x32x32 | ikj | %arg5 | %arg4, %arg5 | 是 |
| matmul-127x65x93 | ijk | %arg5 | %arg5, %arg4 | 否 |
| matmul-127x65x93 | ikj | %arg5 | %arg4, %arg5 | 是 |
| matmul-1024x256x256 | ijk | %arg5 | %arg5, %arg4 | 否 |
| matmul-1024x256x256 | ikj | %arg5 | %arg4, %arg5 | 是 |

当前 IR 的循环/访问特征足以区分高层选择；静态启发式可以据此识别连续访问和 unroll 所作用的维度。本实验未验证这些启发式能预测全部 tile/unroll 的收益或精确排名。即使存在交互，也没有证明历史信息具有当前 IR 之外的增量价值。未开展 B0/B1/B2、反馈回溯、模型训练或候选选择策略比较，不进入神经架构阶段。

代表性高层/中层 diff、实际优化命令和 kernel 函数反汇编见 [m4-lineage.md](m4-lineage.md)。快照级 parent 链保留；linalg 展开、tiling、unroll、CSE 和 LLVM 优化仍没有 Operation 级 N:M 来源。指令片段只提供定性证据，不能单独断言 cache、端口、寄存器或向量化是性能原因。

## 原默认路径与产物重复

命名 matmul 的原默认路径与本次 generalize + identity-interchange 的 default 直接比较目标反汇编（不计算 hash）：

```json
[
  {
    "case_id": "matmul-32x32x32",
    "identical_disassembly": true,
    "compile_seconds": 0.2185605401173234,
    "record_path": "logs/m4-20261018T101246123107Z/named-default/matmul-32x32x32/record.json",
    "original_disassembly": "tmp/m4-20261018T101246123107Z/named-default/matmul-32x32x32/kernel.disasm"
  },
  {
    "case_id": "matmul-127x65x93",
    "identical_disassembly": true,
    "compile_seconds": 0.19230748806148767,
    "record_path": "logs/m4-20261018T101246123107Z/named-default/matmul-127x65x93/record.json",
    "original_disassembly": "tmp/m4-20261018T101246123107Z/named-default/matmul-127x65x93/kernel.disasm"
  },
  {
    "case_id": "matmul-1024x256x256",
    "identical_disassembly": true,
    "compile_seconds": 0.1928068995475769,
    "record_path": "logs/m4-20261018T101246123107Z/named-default/matmul-1024x256x256/record.json",
    "original_disassembly": "tmp/m4-20261018T101246123107Z/named-default/matmul-1024x256x256/kernel.disasm"
  }
]
```

主实验目标反汇编重复：[]。不同字节不自动证明算法或动态指令不同。

## 复现与边界

```bash
bash scripts/phase0.sh m4
```

单入口完成两轮重编译、独立正确性、交错计时、默认代码等价检查和报告导出。[m4-interaction.json](m4-interaction.json) 保存配置、主机/工具/Git 状态、原始计时和区间；完整日志和产物只保留在 logs/ 与 tmp/。

局限：一个程序家族，三个 Shape，两次独立启动但同主机连续实验；频率/共享负载/ASLR 未控制，无 PMU；热态复用不等于冷缓存带宽。参考 C++ 是有根据的常规实现，不是成熟自动调优系统的最优值。未计算 hash、未写单元测试。下一步先扩展独立家族和重复批次，围绕当前 IR 的透明规则建立对照，只有出现历史增量价值的可靠证据后再讨论持续状态。
