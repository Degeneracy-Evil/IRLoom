# Phase 0 记录格式 v1

`logs/<run>/records.jsonl` 每行一个 Case × Candidate；数值统一使用秒（编译/命令墙钟）和 ms/kernel（执行）。不是每个计时批次一行。`run.json` 保留完整配置、批次测量及整轮墙钟时间，`manifest.json` 保留测量前的构建清单，`schedule.json` 保留按轮随机交错的执行顺序。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前为 1 |
| `program_id/case_id/candidate_id` | 程序类、静态输入规格、候选标识 |
| `seed/input` | 输入种子，shape 与 f32；matmul shape 顺序为 M,N,K，其他为 M,N |
| `metadata` | 编译主机 OS、CPU/ISA/cache/内存、工具绝对路径/版本、Git commit/工作区状态、线程环境 |
| `measurement_host` | 采样时再次取得的主机/工具状态（单独 run 时尤其重要） |
| `transforms` | 按顺序执行的候选变换参数；所有候选共用 lowering 和 clang 参数 |
| `snapshots` | high → affine → transformed → scf → llvm_dialect → llvm；每项含绝对 path、parent，以及对应 flags/event_index |
| `events` | 有序 command 数组、cwd、stage、log、status、returncode、seconds；校验/校准/计时亦保留 |
| `binary/object/disassembly` | 本机产物绝对路径；同目录包含生成高层 IR 和编译时复制的 runner.c |
| `compile_seconds` | 编译、翻译、反汇编、链接子进程耗时总和；不包含采样 |
| `correctness` | status、atol/rtol、是否允许重关联、每个 seed 的最大绝对误差 |
| `measurement` | 线程数、CPU、预热、时钟、每批次数、目标样本数、超时、顺序种子、缓存策略 |
| `timings_ms` | 按测量轮次保存的原始每次 kernel 延迟；一次测量为批量执行后除以次数 |
| `ir_changed` | transformed 文本相对本 case 默认路径是否改变 |
| `duplicate_of` | 与同 case 更早候选完整可执行文件字节一致时指向其 id，否则 null |
| `code_duplicate_of` | 目标文件反汇编（含指令字节、地址与重定位，去掉文件路径标题）一致时指向更早候选 |
| `fingerprints` | 固定注明按用户要求不计算 hash；路径引用不保证文件未被篡改 |
| `status/failure_stage/error` | compiled、ok 或 failed；失败的阶段及异常/日志引用 |

`status=ok` 且独立正确性通过并有至少 10 个样本才进入延迟排名。失败可能发生在 affine/transformed/scf/llvm_dialect/translate/codegen/disassembly/link/correctness/calibration/timing 等阶段，超时在对应 event 中标为 timeout。失败的已完成快照仍保留。

这里的 lineage 仅表示文件级「上一阶段经过哪些命令得到下一阶段」，不是 Operation 级 N:M 溯源。多个 Pass 位于一个命令时，只记录该命令的整体成功；不把它当成每个内部 Pattern 的跟踪记录。编译器内部 clang -O3 优化也只记录边界和 Flags。

指令字节不同仍不保证语义实现不同；反汇编相同也不等于密码学完整性证明。当前基准无外部数据依赖，后续若增加全局数据或链接库，需要扩展重复识别。报告必须披露这一限制。
