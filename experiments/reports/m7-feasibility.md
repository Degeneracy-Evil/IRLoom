# M7.0：MLIR 18 多消费者图可达性

已拉取并完整阅读 `6e56d05..b1b60b9` 的 README/M7_PLAN diff。沿用 MLIR18.1.3，不开发自定义Pass或模型。

## 一个目标、三个上下文

目标 D[i,j]=A[i,j]*B[i,j]，输入/Shape/f32/行主布局和原始producer文本在三个上下文完全一致。

- single：C=D+A。
- transpose：Y=D+A；C[i,j]=Y[i,j]+D[j,i]，正方形输入。
- row_reduce：Y=D+A；Z[i]=sum_j D[i,j]；C[i,j]=Y[i,j]+Z[i]。

各上下文候选分别输出相同结果；不同上下文的总工作量不同，因此不横比它们的绝对延迟。double参考在独立C运行器中求值，归约真值每行只求一次；沿用原ABI、哨兵、NaN检查、四种子与数值容差。

## 实际可达路径

高层 separate / linalg-fuse-elementwise-ops，随后 one-shot-bufferize、buffer-deallocation；低层 none/tile8/tile32，复用既有 affine→LLVM→native。8×8功能规格共18条，18/18编译与正确性通过、180次计时。此小规格只验收可运行，不据此挑选正式动作或宣称性能。

| Context | separate generic / alloc | fused generic / alloc |
| --- | --- | --- |
| single | 2 / 1 | 1 / 1 |
| transpose | 3 / 2 | 2 / 2 |
| row_reduce | 4 / 3 | 3 / 2 |

关键限制：原生融合Pass会保留多消费者D，transpose/row_reduce只融合其它合法consumer边，不能把 fused 解释为强行把D内联到所有消费者。single的D被融合消失；transpose仍为两个generic，两路均两个alloc；row_reduce的generic从4降3、alloc从3降2。这已经说明编译器本身利用fan-out上下文，不能把合法性判断当作学习贡献。所有临时分配/释放和输出copy包含在kernel计时内。

## 快照样例（row_reduce）

### high_decision

```diff
--- before

+++ after

@@ -3,4 +3,5 @@

 module {
   func.func @kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) attributes {llvm.emit_c_interface} {
+    %cst = arith.constant 0.000000e+00 : f32
     %0 = bufferization.to_tensor %arg0 restrict : memref<8x8xf32>
     %1 = bufferization.to_tensor %arg1 restrict : memref<8x8xf32>
@@ -8,27 +9,22 @@

     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%2 : tensor<8x8xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.mulf %in, %in_0 : f32
+      %9 = arith.mulf %in, %in_0 : f32
+      linalg.yield %9 : f32
+    } -> tensor<8x8xf32>
+    %4 = tensor.empty() : tensor<8x1xf32>
+    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8x1xf32>) -> tensor<8x1xf32>
+    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<8x8xf32>) outs(%5 : tensor<8x1xf32>) {
+    ^bb0(%in: f32, %out: f32):
+      %9 = arith.addf %out, %in : f32
+      linalg.yield %9 : f32
+    } -> tensor<8x1xf32>
+    %7 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0, %6 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x1xf32>) outs(%2 : tensor<8x8xf32>) {
+    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
+      %9 = arith.addf %in, %in_0 : f32
+      %10 = arith.addf %9, %in_1 : f32
       linalg.yield %10 : f32
     } -> tensor<8x8xf32>
-    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%2 : tensor<8x8xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.addf %in, %in_0 : f32
-      linalg.yield %10 : f32
-    } -> tensor<8x8xf32>
-    %cst = arith.constant 0.000000e+00 : f32
-    %5 = tensor.empty() : tensor<8x1xf32>
-    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x1xf32>) -> tensor<8x1xf32>
-    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<8x8xf32>) outs(%6 : tensor<8x1xf32>) {
-    ^bb0(%in: f32, %out: f32):
-      %10 = arith.addf %out, %in : f32
-      linalg.yield %10 : f32
-    } -> tensor<8x1xf32>
-    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<8x8xf32>, tensor<8x1xf32>) outs(%2 : tensor<8x8xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.addf %in, %in_0 : f32
-      linalg.yield %10 : f32
-    } -> tensor<8x8xf32>
-    %9 = bufferization.to_memref %8 : memref<8x8xf32>
-    memref.copy %9, %arg2 : memref<8x8xf32> to memref<8x8xf32>
+    %8 = bufferization.to_memref %7 : memref<8x8xf32>
+    memref.copy %8, %arg2 : memref<8x8xf32> to memref<8x8xf32>
     return
   }
```

### bufferized

```diff
--- before

+++ after

@@ -5,29 +5,23 @@

     %cst = arith.constant 0.000000e+00 : f32
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
-    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<8x8xf32>, memref<8x8xf32>) outs(%alloc_0 : memref<8x8xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.mulf %in, %in_2 : f32
+    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<8x8xf32>, memref<8x8xf32>) outs(%alloc : memref<8x8xf32>) {
+    ^bb0(%in: f32, %in_1: f32, %out: f32):
+      %0 = arith.mulf %in, %in_1 : f32
       linalg.yield %0 : f32
     }
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %arg0 : memref<8x8xf32>, memref<8x8xf32>) outs(%alloc : memref<8x8xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.addf %in, %in_2 : f32
-      linalg.yield %0 : f32
-    }
-    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32>
-    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<8x1xf32>)
-    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc_0 : memref<8x8xf32>) outs(%alloc_1 : memref<8x1xf32>) {
+    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32>
+    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<8x1xf32>)
+    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<8x8xf32>) outs(%alloc_0 : memref<8x1xf32>) {
     ^bb0(%in: f32, %out: f32):
       %0 = arith.addf %out, %in : f32
       linalg.yield %0 : f32
     }
-    memref.dealloc %alloc_0 : memref<8x8xf32>
-    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_1 : memref<8x8xf32>, memref<8x1xf32>) outs(%alloc : memref<8x8xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.addf %in, %in_2 : f32
-      linalg.yield %0 : f32
+    linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg0, %alloc_0 : memref<8x8xf32>, memref<8x8xf32>, memref<8x1xf32>) outs(%alloc : memref<8x8xf32>) {
+    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
+      %0 = arith.addf %in, %in_1 : f32
+      %1 = arith.addf %0, %in_2 : f32
+      linalg.yield %1 : f32
     }
-    memref.dealloc %alloc_1 : memref<8x1xf32>
+    memref.dealloc %alloc_0 : memref<8x1xf32>
     memref.copy %alloc, %arg2 : memref<8x8xf32> to memref<8x8xf32>
     memref.dealloc %alloc : memref<8x8xf32>
```

## 复现与下一步

```bash
bash scripts/phase0.sh collect --config experiments/m7-feasibility.json
```

本次实际运行配置副本为tmp/m7-feasibility.json，记录logs/20261018T174414205476Z/records.jsonl，IR/二进制位于对应tmp时间戳。发布配置与副本内容一致；不计算hash、不增加单元测试。下一步在看正式标签前冻结64×64、513×513两个规模、同样六条路径和三种上下文；采集两批，检验上下文相对效应/最优路径，并区分局部、完整当前affine与上游图的可见信息。小样本仅作规则/信息可分辨性pilot，不训练Attention。
