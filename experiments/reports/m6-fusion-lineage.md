# M6 融合链快照差异

Case pointwise_chain-32x64: C=A*B+A；输入/输出相同，两个阶段与融合单阶段对比。

## high_decision

```diff
--- before

+++ after

@@ -7,14 +7,10 @@

     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%2 : tensor<32x64xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %6 = arith.mulf %in, %in_0 : f32
+      %5 = arith.mulf %in, %in_0 : f32
+      %6 = arith.addf %5, %in : f32
       linalg.yield %6 : f32
     } -> tensor<32x64xf32>
-    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%2 : tensor<32x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %6 = arith.addf %in, %in_0 : f32
-      linalg.yield %6 : f32
-    } -> tensor<32x64xf32>
-    %5 = bufferization.to_memref %4 : memref<32x64xf32>
-    memref.copy %5, %arg2 : memref<32x64xf32> to memref<32x64xf32>
+    %4 = bufferization.to_memref %3 : memref<32x64xf32>
+    memref.copy %4, %arg2 : memref<32x64xf32> to memref<32x64xf32>
     return
   }
```

## bufferized

```diff
--- before

+++ after

@@ -6,10 +6,6 @@

     ^bb0(%in: f32, %in_0: f32, %out: f32):
       %0 = arith.mulf %in, %in_0 : f32
-      linalg.yield %0 : f32
-    }
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg0 : memref<32x64xf32>, memref<32x64xf32>) outs(%alloc : memref<32x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %0 = arith.addf %in, %in_0 : f32
-      linalg.yield %0 : f32
+      %1 = arith.addf %0, %in : f32
+      linalg.yield %1 : f32
     }
     memref.copy %alloc, %arg2 : memref<32x64xf32> to memref<32x64xf32>
```
