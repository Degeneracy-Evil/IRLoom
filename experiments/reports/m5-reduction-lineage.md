# M5 行归约的真实高层决策

Case: reduction-32x96，每行加法顺序保留，独立行交错次序改变。

## ij

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %reduction = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %reduction iterator_interchange = [0, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

## ji

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %reduction = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %reduction iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

## high_decision

```diff
--- before

+++ after

@@ -1,9 +1,9 @@

-#map = affine_map<(d0, d1) -> (d0, d1)>
-#map1 = affine_map<(d0, d1) -> (d0, 0)>
+#map = affine_map<(d0, d1) -> (d1, d0)>
+#map1 = affine_map<(d0, d1) -> (d1, 0)>
 module {
   func.func @kernel(%arg0: memref<32x96xf32>, %arg1: memref<1x1xf32>, %arg2: memref<32x1xf32>) attributes {llvm.emit_c_interface} {
     %cst = arith.constant 0.000000e+00 : f32
     linalg.fill ins(%cst : f32) outs(%arg2 : memref<32x1xf32>)
-    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<32x96xf32>) outs(%arg2 : memref<32x1xf32>) {
+    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : memref<32x96xf32>) outs(%arg2 : memref<32x1xf32>) {
     ^bb0(%in: f32, %out: f32):
       %0 = arith.addf %out, %in : f32
```

## affine

```diff
--- before

+++ after

@@ -8,10 +8,10 @@

       }
     }
-    affine.for %arg3 = 0 to 32 {
-      affine.for %arg4 = 0 to 96 {
-        %0 = affine.load %arg0[%arg3, %arg4] : memref<32x96xf32>
-        %1 = affine.load %arg2[%arg3, %c0] : memref<32x1xf32>
+    affine.for %arg3 = 0 to 96 {
+      affine.for %arg4 = 0 to 32 {
+        %0 = affine.load %arg0[%arg4, %arg3] : memref<32x96xf32>
+        %1 = affine.load %arg2[%arg4, %c0] : memref<32x1xf32>
         %2 = arith.addf %1, %0 : f32
-        affine.store %2, %arg2[%arg3, %c0] : memref<32x1xf32>
+        affine.store %2, %arg2[%arg4, %c0] : memref<32x1xf32>
       }
     }
```
