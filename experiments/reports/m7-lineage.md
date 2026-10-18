# M7 完整高层图与决策边界差异

仅比较同一上下文内合法候选；局部原始D在三图一致。

## single: 完整原始图

```mlir
module {
  func.func @kernel(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) attributes {llvm.emit_c_interface} {
    %ta = bufferization.to_tensor %a restrict : memref<64x64xf32>
    %tb = bufferization.to_tensor %b restrict : memref<64x64xf32>
    %empty = tensor.empty() : tensor<64x64xf32>
    // IRLoom target begin
    %d = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%ta, %tb : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %product = arith.mulf %x, %y : f32
      linalg.yield %product : f32
    } -> tensor<64x64xf32>
    // IRLoom target end

    %y = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%d, %ta : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%p: f32, %x: f32, %unused: f32):
      %value = arith.addf %p, %x : f32
      linalg.yield %value : f32
    } -> tensor<64x64xf32>
    %out = bufferization.to_memref %y : memref<64x64xf32>
    memref.copy %out, %c : memref<64x64xf32> to memref<64x64xf32>
    return
  }
}
```

### high_decision: separate → fused

```diff
--- before

+++ after

@@ -7,14 +7,10 @@

     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %6 = arith.mulf %in, %in_0 : f32
+      %5 = arith.mulf %in, %in_0 : f32
+      %6 = arith.addf %5, %in : f32
       linalg.yield %6 : f32
     } -> tensor<64x64xf32>
-    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %6 = arith.addf %in, %in_0 : f32
-      linalg.yield %6 : f32
-    } -> tensor<64x64xf32>
-    %5 = bufferization.to_memref %4 : memref<64x64xf32>
-    memref.copy %5, %arg2 : memref<64x64xf32> to memref<64x64xf32>
+    %4 = bufferization.to_memref %3 : memref<64x64xf32>
+    memref.copy %4, %arg2 : memref<64x64xf32> to memref<64x64xf32>
     return
   }
```

### bufferized: separate → fused

```diff
--- before

+++ after

@@ -6,10 +6,6 @@

     ^bb0(%in: f32, %in_0: f32, %out: f32):
       %0 = arith.mulf %in, %in_0 : f32
-      linalg.yield %0 : f32
-    }
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg0 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %0 = arith.addf %in, %in_0 : f32
-      linalg.yield %0 : f32
+      %1 = arith.addf %0, %in : f32
+      linalg.yield %1 : f32
     }
     memref.copy %alloc, %arg2 : memref<64x64xf32> to memref<64x64xf32>
```

### affine: separate → fused

```diff
--- before

+++ after

@@ -7,13 +7,6 @@

         %1 = affine.load %arg1[%arg3, %arg4] : memref<64x64xf32>
         %2 = arith.mulf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
-      }
-    }
-    affine.for %arg3 = 0 to 64 {
-      affine.for %arg4 = 0 to 64 {
-        %0 = affine.load %alloc[%arg3, %arg4] : memref<64x64xf32>
-        %1 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
-        %2 = arith.addf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
+        %3 = arith.addf %2, %0 : f32
+        affine.store %3, %alloc[%arg3, %arg4] : memref<64x64xf32>
       }
     }
```

### 当前窗口与高层历史统计

```json
{
  "separate": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      4,
      4,
      2,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    "upstream": [
      1,
      2,
      0,
      0
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        affine.store %v6, %v7[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 0.00036554038524627686,
      "L1": 0.0004899743944406509,
      "L2": 0.0005403608083724976
    }
  },
  "fused": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      2,
      2,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    "upstream": [
      1,
      2,
      0,
      0
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        %v7 = arith.addf %v6, %v2 : f32\n        affine.store %v7, %v8[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 0.0001070825383067131,
      "L1": 0.0001926543191075325,
      "L2": 0.00021448079496622086
    }
  }
}
```

## transpose: 完整原始图

```mlir
module {
  func.func @kernel(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) attributes {llvm.emit_c_interface} {
    %ta = bufferization.to_tensor %a restrict : memref<64x64xf32>
    %tb = bufferization.to_tensor %b restrict : memref<64x64xf32>
    %empty = tensor.empty() : tensor<64x64xf32>
    // IRLoom target begin
    %d = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%ta, %tb : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %product = arith.mulf %x, %y : f32
      linalg.yield %product : f32
    } -> tensor<64x64xf32>
    // IRLoom target end

    %y = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%d, %ta : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%p: f32, %x: f32, %unused: f32):
      %value = arith.addf %p, %x : f32
      linalg.yield %value : f32
    } -> tensor<64x64xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(j,i)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y, %d : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%yv: f32, %zv: f32, %unused: f32):
      %sum = arith.addf %yv, %zv : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    %out = bufferization.to_memref %result : memref<64x64xf32>
    memref.copy %out, %c : memref<64x64xf32> to memref<64x64xf32>
    return
  }
}
```

### high_decision: separate → fused

```diff
--- before

+++ after

@@ -8,19 +8,15 @@

     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %7 = arith.mulf %in, %in_0 : f32
+      %6 = arith.mulf %in, %in_0 : f32
+      linalg.yield %6 : f32
+    } -> tensor<64x64xf32>
+    %4 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0, %3 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
+    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
+      %6 = arith.addf %in, %in_0 : f32
+      %7 = arith.addf %6, %in_1 : f32
       linalg.yield %7 : f32
     } -> tensor<64x64xf32>
-    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %7 = arith.addf %in, %in_0 : f32
-      linalg.yield %7 : f32
-    } -> tensor<64x64xf32>
-    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %7 = arith.addf %in, %in_0 : f32
-      linalg.yield %7 : f32
-    } -> tensor<64x64xf32>
-    %6 = bufferization.to_memref %5 : memref<64x64xf32>
-    memref.copy %6, %arg2 : memref<64x64xf32> to memref<64x64xf32>
+    %5 = bufferization.to_memref %4 : memref<64x64xf32>
+    memref.copy %5, %arg2 : memref<64x64xf32> to memref<64x64xf32>
     return
   }
```

### bufferized: separate → fused

```diff
--- before

+++ after

@@ -10,13 +10,9 @@

       linalg.yield %0 : f32
     }
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %arg0 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_1: f32, %out: f32):
+    linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %arg0, %alloc_0 : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
+    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
       %0 = arith.addf %in, %in_1 : f32
-      linalg.yield %0 : f32
-    }
-    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_0 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_1: f32, %out: f32):
-      %0 = arith.addf %in, %in_1 : f32
-      linalg.yield %0 : f32
+      %1 = arith.addf %0, %in_2 : f32
+      linalg.yield %1 : f32
     }
     memref.dealloc %alloc_0 : memref<64x64xf32>
```

### affine: separate → fused

```diff
--- before

+++ after

@@ -15,14 +15,8 @@

         %0 = affine.load %alloc_0[%arg3, %arg4] : memref<64x64xf32>
         %1 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
-        %2 = arith.addf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
-      }
-    }
-    affine.for %arg3 = 0 to 64 {
-      affine.for %arg4 = 0 to 64 {
-        %0 = affine.load %alloc[%arg3, %arg4] : memref<64x64xf32>
-        %1 = affine.load %alloc_0[%arg4, %arg3] : memref<64x64xf32>
-        %2 = arith.addf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
+        %2 = affine.load %alloc_0[%arg4, %arg3] : memref<64x64xf32>
+        %3 = arith.addf %0, %1 : f32
+        %4 = arith.addf %3, %2 : f32
+        affine.store %4, %alloc[%arg3, %arg4] : memref<64x64xf32>
       }
     }
```

### 当前窗口与高层历史统计

```json
{
  "separate": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      6,
      6,
      3,
      1,
      2,
      2,
      2,
      1,
      1
    ],
    "upstream": [
      2,
      3,
      1,
      0
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        affine.store %v6, %v7[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 0.0001046452671289444,
      "L1": 0.0002628229558467865,
      "L2": 0.00028590206056833267
    }
  },
  "fused": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      4,
      5,
      2,
      1,
      2,
      2,
      2,
      1,
      1
    ],
    "upstream": [
      2,
      3,
      1,
      0
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        affine.store %v6, %v7[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 0.00010094325989484787,
      "L1": 0.0002375394105911255,
      "L2": 0.00026120804250240326
    }
  }
}
```

## row_reduce: 完整原始图

```mlir
module {
  func.func @kernel(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) attributes {llvm.emit_c_interface} {
    %ta = bufferization.to_tensor %a restrict : memref<64x64xf32>
    %tb = bufferization.to_tensor %b restrict : memref<64x64xf32>
    %empty = tensor.empty() : tensor<64x64xf32>
    // IRLoom target begin
    %d = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%ta, %tb : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %product = arith.mulf %x, %y : f32
      linalg.yield %product : f32
    } -> tensor<64x64xf32>
    // IRLoom target end

    %y = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%d, %ta : tensor<64x64xf32>, tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%p: f32, %x: f32, %unused: f32):
      %value = arith.addf %p, %x : f32
      linalg.yield %value : f32
    } -> tensor<64x64xf32>
    %zero = arith.constant 0.0 : f32
    %rowempty = tensor.empty() : tensor<64x1xf32>
    %init = linalg.fill ins(%zero : f32) outs(%rowempty : tensor<64x1xf32>) -> tensor<64x1xf32>
    %z = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%d : tensor<64x64xf32>) outs(%init : tensor<64x1xf32>) {
    ^bb0(%p: f32, %acc: f32):
      %sum = arith.addf %acc, %p : f32
      linalg.yield %sum : f32
    } -> tensor<64x1xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,0)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y, %z : tensor<64x64xf32>, tensor<64x1xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%yv: f32, %zv: f32, %unused: f32):
      %sum = arith.addf %yv, %zv : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    %out = bufferization.to_memref %result : memref<64x64xf32>
    memref.copy %out, %c : memref<64x64xf32> to memref<64x64xf32>
    return
  }
}
```

### high_decision: separate → fused

```diff
--- before

+++ after

@@ -3,4 +3,5 @@

 module {
   func.func @kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {llvm.emit_c_interface} {
+    %cst = arith.constant 0.000000e+00 : f32
     %0 = bufferization.to_tensor %arg0 restrict : memref<64x64xf32>
     %1 = bufferization.to_tensor %arg1 restrict : memref<64x64xf32>
@@ -8,27 +9,22 @@

     %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.mulf %in, %in_0 : f32
+      %9 = arith.mulf %in, %in_0 : f32
+      linalg.yield %9 : f32
+    } -> tensor<64x64xf32>
+    %4 = tensor.empty() : tensor<64x1xf32>
+    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x1xf32>) -> tensor<64x1xf32>
+    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x64xf32>) outs(%5 : tensor<64x1xf32>) {
+    ^bb0(%in: f32, %out: f32):
+      %9 = arith.addf %out, %in : f32
+      linalg.yield %9 : f32
+    } -> tensor<64x1xf32>
+    %7 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0, %6 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) {
+    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
+      %9 = arith.addf %in, %in_0 : f32
+      %10 = arith.addf %9, %in_1 : f32
       linalg.yield %10 : f32
     } -> tensor<64x64xf32>
-    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.addf %in, %in_0 : f32
-      linalg.yield %10 : f32
-    } -> tensor<64x64xf32>
-    %cst = arith.constant 0.000000e+00 : f32
-    %5 = tensor.empty() : tensor<64x1xf32>
-    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x1xf32>) -> tensor<64x1xf32>
-    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<64x64xf32>) outs(%6 : tensor<64x1xf32>) {
-    ^bb0(%in: f32, %out: f32):
-      %10 = arith.addf %out, %in : f32
-      linalg.yield %10 : f32
-    } -> tensor<64x1xf32>
-    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) {
-    ^bb0(%in: f32, %in_0: f32, %out: f32):
-      %10 = arith.addf %in, %in_0 : f32
-      linalg.yield %10 : f32
-    } -> tensor<64x64xf32>
-    %9 = bufferization.to_memref %8 : memref<64x64xf32>
-    memref.copy %9, %arg2 : memref<64x64xf32> to memref<64x64xf32>
+    %8 = bufferization.to_memref %7 : memref<64x64xf32>
+    memref.copy %8, %arg2 : memref<64x64xf32> to memref<64x64xf32>
     return
   }
```

### bufferized: separate → fused

```diff
--- before

+++ after

@@ -5,29 +5,23 @@

     %cst = arith.constant 0.000000e+00 : f32
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
-    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.mulf %in, %in_2 : f32
+    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
+    ^bb0(%in: f32, %in_1: f32, %out: f32):
+      %0 = arith.mulf %in, %in_1 : f32
       linalg.yield %0 : f32
     }
-    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_0, %arg0 : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.addf %in, %in_2 : f32
-      linalg.yield %0 : f32
-    }
-    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32>
-    linalg.fill ins(%cst : f32) outs(%alloc_1 : memref<64x1xf32>)
-    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc_0 : memref<64x64xf32>) outs(%alloc_1 : memref<64x1xf32>) {
+    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32>
+    linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64x1xf32>)
+    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<64x64xf32>) outs(%alloc_0 : memref<64x1xf32>) {
     ^bb0(%in: f32, %out: f32):
       %0 = arith.addf %out, %in : f32
       linalg.yield %0 : f32
     }
-    memref.dealloc %alloc_0 : memref<64x64xf32>
-    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_1 : memref<64x64xf32>, memref<64x1xf32>) outs(%alloc : memref<64x64xf32>) {
-    ^bb0(%in: f32, %in_2: f32, %out: f32):
-      %0 = arith.addf %in, %in_2 : f32
-      linalg.yield %0 : f32
+    linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc, %arg0, %alloc_0 : memref<64x64xf32>, memref<64x64xf32>, memref<64x1xf32>) outs(%alloc : memref<64x64xf32>) {
+    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
+      %0 = arith.addf %in, %in_1 : f32
+      %1 = arith.addf %0, %in_2 : f32
+      linalg.yield %1 : f32
     }
-    memref.dealloc %alloc_1 : memref<64x1xf32>
+    memref.dealloc %alloc_0 : memref<64x1xf32>
     memref.copy %alloc, %arg2 : memref<64x64xf32> to memref<64x64xf32>
     memref.dealloc %alloc : memref<64x64xf32>
```

### affine: separate → fused

```diff
--- before

+++ after

@@ -4,5 +4,4 @@

     %cst = arith.constant 0.000000e+00 : f32
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
-    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
     affine.for %arg3 = 0 to 64 {
       affine.for %arg4 = 0 to 64 {
@@ -10,39 +9,32 @@

         %1 = affine.load %arg1[%arg3, %arg4] : memref<64x64xf32>
         %2 = arith.mulf %0, %1 : f32
-        affine.store %2, %alloc_0[%arg3, %arg4] : memref<64x64xf32>
+        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
+      }
+    }
+    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32>
+    affine.for %arg3 = 0 to 64 {
+      affine.for %arg4 = 0 to 1 {
+        affine.store %cst, %alloc_0[%arg3, %arg4] : memref<64x1xf32>
       }
     }
     affine.for %arg3 = 0 to 64 {
       affine.for %arg4 = 0 to 64 {
-        %0 = affine.load %alloc_0[%arg3, %arg4] : memref<64x64xf32>
-        %1 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
-        %2 = arith.addf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
-      }
-    }
-    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x1xf32>
-    affine.for %arg3 = 0 to 64 {
-      affine.for %arg4 = 0 to 1 {
-        affine.store %cst, %alloc_1[%arg3, %arg4] : memref<64x1xf32>
+        %0 = affine.load %alloc[%arg3, %arg4] : memref<64x64xf32>
+        %1 = affine.load %alloc_0[%arg3, %c0] : memref<64x1xf32>
+        %2 = arith.addf %1, %0 : f32
+        affine.store %2, %alloc_0[%arg3, %c0] : memref<64x1xf32>
       }
     }
     affine.for %arg3 = 0 to 64 {
       affine.for %arg4 = 0 to 64 {
-        %0 = affine.load %alloc_0[%arg3, %arg4] : memref<64x64xf32>
-        %1 = affine.load %alloc_1[%arg3, %c0] : memref<64x1xf32>
-        %2 = arith.addf %1, %0 : f32
-        affine.store %2, %alloc_1[%arg3, %c0] : memref<64x1xf32>
+        %0 = affine.load %alloc[%arg3, %arg4] : memref<64x64xf32>
+        %1 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
+        %2 = affine.load %alloc_0[%arg3, %c0] : memref<64x1xf32>
+        %3 = arith.addf %0, %1 : f32
+        %4 = arith.addf %3, %2 : f32
+        affine.store %4, %alloc[%arg3, %arg4] : memref<64x64xf32>
       }
     }
-    memref.dealloc %alloc_0 : memref<64x64xf32>
-    affine.for %arg3 = 0 to 64 {
-      affine.for %arg4 = 0 to 64 {
-        %0 = affine.load %alloc[%arg3, %arg4] : memref<64x64xf32>
-        %1 = affine.load %alloc_1[%arg3, %c0] : memref<64x1xf32>
-        %2 = arith.addf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<64x64xf32>
-      }
-    }
-    memref.dealloc %alloc_1 : memref<64x1xf32>
+    memref.dealloc %alloc_0 : memref<64x1xf32>
     memref.copy %alloc, %arg2 : memref<64x64xf32> to memref<64x64xf32>
     memref.dealloc %alloc : memref<64x64xf32>
```

### 当前窗口与高层历史统计

```json
{
  "separate": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      10,
      8,
      5,
      1,
      3,
      3,
      3,
      1,
      0
    ],
    "upstream": [
      2,
      4,
      0,
      1
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        affine.store %v6, %v7[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 9.524263441562653e-05,
      "L1": 0.00032582879066467285,
      "L2": 0.0003494545817375183
    }
  },
  "fused": {
    "ops": [
      "affine.for",
      "affine.load",
      "affine.store",
      "arith.mulf",
      "arith.addf",
      "memref.alloc",
      "memref.dealloc",
      "memref.copy",
      "reversed_loads"
    ],
    "local_current": [
      2,
      2,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "full_current": [
      8,
      7,
      4,
      1,
      3,
      2,
      2,
      1,
      0
    ],
    "upstream": [
      2,
      4,
      0,
      1
    ],
    "upstream_names": [
      "target_uses",
      "generic_ops",
      "transposed_consumer",
      "reduction_consumer"
    ],
    "local_window": "affine.for %v0 = 0 to 64 {\n      affine.for %v1 = 0 to 64 {\n        %v2 = affine.load %v3[%v0, %v1] : memref<64x64xf32>\n        %v4 = affine.load %v5[%v0, %v1] : memref<64x64xf32>\n        %v6 = arith.mulf %v2, %v4 : f32\n        affine.store %v6, %v7[%v0, %v1] : memref<64x64xf32>\n      }\n    }",
    "stages": {
      "local_current": "affine:first outer loop nest",
      "full_current": "affine:whole function",
      "upstream": "high:whole graph"
    },
    "seconds": {
      "L0": 0.00010034721344709396,
      "L1": 0.000292384997010231,
      "L2": 0.0003163740038871765
    }
  }
}
```
