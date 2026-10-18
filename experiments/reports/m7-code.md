# M7 低层IR与目标代码证据（正式首批）

来源为干净提交4b9347d的logs/20261018T175248152318Z。以下均为采集后审阅，不是预测特征。对照高层固定fused，低层none/tile32；完整指令和IR路径在m7-results.json。

tile作用于完整图，消费者和producer循环都会改变；不能据此把差异全部归为目标D本身的优化。两路都出现向量浮点乘法，不能仅凭有无vmulps解释快慢；归约仍受严格浮点顺序约束。没有PMU，不进行缓存或端口的排他归因。

## single, 513×513

### transformed：none → tile32

```diff
--- before

+++ after

@@ -1,12 +1,18 @@

+#map = affine_map<(d0) -> (d0)>
+#map1 = affine_map<(d0) -> (d0 + 32, 513)>
 module {
   func.func @kernel(%arg0: memref<513x513xf32>, %arg1: memref<513x513xf32>, %arg2: memref<513x513xf32>) attributes {llvm.emit_c_interface} {
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<513x513xf32>
-    affine.for %arg3 = 0 to 513 {
-      affine.for %arg4 = 0 to 513 {
-        %0 = affine.load %arg0[%arg3, %arg4] : memref<513x513xf32>
-        %1 = affine.load %arg1[%arg3, %arg4] : memref<513x513xf32>
-        %2 = arith.mulf %0, %1 : f32
-        %3 = arith.addf %2, %0 : f32
-        affine.store %3, %alloc[%arg3, %arg4] : memref<513x513xf32>
+    affine.for %arg3 = 0 to 513 step 32 {
+      affine.for %arg4 = 0 to 513 step 32 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = #map(%arg4) to min #map1(%arg4) {
+            %0 = affine.load %arg0[%arg5, %arg6] : memref<513x513xf32>
+            %1 = affine.load %arg1[%arg5, %arg6] : memref<513x513xf32>
+            %2 = arith.mulf %0, %1 : f32
+            %3 = arith.addf %2, %0 : f32
+            affine.store %3, %alloc[%arg5, %arg6] : memref<513x513xf32>
+          }
+        }
       }
     }
```

### fused-none 首处向量乘法附近

`/home/chen/develop/IRLoom/tmp/20261018T175248152318Z/single-513/fused-none/kernel.disasm`

```asm
     119: c5 fc 10 8c 87 40 ff ff ff   	vmovups	-0xc0(%rdi,%rax,4), %ymm1
     122: c5 fc 10 94 87 60 ff ff ff   	vmovups	-0xa0(%rdi,%rax,4), %ymm2
     12b: c5 fc 10 5c 87 80            	vmovups	-0x80(%rdi,%rax,4), %ymm3
     131: c5 fc 59 a4 82 20 ff ff ff   	vmulps	-0xe0(%rdx,%rax,4), %ymm0, %ymm4
     13a: c5 f4 59 ac 82 40 ff ff ff   	vmulps	-0xc0(%rdx,%rax,4), %ymm1, %ymm5
     143: c5 ec 59 b4 82 60 ff ff ff   	vmulps	-0xa0(%rdx,%rax,4), %ymm2, %ymm6
     14c: c5 e4 59 7c 82 80            	vmulps	-0x80(%rdx,%rax,4), %ymm3, %ymm7
     152: c5 fc 58 c4                  	vaddps	%ymm4, %ymm0, %ymm0
     156: c5 f4 58 cd                  	vaddps	%ymm5, %ymm1, %ymm1
     15a: c5 ec 58 d6                  	vaddps	%ymm6, %ymm2, %ymm2
     15e: c5 e4 58 df                  	vaddps	%ymm7, %ymm3, %ymm3
     162: c5 fc 11 84 81 20 ff ff ff   	vmovups	%ymm0, -0xe0(%rcx,%rax,4)
     16b: c5 fc 11 8c 81 40 ff ff ff   	vmovups	%ymm1, -0xc0(%rcx,%rax,4)
     174: c5 fc 11 94 81 60 ff ff ff   	vmovups	%ymm2, -0xa0(%rcx,%rax,4)
     17d: c5 fc 11 5c 81 80            	vmovups	%ymm3, -0x80(%rcx,%rax,4)
```

### fused-tile32 首处向量乘法附近

`/home/chen/develop/IRLoom/tmp/20261018T175248152318Z/single-513/fused-tile32/kernel.disasm`

```asm
     266: c5 fc 10 4c 82 c0            	vmovups	-0x40(%rdx,%rax,4), %ymm1
     26c: c5 fc 10 54 82 e0            	vmovups	-0x20(%rdx,%rax,4), %ymm2
     272: c5 fc 10 1c 82               	vmovups	(%rdx,%rax,4), %ymm3
     277: c4 c1 7c 59 64 83 a0         	vmulps	-0x60(%r11,%rax,4), %ymm0, %ymm4
     27e: c4 c1 74 59 6c 83 c0         	vmulps	-0x40(%r11,%rax,4), %ymm1, %ymm5
     285: c4 c1 6c 59 74 83 e0         	vmulps	-0x20(%r11,%rax,4), %ymm2, %ymm6
     28c: c4 c1 64 59 3c 83            	vmulps	(%r11,%rax,4), %ymm3, %ymm7
     292: c5 fc 58 c4                  	vaddps	%ymm4, %ymm0, %ymm0
     296: c5 f4 58 cd                  	vaddps	%ymm5, %ymm1, %ymm1
     29a: c5 ec 58 d6                  	vaddps	%ymm6, %ymm2, %ymm2
     29e: c5 e4 58 df                  	vaddps	%ymm7, %ymm3, %ymm3
     2a2: c5 fc 11 44 81 a0            	vmovups	%ymm0, -0x60(%rcx,%rax,4)
     2a8: c5 fc 11 4c 81 c0            	vmovups	%ymm1, -0x40(%rcx,%rax,4)
     2ae: c5 fc 11 54 81 e0            	vmovups	%ymm2, -0x20(%rcx,%rax,4)
     2b4: c5 fc 11 1c 81               	vmovups	%ymm3, (%rcx,%rax,4)
```

## row_reduce, 513×513

### transformed：none → tile32

```diff
--- before

+++ after

@@ -1,36 +1,54 @@

+#map = affine_map<(d0) -> (d0)>
+#map1 = affine_map<(d0) -> (d0 + 32, 513)>
 module {
   func.func @kernel(%arg0: memref<513x513xf32>, %arg1: memref<513x513xf32>, %arg2: memref<513x513xf32>) attributes {llvm.emit_c_interface} {
     %cst = arith.constant 0.000000e+00 : f32
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<513x513xf32>
-    affine.for %arg3 = 0 to 513 {
-      affine.for %arg4 = 0 to 513 {
-        %0 = affine.load %arg0[%arg3, %arg4] : memref<513x513xf32>
-        %1 = affine.load %arg1[%arg3, %arg4] : memref<513x513xf32>
-        %2 = arith.mulf %0, %1 : f32
-        affine.store %2, %alloc[%arg3, %arg4] : memref<513x513xf32>
+    affine.for %arg3 = 0 to 513 step 32 {
+      affine.for %arg4 = 0 to 513 step 32 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = #map(%arg4) to min #map1(%arg4) {
+            %0 = affine.load %arg0[%arg5, %arg6] : memref<513x513xf32>
+            %1 = affine.load %arg1[%arg5, %arg6] : memref<513x513xf32>
+            %2 = arith.mulf %0, %1 : f32
+            affine.store %2, %alloc[%arg5, %arg6] : memref<513x513xf32>
+          }
+        }
       }
     }
     %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<513x1xf32>
-    affine.for %arg3 = 0 to 513 {
-      affine.for %arg4 = 0 to 1 {
-        affine.store %cst, %alloc_0[%arg3, %arg4] : memref<513x1xf32>
+    affine.for %arg3 = 0 to 513 step 32 {
+      affine.for %arg4 = 0 to 1 step 32 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = 0 to 1 {
+            affine.store %cst, %alloc_0[%arg5, %arg6] : memref<513x1xf32>
+          }
+        }
       }
     }
-    affine.for %arg3 = 0 to 513 {
-      affine.for %arg4 = 0 to 513 {
-        %0 = affine.load %alloc[%arg3, %arg4] : memref<513x513xf32>
-        %1 = affine.load %alloc_0[%arg3, 0] : memref<513x1xf32>
-        %2 = arith.addf %1, %0 : f32
-        affine.store %2, %alloc_0[%arg3, 0] : memref<513x1xf32>
+    affine.for %arg3 = 0 to 513 step 32 {
+      affine.for %arg4 = 0 to 513 step 32 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = #map(%arg4) to min #map1(%arg4) {
+            %0 = affine.load %alloc[%arg5, %arg6] : memref<513x513xf32>
+            %1 = affine.load %alloc_0[%arg5, 0] : memref<513x1xf32>
+            %2 = arith.addf %1, %0 : f32
+            affine.store %2, %alloc_0[%arg5, 0] : memref<513x1xf32>
+          }
+        }
       }
     }
-    affine.for %arg3 = 0 to 513 {
-      affine.for %arg4 = 0 to 513 {
-        %0 = affine.load %alloc[%arg3, %arg4] : memref<513x513xf32>
-        %1 = affine.load %arg0[%arg3, %arg4] : memref<513x513xf32>
-        %2 = affine.load %alloc_0[%arg3, 0] : memref<513x1xf32>
-        %3 = arith.addf %0, %1 : f32
-        %4 = arith.addf %3, %2 : f32
-        affine.store %4, %alloc[%arg3, %arg4] : memref<513x513xf32>
+    affine.for %arg3 = 0 to 513 step 32 {
+      affine.for %arg4 = 0 to 513 step 32 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = #map(%arg4) to min #map1(%arg4) {
+            %0 = affine.load %alloc[%arg5, %arg6] : memref<513x513xf32>
+            %1 = affine.load %arg0[%arg5, %arg6] : memref<513x513xf32>
+            %2 = affine.load %alloc_0[%arg5, 0] : memref<513x1xf32>
+            %3 = arith.addf %0, %1 : f32
+            %4 = arith.addf %3, %2 : f32
+            affine.store %4, %alloc[%arg5, %arg6] : memref<513x513xf32>
+          }
+        }
       }
     }
```

### fused-none 首处向量乘法附近

`/home/chen/develop/IRLoom/tmp/20261018T175248152318Z/row_reduce-513/fused-none/kernel.disasm`

```asm
     109: c5 fc 10 8c 86 40 ff ff ff   	vmovups	-0xc0(%rsi,%rax,4), %ymm1
     112: c5 fc 10 94 86 60 ff ff ff   	vmovups	-0xa0(%rsi,%rax,4), %ymm2
     11b: c5 fc 10 5c 86 80            	vmovups	-0x80(%rsi,%rax,4), %ymm3
     121: c5 fc 59 84 82 20 ff ff ff   	vmulps	-0xe0(%rdx,%rax,4), %ymm0, %ymm0
     12a: c5 f4 59 8c 82 40 ff ff ff   	vmulps	-0xc0(%rdx,%rax,4), %ymm1, %ymm1
     133: c5 ec 59 94 82 60 ff ff ff   	vmulps	-0xa0(%rdx,%rax,4), %ymm2, %ymm2
     13c: c5 e4 59 5c 82 80            	vmulps	-0x80(%rdx,%rax,4), %ymm3, %ymm3
     142: c5 fc 11 84 81 20 ff ff ff   	vmovups	%ymm0, -0xe0(%rcx,%rax,4)
     14b: c5 fc 11 8c 81 40 ff ff ff   	vmovups	%ymm1, -0xc0(%rcx,%rax,4)
     154: c5 fc 11 94 81 60 ff ff ff   	vmovups	%ymm2, -0xa0(%rcx,%rax,4)
     15d: c5 fc 11 5c 81 80            	vmovups	%ymm3, -0x80(%rcx,%rax,4)
     163: c5 fc 10 44 86 a0            	vmovups	-0x60(%rsi,%rax,4), %ymm0
     169: c5 fc 10 4c 86 c0            	vmovups	-0x40(%rsi,%rax,4), %ymm1
     16f: c5 fc 10 54 86 e0            	vmovups	-0x20(%rsi,%rax,4), %ymm2
     175: c5 fc 10 1c 86               	vmovups	(%rsi,%rax,4), %ymm3
```

### fused-tile32 首处向量乘法附近

`/home/chen/develop/IRLoom/tmp/20261018T175248152318Z/row_reduce-513/fused-tile32/kernel.disasm`

```asm
     267: c4 c1 7c 10 4c 85 c0         	vmovups	-0x40(%r13,%rax,4), %ymm1
     26e: c4 c1 7c 10 54 85 e0         	vmovups	-0x20(%r13,%rax,4), %ymm2
     275: c4 c1 7c 10 5c 85 00         	vmovups	(%r13,%rax,4), %ymm3
     27c: c4 c1 7c 59 44 83 a0         	vmulps	-0x60(%r11,%rax,4), %ymm0, %ymm0
     283: c4 c1 74 59 4c 83 c0         	vmulps	-0x40(%r11,%rax,4), %ymm1, %ymm1
     28a: c4 c1 6c 59 54 83 e0         	vmulps	-0x20(%r11,%rax,4), %ymm2, %ymm2
     291: c4 c1 64 59 1c 83            	vmulps	(%r11,%rax,4), %ymm3, %ymm3
     297: c5 fc 11 44 85 a0            	vmovups	%ymm0, -0x60(%rbp,%rax,4)
     29d: c5 fc 11 4c 85 c0            	vmovups	%ymm1, -0x40(%rbp,%rax,4)
     2a3: c5 fc 11 54 85 e0            	vmovups	%ymm2, -0x20(%rbp,%rax,4)
     2a9: c5 fc 11 5c 85 00            	vmovups	%ymm3, (%rbp,%rax,4)
     2af: 48 83 c0 20                  	addq	$0x20, %rax
     2b3: 48 39 c2                     	cmpq	%rax, %rdx
     2b6: 75 a8                        	jne	0x260 <kernel+0x260>
     2b8: 4c 89 d0                     	movq	%r10, %rax
```

