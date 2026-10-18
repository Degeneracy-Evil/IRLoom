# M4 代表性快照与函数代码差异

这里只保留紧凑文本摘录。地址与静态指令数量不作为性能指标；完整文件在所列本地产物路径。

## matmul-127x65x93: default → ikj-none

median 0.314736 → 0.076537 ms/kernel。

`default` 的高层 schedule：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %generic iterator_interchange = [0, 1, 2] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

编译命令：

```text
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/default/high.mlir --transform-preload-library=transform-library-paths=tmp/20261018T101246162932Z/matmul-127x65x93/default/schedule.mlir --transform-interpreter -o tmp/20261018T101246162932Z/matmul-127x65x93/default/high_decision.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/default/high_decision.mlir --convert-linalg-to-affine-loops -o tmp/20261018T101246162932Z/matmul-127x65x93/default/affine.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/default/affine.mlir --canonicalize --cse -o tmp/20261018T101246162932Z/matmul-127x65x93/default/transformed.mlir
/usr/lib/llvm-18/bin/clang -O3 -march=native -ffp-contract=off -c tmp/20261018T101246162932Z/matmul-127x65x93/default/kernel.ll -o tmp/20261018T101246162932Z/matmul-127x65x93/default/kernel.o
```

`ikj-none` 的高层 schedule：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

编译命令：

```text
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high.mlir --transform-preload-library=transform-library-paths=tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/schedule.mlir --transform-interpreter -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir --convert-linalg-to-affine-loops -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir --canonicalize --cse -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/transformed.mlir
/usr/lib/llvm-18/bin/clang -O3 -march=native -ffp-contract=off -c tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/kernel.ll -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/kernel.o
```

high_decision diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/default/high_decision.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir`）：

```diff
--- before

+++ after

@@ -1,10 +1,10 @@

-#map = affine_map<(d0, d1, d2) -> (d0, d2)>
-#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
-#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
+#map = affine_map<(d0, d1, d2) -> (d0, d1)>
+#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
+#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
 module {
   func.func @kernel(%arg0: memref<127x93xf32>, %arg1: memref<93x65xf32>, %arg2: memref<127x65xf32>) attributes {llvm.emit_c_interface} {
     %cst = arith.constant 0.000000e+00 : f32
     linalg.fill ins(%cst : f32) outs(%arg2 : memref<127x65xf32>)
-    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<127x93xf32>, memref<93x65xf32>) outs(%arg2 : memref<127x65xf32>) {
+    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : memref<127x93xf32>, memref<93x65xf32>) outs(%arg2 : memref<127x65xf32>) {
     ^bb0(%in: f32, %in_0: f32, %out: f32):
       %0 = arith.mulf %in, %in_0 : f32
```

affine diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/default/affine.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir`）：

```diff
--- before

+++ after

@@ -8,12 +8,12 @@

     }
     affine.for %arg3 = 0 to 127 {
-      affine.for %arg4 = 0 to 65 {
-        affine.for %arg5 = 0 to 93 {
-          %0 = affine.load %arg0[%arg3, %arg5] : memref<127x93xf32>
-          %1 = affine.load %arg1[%arg5, %arg4] : memref<93x65xf32>
-          %2 = affine.load %arg2[%arg3, %arg4] : memref<127x65xf32>
+      affine.for %arg4 = 0 to 93 {
+        affine.for %arg5 = 0 to 65 {
+          %0 = affine.load %arg0[%arg3, %arg4] : memref<127x93xf32>
+          %1 = affine.load %arg1[%arg4, %arg5] : memref<93x65xf32>
+          %2 = affine.load %arg2[%arg3, %arg5] : memref<127x65xf32>
           %3 = arith.mulf %0, %1 : f32
           %4 = arith.addf %2, %3 : f32
-          affine.store %4, %arg2[%arg3, %arg4] : memref<127x65xf32>
+          affine.store %4, %arg2[%arg3, %arg5] : memref<127x65xf32>
         }
       }
```

transformed diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/default/transformed.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/transformed.mlir`）：

```diff
--- before

+++ after

@@ -8,12 +8,12 @@

     }
     affine.for %arg3 = 0 to 127 {
-      affine.for %arg4 = 0 to 65 {
-        affine.for %arg5 = 0 to 93 {
-          %0 = affine.load %arg0[%arg3, %arg5] : memref<127x93xf32>
-          %1 = affine.load %arg1[%arg5, %arg4] : memref<93x65xf32>
-          %2 = affine.load %arg2[%arg3, %arg4] : memref<127x65xf32>
+      affine.for %arg4 = 0 to 93 {
+        affine.for %arg5 = 0 to 65 {
+          %0 = affine.load %arg0[%arg3, %arg4] : memref<127x93xf32>
+          %1 = affine.load %arg1[%arg4, %arg5] : memref<93x65xf32>
+          %2 = affine.load %arg2[%arg3, %arg5] : memref<127x65xf32>
           %3 = arith.mulf %0, %1 : f32
           %4 = arith.addf %2, %3 : f32
-          affine.store %4, %arg2[%arg3, %arg4] : memref<127x65xf32>
+          affine.store %4, %arg2[%arg3, %arg5] : memref<127x65xf32>
         }
       }
```

kernel 函数反汇编 diff（截取）：

```diff
--- before

+++ after

@@ -1,86 +1,163 @@

 0000000000000000 <kernel>:
-       0: 41 57                        	pushq	%r15
-       2: 41 56                        	pushq	%r14
-       4: 41 54                        	pushq	%r12
-       6: 53                           	pushq	%rbx
-       7: 50                           	pushq	%rax
-       8: 48 89 f3                     	movq	%rsi, %rbx
-       b: 4c 8b 74 24 78               	movq	0x78(%rsp), %r14
-      10: 4c 8b 7c 24 40               	movq	0x40(%rsp), %r15
-      15: 45 31 e4                     	xorl	%r12d, %r12d
-      18: ba fc 80 00 00               	movl	$0x80fc, %edx           # imm = 0x80FC
-      1d: 4c 89 f7                     	movq	%r14, %rdi
-      20: 31 f6                        	xorl	%esi, %esi
-      22: e8 00 00 00 00               	callq	0x27 <kernel+0x27>
-		0000000000000023:  R_X86_64_PLT32	memset-0x4
-      27: 49 8d 47 04                  	leaq	0x4(%r15), %rax
-      2b: eb 17                        	jmp	0x44 <kernel+0x44>
-      2d: 0f 1f 00                     	nopl	(%rax)
-      30: 49 ff c4                     	incq	%r12
-      33: 48 81 c3 74 01 00 00         	addq	$0x174, %rbx            # imm = 0x174
-      3a: 49 83 fc 7f                  	cmpq	$0x7f, %r12
-      3e: 0f 84 17 01 00 00            	je	0x15b <kernel+0x15b>
-      44: 4c 89 e1                     	movq	%r12, %rcx
-      47: 48 c1 e1 08                  	shlq	$0x8, %rcx
-      4b: 4a 8d 0c a1                  	leaq	(%rcx,%r12,4), %rcx
-      4f: 4c 01 f1                     	addq	%r14, %rcx
-      52: 48 89 c2                     	movq	%rax, %rdx
-      55: 4c 89 fe                     	movq	%r15, %rsi
-      58: 31 ff                        	xorl	%edi, %edi
-      5a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
-      60: c5 fa 10 04 b9               	vmovss	(%rcx,%rdi,4), %xmm0
-      65: 41 b8 02 00 00 00            	movl	$0x2, %r8d
-      6b: 49 89 f1                     	movq	%rsi, %r9
-      6e: 66 90                        	nop
-      70: c4 a1 7a 10 4c 83 f8         	vmovss	-0x8(%rbx,%r8,4), %xmm1
-      77: c4 c1 72 59 09               	vmulss	(%r9), %xmm1, %xmm1
-      7c: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
-      80: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
-      85: c4 a1 7a 10 4c 83 fc         	vmovss	-0x4(%rbx,%r8,4), %xmm1
-      8c: c4 c1 72 59 89 04 01 00 00   	vmulss	0x104(%r9), %xmm1, %xmm1
-      95: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
-      99: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
-      9e: c4 a1 7a 10 0c 83            	vmovss	(%rbx,%r8,4), %xmm1
-      a4: c4 c1 72 59 89 08 02 00 00   	vmulss	0x208(%r9), %xmm1, %xmm1
-      ad: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
-      b1: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
-      b6: 49 83 c0 03                  	addq	$0x3, %r8
-      ba: 49 81 c1 0c 03 00 00         	addq	$0x30c, %r9             # imm = 0x30C
-      c1: 49 83 f8 5f                  	cmpq	$0x5f, %r8
-      c5: 75 a9                        	jne	0x70 <kernel+0x70>
-      c7: 49 89 f8                     	movq	%rdi, %r8
-      ca: 49 83 c8 01                  	orq	$0x1, %r8
-      ce: 49 83 f8 41                  	cmpq	$0x41, %r8
-      d2: 0f 84 58 ff ff ff            	je	0x30 <kernel+0x30>
-      d8: c4 a1 7a 10 04 81            	vmovss	(%rcx,%r8,4), %xmm0
-      de: 41 b9 02 00 00 00            	movl	$0x2, %r9d
-      e4: 49 89 d2                     	movq	%rdx, %r10
-      e7: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
-      f0: c4 a1 7a 10 4c 8b f8         	vmovss	-0x8(%rbx,%r9,4), %xmm1
-      f7: c4 c1 72 59 0a               	vmulss	(%r10), %xmm1, %xmm1
-      fc: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
-     100: c4 a1 7a 11 04 81            	vmovss	%xmm0, (%rcx,%r8,4)
-     106: c4 a1 7a 10 4c 8b fc         	vmovss	-0x4(%rbx,%r9,4), %xmm1
-     10d: c4 c1 72 59 8a 04 01 00 00   	vmulss	0x104(%r10), %xmm1, %xmm1
-     116: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
-     11a: c4 a1 7a 11 04 81            	vmovss	%xmm0, (%rcx,%r8,4)
-     120: c4 a1 7a 10 0c 8b            	vmovss	(%rbx,%r9,4), %xmm1
... 180 more diff lines; see local artifacts
```

default 首个浮点乘法附近：

```asm
      6b: 49 89 f1                     	movq	%rsi, %r9
      6e: 66 90                        	nop
      70: c4 a1 7a 10 4c 83 f8         	vmovss	-0x8(%rbx,%r8,4), %xmm1
      77: c4 c1 72 59 09               	vmulss	(%r9), %xmm1, %xmm1
      7c: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
      80: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
      85: c4 a1 7a 10 4c 83 fc         	vmovss	-0x4(%rbx,%r8,4), %xmm1
      8c: c4 c1 72 59 89 04 01 00 00   	vmulss	0x104(%r9), %xmm1, %xmm1
      95: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
      99: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
      9e: c4 a1 7a 10 0c 83            	vmovss	(%rbx,%r8,4), %xmm1
      a4: c4 c1 72 59 89 08 02 00 00   	vmulss	0x208(%r9), %xmm1, %xmm1
      ad: c5 fa 58 c1                  	vaddss	%xmm1, %xmm0, %xmm0
      b1: c5 fa 11 04 b9               	vmovss	%xmm0, (%rcx,%rdi,4)
      b6: 49 83 c0 03                  	addq	$0x3, %r8
      ba: 49 81 c1 0c 03 00 00         	addq	$0x30c, %r9             # imm = 0x30C
```

ikj-none 首个浮点乘法附近：

```asm
      f3: 49 c1 e5 06                  	shlq	$0x6, %r13
      f7: 4d 01 dd                     	addq	%r11, %r13
      fa: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     100: c4 81 7c 59 04 af            	vmulps	(%r15,%r13,4), %ymm0, %ymm0
     106: c4 81 7c 58 04 86            	vaddps	(%r14,%r8,4), %ymm0, %ymm0
     10c: c4 81 7c 11 04 86            	vmovups	%ymm0, (%r14,%r8,4)
     112: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     118: c4 81 7c 59 44 af 20         	vmulps	0x20(%r15,%r13,4), %ymm0, %ymm0
     11f: c4 81 7c 58 44 86 20         	vaddps	0x20(%r14,%r8,4), %ymm0, %ymm0
     126: c4 81 7c 11 44 86 20         	vmovups	%ymm0, 0x20(%r14,%r8,4)
     12d: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     133: c4 81 7c 59 44 af 40         	vmulps	0x40(%r15,%r13,4), %ymm0, %ymm0
     13a: c4 81 7c 58 44 86 40         	vaddps	0x40(%r14,%r8,4), %ymm0, %ymm0
     141: c4 81 7c 11 44 86 40         	vmovups	%ymm0, 0x40(%r14,%r8,4)
     148: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     14e: c4 81 7c 59 44 af 60         	vmulps	0x60(%r15,%r13,4), %ymm0, %ymm0
```

## matmul-127x65x93: ikj-none → ikj-tile16

median 0.076537 → 0.235122 ms/kernel。

`ikj-none` 的高层 schedule：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

编译命令：

```text
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high.mlir --transform-preload-library=transform-library-paths=tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/schedule.mlir --transform-interpreter -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir --convert-linalg-to-affine-loops -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir --canonicalize --cse -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/transformed.mlir
/usr/lib/llvm-18/bin/clang -O3 -march=native -ffp-contract=off -c tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/kernel.ll -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/kernel.o
```

`ikj-tile16` 的高层 schedule：

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %ordered = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

编译命令：

```text
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/high.mlir --transform-preload-library=transform-library-paths=tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/schedule.mlir --transform-interpreter -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/high_decision.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/high_decision.mlir --convert-linalg-to-affine-loops -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/affine.mlir
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/affine.mlir --affine-loop-tile=tile-size=16 --canonicalize --cse -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/transformed.mlir
/usr/lib/llvm-18/bin/clang -O3 -march=native -ffp-contract=off -c tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/kernel.ll -o tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/kernel.o
```

high_decision diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/high_decision.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/high_decision.mlir`）：

```diff
(identical)
```

affine diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/affine.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/affine.mlir`）：

```diff
(identical)
```

transformed diff（路径 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-none/transformed.mlir` 与 `tmp/20261018T101246162932Z/matmul-127x65x93/ikj-tile16/transformed.mlir`）：

```diff
--- before

+++ after

@@ -1,19 +1,33 @@

+#map = affine_map<(d0) -> (d0)>
+#map1 = affine_map<(d0) -> (d0 + 16, 127)>
+#map2 = affine_map<(d0) -> (d0 + 16, 65)>
+#map3 = affine_map<(d0) -> (d0 + 16, 93)>
 module {
   func.func @kernel(%arg0: memref<127x93xf32>, %arg1: memref<93x65xf32>, %arg2: memref<127x65xf32>) attributes {llvm.emit_c_interface} {
     %cst = arith.constant 0.000000e+00 : f32
-    affine.for %arg3 = 0 to 127 {
-      affine.for %arg4 = 0 to 65 {
-        affine.store %cst, %arg2[%arg3, %arg4] : memref<127x65xf32>
+    affine.for %arg3 = 0 to 127 step 16 {
+      affine.for %arg4 = 0 to 65 step 16 {
+        affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
+          affine.for %arg6 = #map(%arg4) to min #map2(%arg4) {
+            affine.store %cst, %arg2[%arg5, %arg6] : memref<127x65xf32>
+          }
+        }
       }
     }
-    affine.for %arg3 = 0 to 127 {
-      affine.for %arg4 = 0 to 93 {
-        affine.for %arg5 = 0 to 65 {
-          %0 = affine.load %arg0[%arg3, %arg4] : memref<127x93xf32>
-          %1 = affine.load %arg1[%arg4, %arg5] : memref<93x65xf32>
-          %2 = affine.load %arg2[%arg3, %arg5] : memref<127x65xf32>
-          %3 = arith.mulf %0, %1 : f32
-          %4 = arith.addf %2, %3 : f32
-          affine.store %4, %arg2[%arg3, %arg5] : memref<127x65xf32>
+    affine.for %arg3 = 0 to 127 step 16 {
+      affine.for %arg4 = 0 to 93 step 16 {
+        affine.for %arg5 = 0 to 65 step 16 {
+          affine.for %arg6 = #map(%arg3) to min #map1(%arg3) {
+            affine.for %arg7 = #map(%arg4) to min #map3(%arg4) {
+              affine.for %arg8 = #map(%arg5) to min #map2(%arg5) {
+                %0 = affine.load %arg0[%arg6, %arg7] : memref<127x93xf32>
+                %1 = affine.load %arg1[%arg7, %arg8] : memref<93x65xf32>
+                %2 = affine.load %arg2[%arg6, %arg8] : memref<127x65xf32>
+                %3 = arith.mulf %0, %1 : f32
+                %4 = arith.addf %2, %3 : f32
+                affine.store %4, %arg2[%arg6, %arg8] : memref<127x65xf32>
+              }
+            }
+          }
         }
       }
```

kernel 函数反汇编 diff（截取）：

```diff
--- before

+++ after

@@ -7,157 +7,830 @@

        9: 53                           	pushq	%rbx
        a: 48 83 ec 18                  	subq	$0x18, %rsp
-       e: 48 89 f3                     	movq	%rsi, %rbx
-      11: 4c 8b b4 24 98 00 00 00      	movq	0x98(%rsp), %r14
-      19: 4c 8b 7c 24 60               	movq	0x60(%rsp), %r15
-      1e: 45 31 e4                     	xorl	%r12d, %r12d
-      21: ba fc 80 00 00               	movl	$0x80fc, %edx           # imm = 0x80FC
-      26: 4c 89 f7                     	movq	%r14, %rdi
-      29: 31 f6                        	xorl	%esi, %esi
-      2b: e8 00 00 00 00               	callq	0x30 <kernel+0x30>
-		000000000000002c:  R_X86_64_PLT32	memset-0x4
-      30: 49 8d 87 74 5e 00 00         	leaq	0x5e74(%r15), %rax
-      37: 48 89 44 24 10               	movq	%rax, 0x10(%rsp)
-      3c: 49 8d 86 04 01 00 00         	leaq	0x104(%r14), %rax
-      43: 48 89 44 24 08               	movq	%rax, 0x8(%rsp)
-      48: 48 89 d9                     	movq	%rbx, %rcx
-      4b: 48 8d 93 74 01 00 00         	leaq	0x174(%rbx), %rdx
-      52: 4c 89 f6                     	movq	%r14, %rsi
-      55: eb 1d                        	jmp	0x74 <kernel+0x74>
-      57: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
-      60: 49 ff c4                     	incq	%r12
-      63: 48 81 c6 04 01 00 00         	addq	$0x104, %rsi            # imm = 0x104
-      6a: 49 83 fc 7f                  	cmpq	$0x7f, %r12
-      6e: 0f 84 b4 02 00 00            	je	0x328 <kernel+0x328>
-      74: 4c 89 e0                     	movq	%r12, %rax
-      77: 48 c1 e0 08                  	shlq	$0x8, %rax
-      7b: 4a 8d 04 a0                  	leaq	(%rax,%r12,4), %rax
-      7f: 4d 8d 0c 06                  	leaq	(%r14,%rax), %r9
-      83: 48 03 44 24 08               	addq	0x8(%rsp), %rax
-      88: 4d 69 d4 74 01 00 00         	imulq	$0x174, %r12, %r10      # imm = 0x174
-      8f: 4a 8d 3c 11                  	leaq	(%rcx,%r10), %rdi
-      93: 49 01 d2                     	addq	%rdx, %r10
-      96: 4d 89 e0                     	movq	%r12, %r8
-      99: 49 c1 e0 06                  	shlq	$0x6, %r8
-      9d: 4d 01 e0                     	addq	%r12, %r8
-      a0: 4d 39 d1                     	cmpq	%r10, %r9
-      a3: 41 0f 92 c2                  	setb	%r10b
-      a7: 48 39 c7                     	cmpq	%rax, %rdi
-      aa: 41 0f 92 c3                  	setb	%r11b
-      ae: 45 20 d3                     	andb	%r10b, %r11b
-      b1: 4c 3b 4c 24 10               	cmpq	0x10(%rsp), %r9
-      b6: 41 0f 92 c2                  	setb	%r10b
-      ba: 4c 39 f8                     	cmpq	%r15, %rax
-      bd: 41 0f 97 c1                  	seta	%r9b
-      c1: 45 20 d1                     	andb	%r10b, %r9b
-      c4: 45 08 d9                     	orb	%r11b, %r9b
-      c7: 4d 89 fa                     	movq	%r15, %r10
-      ca: 45 31 db                     	xorl	%r11d, %r11d
-      cd: eb 11                        	jmp	0xe0 <kernel+0xe0>
-      cf: 90                           	nop
-      d0: 49 ff c3                     	incq	%r11
-      d3: 49 81 c2 04 01 00 00         	addq	$0x104, %r10            # imm = 0x104
-      da: 49 83 fb 5d                  	cmpq	$0x5d, %r11
-      de: 74 80                        	je	0x60 <kernel+0x60>
-      e0: 45 84 c9                     	testb	%r9b, %r9b
-      e3: 74 0b                        	je	0xf0 <kernel+0xf0>
-      e5: 45 31 ed                     	xorl	%r13d, %r13d
-      e8: e9 0c 01 00 00               	jmp	0x1f9 <kernel+0x1f9>
-      ed: 0f 1f 00                     	nopl	(%rax)
-      f0: 4d 89 dd                     	movq	%r11, %r13
-      f3: 49 c1 e5 06                  	shlq	$0x6, %r13
-      f7: 4d 01 dd                     	addq	%r11, %r13
-      fa: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
-     100: c4 81 7c 59 04 af            	vmulps	(%r15,%r13,4), %ymm0, %ymm0
-     106: c4 81 7c 58 04 86            	vaddps	(%r14,%r8,4), %ymm0, %ymm0
-     10c: c4 81 7c 11 04 86            	vmovups	%ymm0, (%r14,%r8,4)
-     112: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
... 917 more diff lines; see local artifacts
```

ikj-none 首个浮点乘法附近：

```asm
      f3: 49 c1 e5 06                  	shlq	$0x6, %r13
      f7: 4d 01 dd                     	addq	%r11, %r13
      fa: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     100: c4 81 7c 59 04 af            	vmulps	(%r15,%r13,4), %ymm0, %ymm0
     106: c4 81 7c 58 04 86            	vaddps	(%r14,%r8,4), %ymm0, %ymm0
     10c: c4 81 7c 11 04 86            	vmovups	%ymm0, (%r14,%r8,4)
     112: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     118: c4 81 7c 59 44 af 20         	vmulps	0x20(%r15,%r13,4), %ymm0, %ymm0
     11f: c4 81 7c 58 44 86 20         	vaddps	0x20(%r14,%r8,4), %ymm0, %ymm0
     126: c4 81 7c 11 44 86 20         	vmovups	%ymm0, 0x20(%r14,%r8,4)
     12d: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     133: c4 81 7c 59 44 af 40         	vmulps	0x40(%r15,%r13,4), %ymm0, %ymm0
     13a: c4 81 7c 58 44 86 40         	vaddps	0x40(%r14,%r8,4), %ymm0, %ymm0
     141: c4 81 7c 11 44 86 40         	vmovups	%ymm0, 0x40(%r14,%r8,4)
     148: c4 a2 7d 18 04 9f            	vbroadcastss	(%rdi,%r11,4), %ymm0
     14e: c4 81 7c 59 44 af 60         	vmulps	0x60(%r15,%r13,4), %ymm0, %ymm0
```

ikj-tile16 首个浮点乘法附近：

```asm
     82c: 41 bd 00 00 00 00            	movl	$0x0, %r13d
     832: 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00    	nopw	%cs:(%rax,%rax)
     840: 62 a1 7e 08 10 04 a9         	vmovss	(%rcx,%r13,4), %xmm16
     847: 62 c1 7e 00 59 40 f1         	vmulss	-0x3c(%r8), %xmm16, %xmm16
     84e: 62 31 3e 08 58 c0            	vaddss	%xmm16, %xmm8, %xmm8
     854: c4 21 7a 11 04 90            	vmovss	%xmm8, (%rax,%r10,4)
     85a: 62 a1 7e 08 10 04 a9         	vmovss	(%rcx,%r13,4), %xmm16
     861: 62 c1 7e 00 59 40 f2         	vmulss	-0x38(%r8), %xmm16, %xmm16
     868: 62 31 36 08 58 c8            	vaddss	%xmm16, %xmm9, %xmm9
     86e: c4 21 7a 11 4c 90 04         	vmovss	%xmm9, 0x4(%rax,%r10,4)
     875: 62 a1 7e 08 10 04 a9         	vmovss	(%rcx,%r13,4), %xmm16
     87c: 62 c1 7e 00 59 40 f3         	vmulss	-0x34(%r8), %xmm16, %xmm16
     883: 62 31 2e 08 58 d0            	vaddss	%xmm16, %xmm10, %xmm10
     889: c4 21 7a 11 54 90 08         	vmovss	%xmm10, 0x8(%rax,%r10,4)
     890: 62 a1 7e 08 10 04 a9         	vmovss	(%rcx,%r13,4), %xmm16
     897: 62 c1 7e 00 59 40 f4         	vmulss	-0x30(%r8), %xmm16, %xmm16
```
