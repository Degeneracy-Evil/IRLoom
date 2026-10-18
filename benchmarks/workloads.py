"""Static-shape MLIR inputs; all candidate changes are compiler passes."""


def generate(program, shape):
    m, n = shape[:2]
    k = shape[2] if program == "matmul" else 1
    if program == "matmul":
        a, b, c = f"{m}x{k}", f"{k}x{n}", f"{m}x{n}"
        body = f"""
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<{c}xf32>)
    linalg.matmul ins(%a, %b : memref<{a}xf32>, memref<{b}xf32>)
                  outs(%c : memref<{c}xf32>)"""
        kind = 0
    elif program == "reduction":
        a, b, c = f"{m}x{n}", "1x1", f"{m}x1"
        body = f"""
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<{c}xf32>)
    linalg.generic {{
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,0)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%a : memref<{a}xf32>) outs(%c : memref<{c}xf32>) {{
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %acc, %x : f32
      linalg.yield %sum : f32
    }}"""
        kind = 1
    elif program == "stencil":
        a, b, c = f"{m+2}x{n+2}", "1x1", f"{m}x{n}"
        body = f"""
    %scale = arith.constant 0.2 : f32
    affine.for %i = 0 to {m} {{
      affine.for %j = 0 to {n} {{
        %center = affine.load %a[%i+1, %j+1] : memref<{a}xf32>
        %left = affine.load %a[%i+1, %j] : memref<{a}xf32>
        %right = affine.load %a[%i+1, %j+2] : memref<{a}xf32>
        %up = affine.load %a[%i, %j+1] : memref<{a}xf32>
        %down = affine.load %a[%i+2, %j+1] : memref<{a}xf32>
        %s1 = arith.addf %center, %left : f32
        %s2 = arith.addf %s1, %right : f32
        %s3 = arith.addf %s2, %up : f32
        %s4 = arith.addf %s3, %down : f32
        %value = arith.mulf %s4, %scale : f32
        affine.store %value, %c[%i, %j] : memref<{c}xf32>
      }}
    }}"""
        kind = 2
    else:
        raise ValueError(f"unsupported workload: {program}")
    source = f"""module {{
  func.func @kernel(%a: memref<{a}xf32>, %b: memref<{b}xf32>, %c: memref<{c}xf32>)
      attributes {{llvm.emit_c_interface}} {{
{body}
    return
  }}
}}
"""
    return source, [f"-DM={m}", f"-DN={n}", f"-DK={k}", f"-DKIND={kind}"]
