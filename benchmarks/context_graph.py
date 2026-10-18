"""One shared producer, three surrounding graphs; no hand-written candidate kernels."""

CONTEXTS=("single","transpose","row_reduce")


def target_region(shape):
    m,n=shape
    t=f"tensor<{m}x{n}xf32>"
    return f'''    // IRLoom target begin
    %d = linalg.generic {{
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%ta, %tb : {t}, {t}) outs(%empty : {t}) {{
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %product = arith.mulf %x, %y : f32
      linalg.yield %product : f32
    }} -> {t}
    // IRLoom target end
'''


def generate(context,shape):
    if context not in CONTEXTS:
        raise ValueError(f"unsupported graph context: {context}")
    m,n=shape
    if context=="transpose" and m!=n:
        raise ValueError("transpose context requires square input")
    t=f"tensor<{m}x{n}xf32>"
    row=f"tensor<{m}x1xf32>"
    mem=f"memref<{m}x{n}xf32>"
    source=f'''module {{
  func.func @kernel(%a: {mem}, %b: {mem}, %c: {mem}) attributes {{llvm.emit_c_interface}} {{
    %ta = bufferization.to_tensor %a restrict : {mem}
    %tb = bufferization.to_tensor %b restrict : {mem}
    %empty = tensor.empty() : {t}
'''+target_region(shape)+f'''
    %y = linalg.generic {{
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%d, %ta : {t}, {t}) outs(%empty : {t}) {{
    ^bb0(%p: f32, %x: f32, %unused: f32):
      %value = arith.addf %p, %x : f32
      linalg.yield %value : f32
    }} -> {t}
'''
    if context=="row_reduce":
        source+=f'''    %zero = arith.constant 0.0 : f32
    %rowempty = tensor.empty() : {row}
    %init = linalg.fill ins(%zero : f32) outs(%rowempty : {row}) -> {row}
    %z = linalg.generic {{
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,0)>],
      iterator_types = ["parallel", "reduction"]
    }} ins(%d : {t}) outs(%init : {row}) {{
    ^bb0(%p: f32, %acc: f32):
      %sum = arith.addf %acc, %p : f32
      linalg.yield %sum : f32
    }} -> {row}
'''
    if context!="single":
        other,typ,index=("d",t,"(j,i)") if context=="transpose" else ("z",row,"(i,0)")
        source+=f'''    %result = linalg.generic {{
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->{index}>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%y, %{other} : {t}, {typ}) outs(%empty : {t}) {{
    ^bb0(%yv: f32, %zv: f32, %unused: f32):
      %sum = arith.addf %yv, %zv : f32
      linalg.yield %sum : f32
    }} -> {t}
'''
    result="y" if context=="single" else "result"
    source+=f'''    %out = bufferization.to_memref %{result} : {mem}
    memref.copy %out, %c : {mem} to {mem}
    return
  }}
}}
'''
    return source,[f"-DM={m}",f"-DN={n}","-DK=1",f"-DKIND={4+CONTEXTS.index(context)}"]
