module {
  func.func @kernel(%a: memref<32x32xf32>, %b: memref<32x32xf32>,
                    %c: memref<32x32xf32>) attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<32x32xf32>)
    linalg.matmul ins(%a, %b : memref<32x32xf32>, memref<32x32xf32>)
                  outs(%c : memref<32x32xf32>)
    return
  }
}
