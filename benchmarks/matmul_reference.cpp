// Conventional contiguous-inner-loop matmul reference, not a tuned BLAS.
// Identical row-major layout, output reset, and per-element k accumulation order.
#include "memref.h"

extern "C" void _mlir_ciface_kernel(Matrix *a, Matrix *b, Matrix *c) {
  for (int i = 0; i < M*N; ++i) c->aligned[i] = 0.0f;
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float value = a->aligned[i*K+k];
      for (int j = 0; j < N; ++j)
        c->aligned[i*N+j] += value * b->aligned[k*N+j];
    }
  }
}
