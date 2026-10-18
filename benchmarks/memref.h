#ifndef IRLOOM_MEMREF_H
#define IRLOOM_MEMREF_H
#include <stdint.h>

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} Matrix;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_ciface_kernel(Matrix *, Matrix *, Matrix *);
#ifdef __cplusplus
}
#endif
#endif
