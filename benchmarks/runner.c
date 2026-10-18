#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M
#define M 32
#define N 32
#define K 32
#endif

typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} Matrix;

extern void _mlir_ciface_kernel(Matrix *, Matrix *, Matrix *);

static Matrix matrix(int rows, int cols) {
  float *p = calloc((size_t)rows * cols, sizeof(float));
  if (!p) { perror("calloc"); exit(2); }
  return (Matrix){p, p, 0, {rows, cols}, {cols, 1}};
}

static void fill(Matrix a, unsigned seed) {
  for (int64_t i = 0; i < a.sizes[0] * a.sizes[1]; ++i)
    a.aligned[i] = (float)((int)((i * 17 + seed * 13) % 65) - 32) / 32.0f;
}

static double seconds(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: runner check|measure seed iterations\n"); return 2; }
  unsigned seed = (unsigned)strtoul(argv[2], NULL, 10);
  long iterations = strtol(argv[3], NULL, 10);
  if (iterations < 1) return 2;
  Matrix a = matrix(M, K), b = matrix(K, N), c = matrix(M, N);
  fill(a, seed); fill(b, seed + 1);
  _mlir_ciface_kernel(&a, &b, &c);
  double max_error = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double expected = 0;
      for (int k = 0; k < K; ++k)
        expected += (double)a.aligned[i*K+k] * b.aligned[k*N+j];
      double error = fabs((double)c.aligned[i*N+j] - expected);
      if (!isfinite(c.aligned[i*N+j]) || error > 1e-4 + 1e-4 * fabs(expected)) {
        fprintf(stderr, "incorrect output at %d,%d: %.9g vs %.9g\n", i, j, c.aligned[i*N+j], expected);
        return 3;
      }
      if (error > max_error) max_error = error;
    }
  }
  if (!strcmp(argv[1], "check")) {
    printf("{\"status\":\"pass\",\"max_abs_error\":%.9g}\n", max_error);
  } else if (!strcmp(argv[1], "measure")) {
    for (int i = 0; i < 3; ++i) _mlir_ciface_kernel(&a, &b, &c);
    double start = seconds();
    for (long i = 0; i < iterations; ++i) _mlir_ciface_kernel(&a, &b, &c);
    double elapsed = seconds() - start;
    /* Observable output; kernel is compiled separately without LTO. */
    printf("{\"elapsed_ms\":%.9g,\"iterations\":%ld,\"latency_ms\":%.9g,\"output\":%.9g}\n",
           elapsed * 1e3, iterations, elapsed * 1e3 / iterations, c.aligned[0]);
  } else { return 2; }
  free(a.allocated); free(b.allocated); free(c.allocated);
  return 0;
}
