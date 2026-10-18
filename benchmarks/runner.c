#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "memref.h"

#ifndef M
#define M 32
#define N 32
#define K 32
#endif
#ifndef KIND
#define KIND 0
#endif

#define GUARD 16
#define CANARY 123456.0f

static Matrix matrix(int rows, int cols) {
  size_t count = (size_t)rows * cols;
  float *p = calloc(count + 2*GUARD, sizeof(float));
  if (!p) { perror("calloc"); exit(2); }
  for (int i = 0; i < GUARD; ++i) p[i] = p[GUARD+count+i] = CANARY;
  return (Matrix){p, p+GUARD, 0, {rows, cols}, {cols, 1}};
}

static void fill(Matrix a, unsigned seed) {
  for (int64_t i = 0; i < a.sizes[0] * a.sizes[1]; ++i)
    a.aligned[i] = seed == 0 ? 0.0f : (float)((int)((i * 17 + seed * 13) % 65) - 32) / 32.0f;
}

static int guards_ok(Matrix a) {
  size_t count = (size_t)a.sizes[0] * a.sizes[1];
  for (int i = 0; i < GUARD; ++i)
    if (a.allocated[i] != CANARY || a.aligned[count+i] != CANARY) return 0;
  return 1;
}

/* Independent scalar references, accumulated in double without fast-math. */
static double reference(Matrix a, Matrix b, int i, int j) {
#if KIND == 0
  double result = 0;
  for (int k = 0; k < K; ++k) result += (double)a.aligned[i*K+k] * b.aligned[k*N+j];
  return result;
#elif KIND == 1
  double result = 0;
  for (int k = 0; k < N; ++k) result += (double)a.aligned[i*N+k];
  return result;
#elif KIND == 2
  int width = N+2;
  return ((double)a.aligned[(i+1)*width+j+1] + a.aligned[(i+1)*width+j]
          + a.aligned[(i+1)*width+j+2] + a.aligned[i*width+j+1]
          + a.aligned[(i+2)*width+j+1]) * 0.2;
#elif KIND == 5
  return ((double)a.aligned[i*N+j]*b.aligned[i*N+j] + a.aligned[i*N+j])
         + (double)a.aligned[j*N+i]*b.aligned[j*N+i];
#else
  return (double)a.aligned[i*N+j]*b.aligned[i*N+j] + a.aligned[i*N+j];
#endif
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
#if KIND == 0
  Matrix a = matrix(M, K), b = matrix(K, N), c = matrix(M, N);
#elif KIND == 1
  Matrix a = matrix(M, N), b = matrix(1, 1), c = matrix(M, 1);
#elif KIND == 2
  Matrix a = matrix(M+2, N+2), b = matrix(1, 1), c = matrix(M, N);
#else
  Matrix a = matrix(M, N), b = matrix(M, N), c = matrix(M, N);
#endif
  fill(a, seed); fill(b, seed + 1);
  /* NaNs detect incomplete output writes, including all-zero input checks. */
  for (int64_t i = 0; i < c.sizes[0]*c.sizes[1]; ++i) c.aligned[i] = NAN;
  double *expected = calloc((size_t)c.sizes[0]*c.sizes[1], sizeof(double));
  if (!expected) { perror("calloc"); return 2; }
  for (int i = 0; i < M; ++i) {
#if KIND == 6
    double row_sum = 0;
    for (int j = 0; j < N; ++j)
      row_sum += (double)a.aligned[i*N+j] * b.aligned[i*N+j];
#endif
    for (int j = 0; j < c.sizes[1]; ++j) {
      expected[i*c.sizes[1]+j] = reference(a, b, i, j);
#if KIND == 6
      expected[i*c.sizes[1]+j] += row_sum;
#endif
    }
  }
  _mlir_ciface_kernel(&a, &b, &c);
  double max_error = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < c.sizes[1]; ++j) {
      double ref = expected[i*c.sizes[1]+j];
      float actual = c.aligned[i*c.sizes[1]+j];
      double error = fabs((double)actual - ref);
      if (!isfinite(actual) || error > 1e-4 + 1e-4 * fabs(ref)) {
        fprintf(stderr, "incorrect output at %d,%d: %.9g vs %.9g\n", i, j, actual, ref);
        return 3;
      }
      if (error > max_error) max_error = error;
    }
  }
  if (!guards_ok(a) || !guards_ok(b) || !guards_ok(c)) {
    fprintf(stderr, "buffer guard modified\n"); return 3;
  }
  if (!strcmp(argv[1], "check")) {
    printf("{\"status\":\"pass\",\"max_abs_error\":%.9g}\n", max_error);
  } else if (!strcmp(argv[1], "measure")) {
    for (int i = 0; i < 3; ++i) _mlir_ciface_kernel(&a, &b, &c);
    double start = seconds();
    for (long i = 0; i < iterations; ++i) _mlir_ciface_kernel(&a, &b, &c);
    double elapsed = seconds() - start;
    if (!guards_ok(a) || !guards_ok(b) || !guards_ok(c)) return 3;
    for (int64_t i = 0; i < c.sizes[0]*c.sizes[1]; ++i)
      if (!isfinite(c.aligned[i]) || fabs((double)c.aligned[i]-expected[i]) > 1e-4 + 1e-4*fabs(expected[i]))
        return 3;
    /* Observable output; kernel is compiled separately without LTO. */
    printf("{\"elapsed_ms\":%.9g,\"iterations\":%ld,\"latency_ms\":%.9g,\"output\":%.9g}\n",
           elapsed * 1e3, iterations, elapsed * 1e3 / iterations, c.aligned[0]);
  } else { return 2; }
  free(expected); free(a.allocated); free(b.allocated); free(c.allocated);
  return 0;
}
