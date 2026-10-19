// Small source-built LLVM component workload; compiler bootstrap is deliberately separate.
#include "llvm/Demangle/Demangle.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
  const long iterations = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 200000;
  if (iterations < 1 || llvm::demangle("_Z3fooi") != "foo(int)") return 3;
  const char *names[] = {"_Z3fooi", "_Z3barPKc", "_ZN3foo3barEi"};
  size_t characters = 0;
  const auto start = std::chrono::steady_clock::now();
  for (long i = 0; i < iterations; ++i) characters += llvm::demangle(names[i % 3]).size();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if (characters == 0) return 3;
  std::printf("{\"iterations\":%ld,\"characters\":%zu,\"seconds\":%.9f}\n", iterations, characters, seconds);
}
