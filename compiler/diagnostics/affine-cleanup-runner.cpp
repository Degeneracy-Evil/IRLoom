#include <cstdint>
#include <cstdio>
#include <cstdlib>
extern "C" int64_t same_init(int64_t);
extern "C" int64_t distinct_init(int64_t);
extern "C" int64_t bound_alias(int64_t);
extern "C" int64_t constant_odd();
extern "C" int64_t constant_even();
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const int64_t n = std::strtoll(argv[1], nullptr, 10);
  if (n < 1 || n > 300) return 2;
  std::printf("%lld %lld %lld %lld %lld\n", (long long)same_init(n),
              (long long)distinct_init(n), (long long)bound_alias(n),
              (long long)constant_odd(), (long long)constant_even());
}
