// Measure a complete native process independently of the Python driver's RSS.
#include <chrono>
#include <cerrno>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 3) return 2; // metrics.json executable [arguments...]
  const auto start = std::chrono::steady_clock::now();
  const pid_t parent = getpid();
  const pid_t child = fork();
  if (child < 0) return 2;
  if (child == 0) {
    // A compiler-session timeout must not leave the timed workload running.
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != parent) _exit(125);
    execv(argv[2], argv + 2);
    _exit(127);
  }
  int status = 0;
  rusage usage{};
  while (wait4(child, &status, 0, &usage) < 0) {
    if (errno != EINTR) return 2;
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  std::ofstream out(argv[1]);
  out << std::setprecision(12) << "{\"latency_ms\":" << ms
      << ",\"max_rss_kib\":" << usage.ru_maxrss << ",\"exit_code\":" << code
      << ",\"user_s\":" << usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6
      << ",\"system_s\":" << usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6 << "}\n";
  out.close();
  return out ? code : 2;
}
