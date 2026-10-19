// Fixed-input SQLite API workload. Every query has an integer reference answer.
#include "sqlite3.h"
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

static double cpu_seconds() {
  timespec t{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) throw std::runtime_error("CPU clock failed");
  return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}
static void check(int rc, sqlite3 *db) {
  if (rc != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(db));
}
static void exec(sqlite3 *db, const char *sql) {
  check(sqlite3_exec(db, sql, nullptr, nullptr, nullptr), db);
}
int main(int argc, char **argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: sqlite-workload point|range|train SEED N QUERIES_OR_CPU_SECONDS");
    const std::string mode = argv[1];
    if (mode != "point" && mode != "range" && mode != "train") throw std::runtime_error("unknown workload");
    const auto seed = std::stoul(argv[2]);
    const int n = std::stoi(argv[3]);
    if (n != 100003) throw std::runtime_error("protocol requires N=100003");
    const bool training = mode == "train";
    const bool range = mode == "range";
    const std::uint64_t queries = training ? 0 : std::stoull(argv[4]);
    const double cpu_budget = training ? std::stod(argv[4]) : 0;
    const double cpu_start = cpu_seconds();
    const auto started = std::chrono::steady_clock::now();
    sqlite3 *db = nullptr;
    check(sqlite3_open(":memory:", &db), db);
    exec(db, "CREATE TABLE t(k INTEGER NOT NULL,v INTEGER NOT NULL); BEGIN;");
    sqlite3_stmt *insert = nullptr;
    check(sqlite3_prepare_v2(db, "INSERT INTO t VALUES(?1,?2)", -1, &insert, nullptr), db);
    const int offset = seed % 97;
    for (int i = 0; i < n; ++i) {
      const int k = (std::int64_t(i) * 37) % n;
      sqlite3_bind_int(insert, 1, k);
      sqlite3_bind_int(insert, 2, 3 * k + offset);
      if (sqlite3_step(insert) != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(db));
      check(sqlite3_reset(insert), db);
    }
    check(sqlite3_finalize(insert), db);
    exec(db, "COMMIT; CREATE UNIQUE INDEX lookup ON t(k);");
    sqlite3_stmt *integrity = nullptr;
    check(sqlite3_prepare_v2(db, "PRAGMA integrity_check", -1, &integrity, nullptr), db);
    if (sqlite3_step(integrity) != SQLITE_ROW || std::string(reinterpret_cast<const char *>(sqlite3_column_text(integrity, 0))) != "ok")
      throw std::runtime_error("database integrity failed");
    check(sqlite3_finalize(integrity), db);
    const char *sql = range ? "SELECT count(*),sum(v) FROM t WHERE k>=?1 AND k<?2" : "SELECT v FROM t WHERE k=?1";
    sqlite3_stmt *plan = nullptr;
    const std::string explain = std::string("EXPLAIN QUERY PLAN ") + sql;
    check(sqlite3_prepare_v2(db, explain.c_str(), -1, &plan, nullptr), db);
    std::string plan_text;
    while (sqlite3_step(plan) == SQLITE_ROW) plan_text += reinterpret_cast<const char *>(sqlite3_column_text(plan, 3));
    check(sqlite3_finalize(plan), db);
    if (plan_text.find("lookup") == std::string::npos) throw std::runtime_error("expected explicit index lookup");
    sqlite3_stmt *query = nullptr;
    check(sqlite3_prepare_v2(db, sql, -1, &query, nullptr), db);
    const auto begin = std::chrono::steady_clock::now();
    const double setup = std::chrono::duration<double>(begin-started).count();
    std::mt19937 rng(seed);
    std::uint64_t i = 0;
    std::int64_t result_sum = 0;
    while (training ? (i % 1000 != 0 || cpu_seconds()-cpu_start < cpu_budget) : i < queries) {
      const int k = rng() % (range ? n-32 : n);
      sqlite3_bind_int(query, 1, k);
      if (range) sqlite3_bind_int(query, 2, k+32);
      if (sqlite3_step(query) != SQLITE_ROW) throw std::runtime_error("expected query result");
      const std::int64_t expected = range ? 3LL*(32LL*k+32LL*31/2)+32LL*offset : 3LL*k+offset;
      const std::int64_t actual = sqlite3_column_int64(query, range ? 1 : 0);
      if (actual != expected || (range && sqlite3_column_int64(query, 0) != 32))
        throw std::runtime_error("integer reference answer mismatch");
      if (sqlite3_step(query) != SQLITE_DONE) throw std::runtime_error("unexpected additional row");
      check(sqlite3_reset(query), db);
      result_sum += actual;
      ++i;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    check(sqlite3_finalize(query), db);
    check(sqlite3_close(db), db);
    std::cout << "{\"mode\":\"" << mode << "\",\"seed\":" << seed << ",\"rows\":" << n
              << ",\"verified_queries\":" << i << ",\"value_sum\":" << result_sum
              << ",\"compute_seconds\":" << elapsed << ",\"setup_seconds\":" << setup
              << ",\"process_cpu_seconds\":" << cpu_seconds()-cpu_start
              << ",\"sqlite_version\":\"" << sqlite3_libversion()
              << "\",\"sqlite_source_id\":\"" << sqlite3_sourceid() << "\""
              << ",\"plan\":\"" << plan_text << "\"}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
