// Validation only: record full SQLite results without computing a digest.
#include "sqlite3.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static void write_bytes(FILE *out, const void *data, std::size_t size) {
  if (size && std::fwrite(data, 1, size, out) != size) std::exit(2);
}
extern "C" int irloom_step(sqlite3_stmt *statement) {
  static FILE *out = [] {
    const char *path = std::getenv("IRLOOM_SQL_RESULTS");
    FILE *file = path ? std::fopen(path, "wb") : nullptr;
    if (!file) std::exit(2);
    return file;
  }();
  const int rc = sqlite3_step(statement);
  write_bytes(out, &rc, sizeof(rc));
  if (rc == SQLITE_ROW) {
    const int columns = sqlite3_column_count(statement);
    write_bytes(out, &columns, sizeof(columns));
    for (int i = 0; i < columns; ++i) {
      const int type = sqlite3_column_type(statement, i);
      write_bytes(out, &type, sizeof(type));
      if (type == SQLITE_INTEGER) {
        const auto value = sqlite3_column_int64(statement, i);
        write_bytes(out, &value, sizeof(value));
      } else if (type == SQLITE_FLOAT) {
        const double value = sqlite3_column_double(statement, i);
        write_bytes(out, &value, sizeof(value));
      } else if (type != SQLITE_NULL) {
        const void *value = type == SQLITE_BLOB ? sqlite3_column_blob(statement, i)
                                              : sqlite3_column_text(statement, i);
        const std::uint64_t bytes = sqlite3_column_bytes(statement, i);
        write_bytes(out, &bytes, sizeof(bytes));
        write_bytes(out, value, bytes);
      }
    }
  }
  if (std::ferror(out) || std::fflush(out)) std::exit(2);
  return rc;
}
