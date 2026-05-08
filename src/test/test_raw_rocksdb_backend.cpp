#include "storage/external/hps/raw_rocksdb.h"

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

std::string TempDbPath(const std::string& name) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(path);
  return path.string();
}

} // namespace

TEST(RawRocksDBBackendTest, WriteBatchThenMultiGetReturnsFixedWidthValues) {
  const std::string path = TempDbPath("recstore_raw_rocksdb_backend_test");
  recstore::storage::RawRocksDBBackend backend(path, 4);

  const long long keys[]         = {11, 22, 33};
  const std::vector<char> values = {
      'a',
      'a',
      'a',
      'a',
      'b',
      'b',
      'b',
      'b',
      'c',
      'c',
      'c',
      'c',
  };

  backend.Insert(3, keys, values.data());

  const long long query_keys[] = {33, 44, 11};
  std::vector<char> out(3 * 4, '?');
  size_t misses = 0;
  backend.Fetch(3, query_keys, out.data(), [&](size_t index) {
    EXPECT_EQ(1U, index);
    ++misses;
  });

  EXPECT_EQ(1U, misses);
  EXPECT_EQ(std::string("cccc"), std::string(out.data(), 4));
  EXPECT_EQ(std::string("????"), std::string(out.data() + 4, 4));
  EXPECT_EQ(std::string("aaaa"), std::string(out.data() + 8, 4));

  std::filesystem::remove_all(path);
}

TEST(RawRocksDBBackendTest, MemEnvKeepsDatabaseFilesOffRealFilesystem) {
  const std::string path = TempDbPath("recstore_raw_rocksdb_memenv_test");
  recstore::storage::RawRocksDBBackend backend(path, 4, true);

  const long long keys[] = {7};
  const char values[]    = {'m', 'e', 'm', '0'};
  backend.Insert(1, keys, values);

  std::vector<char> out(4, '?');
  size_t misses = 0;
  backend.Fetch(1, keys, out.data(), [&](size_t) { ++misses; });

  EXPECT_EQ(0U, misses);
  EXPECT_EQ(std::string("mem0"), std::string(out.data(), out.size()));
  EXPECT_FALSE(std::filesystem::exists(path));
}
