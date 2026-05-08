#include "storage/external/fasterkv/fasterkv_backend.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(FasterKVBackendTest, InsertThenFetchReturnsFixedWidthValuesAndMisses) {
  recstore::storage::fasterkv::FasterKVBackend backend(128, 4);

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
}

TEST(FasterKVBackendTest, InsertCanOverwriteExistingFixedWidthValue) {
  recstore::storage::fasterkv::FasterKVBackend backend(128, 4);

  const long long key    = 7;
  const char old_value[] = {'o', 'l', 'd', '0'};
  const char new_value[] = {'n', 'e', 'w', '0'};
  backend.Insert(1, &key, old_value);
  backend.Insert(1, &key, new_value);

  std::vector<char> out(4, '?');
  size_t misses = 0;
  backend.Fetch(1, &key, out.data(), [&](size_t) { ++misses; });

  EXPECT_EQ(0U, misses);
  EXPECT_EQ(std::string("new0"), std::string(out.data(), out.size()));
}
