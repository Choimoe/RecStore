#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "storage/kv_engine/hps_recstore_backend.h"

namespace {

recstore::storage::HpsRecStoreBackendParams MakeParams(
    const std::string& path) {
  recstore::storage::HpsRecStoreBackendParams params;
  params.path              = path;
  params.capacity          = 16;
  params.value_size        = sizeof(float) * 2;
  params.num_threads       = 1;
  params.index_type        = "DRAM_EXTENDIBLE_HASH";
  params.value_store_type  = "DRAM_VALUE_STORE";
  params.dram_allocator    = "PERSIST_LOOP_SLAB";
  params.dram_capacity_bytes = 4096;
  return params;
}

} // namespace

TEST(HpsRecStoreBackendTest, InsertFetchAndMissCallback) {
  const std::string path = "/tmp/test_hps_recstore_backend_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);

  recstore::storage::HpsRecStoreBackend<long long> backend(MakeParams(path));
  const std::string table = "model.table";
  const long long keys[]  = {1, 2};
  const float values[]    = {1.0f, 1.5f, 2.0f, 2.5f};

  EXPECT_EQ(backend.insert(table,
                           2,
                           keys,
                           reinterpret_cast<const char*>(values),
                           sizeof(float) * 2,
                           sizeof(float) * 2),
            2);

  const long long query[] = {2, 3, 1};
  float out[]             = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<size_t> misses;
  const size_t hits = backend.fetch(
      table,
      3,
      query,
      reinterpret_cast<char*>(out),
      sizeof(float) * 2,
      [&](size_t index) {
        misses.push_back(index);
        out[index * 2]     = 0.0f;
        out[index * 2 + 1] = 0.0f;
      },
      std::chrono::nanoseconds::zero());

  EXPECT_EQ(hits, 2);
  ASSERT_EQ(misses, std::vector<size_t>({1}));
  EXPECT_FLOAT_EQ(out[0], 2.0f);
  EXPECT_FLOAT_EQ(out[1], 2.5f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
  EXPECT_FLOAT_EQ(out[4], 1.0f);
  EXPECT_FLOAT_EQ(out[5], 1.5f);

  std::filesystem::remove_all(path);
}
