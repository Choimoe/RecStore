#include <gtest/gtest.h>

#include <vector>

#include "base/random.h"
#include "naiveKVell.h"
using ssdps::NaiveArraySSD;

using base::ConstArray;

TEST(NaiveArraySSD, test) {
  constexpr int emb_dim = 32;
  // int64_t test_key_capability = 10 * 1e6;
  int64_t test_key_capability = 1693782;

  NaiveArraySSD<uint64_t> ssd(emb_dim * 4, test_key_capability, 10);

  std::vector<uint64_t> keys;
  std::vector<float> values;
  keys.reserve(test_key_capability);
  values.reserve(test_key_capability * emb_dim);

  for (int i = 0; i < test_key_capability; i++) {
    keys.push_back(i);
    for (int j = 0; j < emb_dim; j++)
      values.push_back(i);
  }
  ConstArray<uint64_t> keys_array(keys);

  ssd.BulkLoad(keys_array.Size(), values.data());

  LOG(INFO) << "bulk load finished";

  constexpr int batch_get_num = 2000;
  std::vector<uint64_t> test_get_keys(batch_get_num);
  ConstArray<uint64_t> test_get_keys_array(test_get_keys);
  ConstArray<uint64_t> index_array;

  std::vector<float> test_get_values(batch_get_num * emb_dim);

  for (int _ = 0; _ < 1000; _++) {
    for (int i = 0; i < batch_get_num; i++) {
      test_get_keys[i] =
          base::Random::rand32(static_cast<uint32_t>(test_key_capability));
    }
    xmh::Timer timer("get");
    ssd.BatchGet(
        test_get_keys_array, index_array, (void*)test_get_values.data(), 3);
    timer.end();
    for (int i = 0; i < batch_get_num; i++) {
      for (int j = 0; j < emb_dim; j++) {
        CHECK_NEAR(test_get_keys[i], test_get_values[i * emb_dim + j], 1e-6);
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  xmh::Reporter::StartReportThread(1000);
  return RUN_ALL_TESTS();
}
