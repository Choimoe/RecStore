#include "storage/ssd/ssd_extendible_hash.h"
#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

class SSDExtendibleHashTest : public ::testing::Test {
protected:
  std::unique_ptr<Index> index_;

  fs::path tmp_dir_;

  void SetUp() override {
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    tmp_dir_ =
        fs::path(::testing::TempDir()) / ("ehssd_" + std::to_string(stamp));
    fs::create_directories(tmp_dir_);

    IndexConfig cfg;
    cfg.json_config_         = nlohmann::json::object();
    cfg.json_config_["path"] = tmp_dir_.string();

    index_ = std::make_unique<ExtendibleHashSSD>(cfg);
    ASSERT_NE(index_, nullptr)
        << "Failed to construct ExtendibleHashSSD with path=" << tmp_dir_;
  }

  void TearDown() override {
    index_.reset();

    std::error_code ec;
    fs::remove_all(tmp_dir_, ec);
  }
};

TEST_F(SSDExtendibleHashTest, BasicPutAndGet) {
  uint64_t key   = 123;
  uint64_t value = 456;
  unsigned tid   = 0;

  index_->Put(key, value, tid); // 插入
  uint64_t retrieved;
  index_->Get(key, retrieved, tid); // 检索

  EXPECT_EQ(retrieved, value)
      << "Failed to retrieve correct value for key " << key;
}

TEST_F(SSDExtendibleHashTest, GetNonExistentKey) {
  uint64_t key = 999;
  uint64_t retrieved;
  unsigned tid = 0;

  index_->Get(key, retrieved, tid);

  EXPECT_EQ(retrieved, 0)
      << "Non-existent key should return 0"; // 假设 0 为无效
}

TEST_F(SSDExtendibleHashTest, PutOverwrite) {
  uint64_t key    = 100;
  uint64_t value1 = 200;
  uint64_t value2 = 300;
  unsigned tid    = 0;

  index_->Put(key, value1, tid);
  index_->Put(key, value2, tid);

  uint64_t retrieved;
  index_->Get(key, retrieved, tid);
  EXPECT_EQ(retrieved, value2) << "Failed to overwrite value for key " << key;
}

TEST_F(SSDExtendibleHashTest, BatchGetWithCoroutine) {
  const int num_pairs = 100;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> values(num_pairs);
  unsigned tid = 0;

  for (int i = 0; i < num_pairs; i++) {
    keys[i]   = i + 5000;
    values[i] = i * 50;
    index_->Put(keys[i], values[i], tid);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  std::vector<uint64_t> retrieved_values(num_pairs);
  bool sink_called = false;
  int count        = 0;
  boost::coroutines2::coroutine<void>::push_type sink(
      [this, &sink_called, &count](
          boost::coroutines2::coroutine<void>::pull_type& yield) {
        const int batch_size = 32;
        for (int i = 1; i <= 32; i++) {
          yield();
          sink_called = true;
          count++;
        }
      });

  index_->BatchGet(sink, keys_array, retrieved_values.data(), tid);
  LOG(INFO) << "count: " << count;
  EXPECT_TRUE(sink_called) << "Sink should be called at least once";
  for (int i = 0; i < num_pairs; i++) {
    EXPECT_EQ(retrieved_values[i], values[i]) << "Failed for key " << keys[i];
  }
}

TEST_F(SSDExtendibleHashTest, BatchGetWithoutSink) {
  const int num_pairs = 50;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> values(num_pairs);
  unsigned tid = 0;

  for (int i = 0; i < num_pairs; i++) {
    keys[i]   = i + 2000;
    values[i] = i * 30;
    index_->Put(keys[i], values[i], tid);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  std::vector<uint64_t> retrieved_values(num_pairs);
  index_->BatchGet(keys_array, retrieved_values.data(), tid);

  for (int i = 0; i < num_pairs; i++) {
    EXPECT_EQ(retrieved_values[i], values[i]) << "Failed for key " << keys[i];
  }
}

TEST_F(SSDExtendibleHashTest, BatchPutWithCoroutine) {
  const int num_pairs = 100;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> pointers(num_pairs);
  unsigned tid = 0;

  for (int i = 0; i < num_pairs; i++) {
    keys[i]     = i + 6000;
    pointers[i] = i * 60; // 模拟 pointers 值
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  bool sink_called = false;
  int count        = 0;

  boost::coroutines2::coroutine<void>::push_type sink(
      [this, &sink_called, &count](
          boost::coroutines2::coroutine<void>::pull_type& yield) {
        const int batch_size = 32;
        for (int i = 1; i <= 10; i++) { // 计算预期批次次数
          yield();
          sink_called = true;
          count++;
        }
      });

  index_->BatchPut(sink, keys_array, pointers.data(), tid);

  EXPECT_TRUE(sink_called) << "Sink should be called at least once";
  LOG(INFO) << "count: " << count;
  for (int i = 0; i < num_pairs; i++) {
    uint64_t retrieved;
    index_->Get(keys[i], retrieved, tid);
    EXPECT_EQ(retrieved, pointers[i]) << "Failed for key " << keys[i];
  }
}

TEST_F(SSDExtendibleHashTest, BulkLoadTest) {
  const int num_keys = 50;
  std::vector<uint64_t> keys(num_keys);
  std::vector<Value_t> values(num_keys);
  unsigned tid = 0;

  for (int i = 0; i < num_keys; i++) {
    keys[i]   = i + 3000;
    values[i] = i; // 示例值
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_keys);
  index_->BulkLoad(keys_array, values.data());

  for (int i = 0; i < num_keys; i++) {
    uint64_t retrieved;
    index_->Get(keys[i], retrieved, tid);
    EXPECT_EQ(retrieved, values[i]) << "Failed for key " << keys[i];
  }
}

TEST_F(SSDExtendibleHashTest, LoadFakeDataTest) {
  const int64_t key_capacity = 50;
  const int value_size       = sizeof(uint64_t); // 假设 value_size 为指针大小
  unsigned tid               = 0;

  index_->LoadFakeData(key_capacity, value_size);

  for (int64_t i = 0; i < key_capacity; i++) {
    uint64_t retrieved;
    index_->Get(i, retrieved, tid);
    EXPECT_EQ(retrieved, i) << "Failed to load fake data for key " << i;
  }
}

#include <random>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <limits>

TEST_F(SSDExtendibleHashTest, EdgeKeysAndOverwrite) {
  unsigned tid        = 0;
  const uint64_t k0   = 0;
  const uint64_t kmax = std::numeric_limits<uint64_t>::max();

  index_->Put(k0, 111, tid);
  index_->Put(kmax, 222, tid);

  uint64_t got = 0;
  index_->Get(k0, got, tid);
  EXPECT_EQ(got, 111u);

  index_->Get(kmax, got, tid);
  EXPECT_EQ(got, 222u);

  index_->Put(k0, 333, tid);
  index_->Get(k0, got, tid);
  EXPECT_EQ(got, 333u);
}

TEST_F(SSDExtendibleHashTest, RandomizedInsertOverwriteCheck) {
  unsigned tid = 0;
  std::mt19937_64 rng(123456);
  std::uniform_int_distribution<uint64_t> kdist(1, 10'000'000);
  std::uniform_int_distribution<int> action(0, 9);

  std::unordered_map<uint64_t, uint64_t> oracle;
  const int ops = 5000;

  for (int i = 0; i < ops; ++i) {
    uint64_t k = kdist(rng);
    uint64_t v = rng() & 0xffffffffull;
    if (action(rng) <= 5 || !oracle.count(k)) {
      index_->Put(k, v, tid);
      oracle[k] = v;
    } else {
      uint64_t nv = v ^ 0x5a5a5a5aULL;
      index_->Put(k, nv, tid);
      oracle[k] = nv;
    }
  }

  for (int i = 0; i < 200; ++i) {
    uint64_t k   = kdist(rng);
    uint64_t got = 0;
    index_->Get(k, got, tid);
    auto it = oracle.find(k);
    if (it == oracle.end()) {
      EXPECT_EQ(got, 0u) << "non-existent key should be 0";
    } else {
      EXPECT_EQ(got, it->second) << "mismatch at key " << k;
    }
  }
}

TEST_F(SSDExtendibleHashTest, MixedConcurrencyPutGet) {
  const int WRITERS        = 8;
  const int READERS        = 8;
  const int OPS_PER_WRITER = 5000;

  std::atomic<bool> start{false};
  std::vector<std::thread> ths;

  for (int w = 0; w < WRITERS; ++w) {
    ths.emplace_back([&, w] {
      while (!start.load(std::memory_order_acquire)) {
      }
      const uint64_t base = 1'000'000ull + w * 100'000ull;
      for (int i = 0; i < OPS_PER_WRITER; ++i) {
        uint64_t k = base + i;
        uint64_t v = (k << 1) ^ 0x9e3779b97f4a7c15ULL;
        index_->Put(k, v, /*tid=*/w);
      }
    });
  }

  std::mt19937_64 rng(2025);
  for (int r = 0; r < READERS; ++r) {
    ths.emplace_back([&, r] {
      while (!start.load(std::memory_order_acquire)) {
      }
      std::uniform_int_distribution<uint64_t> pick(
          1'000'000ull, 1'000'000ull + WRITERS * 100'000ull - 1);
      for (int i = 0; i < OPS_PER_WRITER; ++i) {
        uint64_t k   = pick(rng);
        uint64_t got = 0;
        index_->Get(k, got, /*tid=*/r);
        if (got != 0) {
          uint64_t expected = (k << 1) ^ 0x9e3779b97f4a7c15ULL;
          EXPECT_EQ(got, expected) << "key=" << k;
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& t : ths)
    t.join();

  for (int i = 0; i < OPS_PER_WRITER; ++i) {
    uint64_t k   = 1'000'000ull + 0 * 100'000ull + i;
    uint64_t got = 0;
    index_->Get(k, got, 0);
    uint64_t expected = (k << 1) ^ 0x9e3779b97f4a7c15ULL;
    EXPECT_EQ(got, expected);
  }
}

TEST_F(SSDExtendibleHashTest, BulkLoadLarge) {
  const int N = 2000;
  std::vector<uint64_t> keys(N), vals(N);
  for (int i = 0; i < N; ++i) {
    keys[i] = 700000 + i;
    vals[i] = i * 7 + 1;
  }
  base::ConstArray<uint64_t> kar(keys.data(), N);

  index_->BulkLoad(kar, vals.data());

  std::mt19937_64 rng(7);
  std::uniform_int_distribution<int> id(0, N - 1);
  for (int t = 0; t < 200; ++t) {
    int i        = id(rng);
    uint64_t got = 0;
    index_->Get(keys[i], got, /*tid=*/0);
    EXPECT_EQ(got, vals[i]) << "key=" << keys[i];
  }
}
