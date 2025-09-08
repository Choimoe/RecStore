#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/json.h"
#include "memory/shm_file.h"  // for PMMmapRegisterCenter config
#include "storage/kv_engine/engine_hybridkv.h"

class KVEngineHybridTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 测试目录
    test_dir_ = "/tmp/test_kv_engine_hybrid_" + std::to_string(getpid());
    std::filesystem::create_directories(test_dir_);

    base::PMMmapRegisterCenter::GetConfig().use_dram = false;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;

    config_.num_threads_ = 16;
    config_.json_config_ = {
        {"path", test_dir_},
        {"shmcapacity", 40 * 1024 * 1024},   // 4MB
        {"ssdcapacity", 80 * 1024 * 1024}    // 8MB
    };

    kv_ = std::make_unique<KVEngineHybrid>(config_);
  }

  void TearDown() override {
    kv_.reset();
    std::filesystem::remove_all(test_dir_);
  }

  // 生成指定长度的 value（内容可控）
  static std::string MakeValue(const std::string& prefix, size_t total_len) {
    std::string s = prefix;
    if (s.size() < total_len) s.resize(total_len, 'x');
    else s.resize(total_len);
    return s;
  }

  // 将 BatchGet 返回的 float 数组还原为字符串
  static std::string FloatsToString(const base::ConstArray<float>& arr) {
    std::string out;
    out.reserve(arr.Size());
    for (int i = 0; i < arr.Size(); ++i) {
      // 存的时候是将每个 char 按 unsigned char → float 存的
      auto byte = static_cast<unsigned char>(static_cast<int>(arr[i]));
      out.push_back(static_cast<char>(byte));
    }
    return out;
  }

  // 简单屏障
  class SimpleBarrier {
  public:
    explicit SimpleBarrier(int n) : n_(n), cur_(0) {}
    void wait() {
      std::unique_lock<std::mutex> lk(mu_);
      if (++cur_ == n_) cv_.notify_all();
      else cv_.wait(lk, [&]{ return cur_ == n_; });
    }
  private:
    int n_, cur_;
    std::mutex mu_;
    std::condition_variable cv_;
  };

  std::string test_dir_;
  BaseKVConfig config_;
  std::unique_ptr<KVEngineHybrid> kv_;
};

// 基础 Put / Get
TEST_F(KVEngineHybridTest, BasicPutAndGet) {
  uint64_t key = 123;
  std::string v = MakeValue("test_value_123", 32);
  std::string got;

  kv_->Put(key, v, 0);
  kv_->Get(key, got, 0);

  EXPECT_EQ(got, v);
}

// 不存在的键
TEST_F(KVEngineHybridTest, GetNonExistentKey) {
  std::string got;
  kv_->Get(999999, got, 0);
  EXPECT_TRUE(got.empty());
}

// 覆盖写
TEST_F(KVEngineHybridTest, KeyOverwrite) {
  uint64_t key = 100;
  std::string v1 = MakeValue("initial", 24);
  std::string v2 = MakeValue("updated", 40);
  std::string got;

  kv_->Put(key, v1, 0);
  kv_->Get(key, got, 0);
  EXPECT_EQ(got, v1);

  kv_->Put(key, v2, 0);
  kv_->Get(key, got, 0);
  EXPECT_EQ(got, v2);
}

// 多键 Put / Get
TEST_F(KVEngineHybridTest, MultiplePutAndGet) {
  const int N = 64;
  for (int i = 0; i < N; ++i) {
    kv_->Put(i, MakeValue("v" + std::to_string(i), 16 + (i % 8)), 0);
  }
  for (int i = 0; i < N; ++i) {
    std::string got;
    kv_->Get(i, got, 0);
    EXPECT_FALSE(got.empty()) << "key=" << i;
    EXPECT_EQ(got.substr(0, 1), "v");
  }
}

// 批量 BatchGet（存在的键）
TEST_F(KVEngineHybridTest, BatchGet) {
  const int N = 100;
  std::vector<uint64_t> keys;
  keys.reserve(N);
  std::vector<std::string> expect(N);

  for (int i = 0; i < N; ++i) {
    keys.push_back(i);
    expect[i] = "batch_" + std::to_string(i);
    kv_->Put(i, expect[i], 0);
  }

  base::ConstArray<uint64_t> ka(keys.data(), keys.size());
  std::vector<base::ConstArray<float>> vals;
  kv_->BatchGet(ka, &vals, 0);

  ASSERT_EQ(vals.size(), keys.size());
  for (int i = 0; i < N; ++i) {
    auto s = FloatsToString(vals[i]);
    EXPECT_EQ(s, expect[i]) << "key=" << i;
  }
}

// 批量 BatchGet（不存在的键）
TEST_F(KVEngineHybridTest, BatchGetNonExistent) {
  std::vector<uint64_t> keys = {1001, 1002, 1003};
  base::ConstArray<uint64_t> ka(keys.data(), keys.size());
  std::vector<base::ConstArray<float>> vals;
  kv_->BatchGet(ka, &vals, 0);

  ASSERT_EQ(vals.size(), keys.size());
  for (auto &arr : vals) {
    EXPECT_EQ(arr.Size(), 0);
  }
}

// 批量 BatchGet（混合键）
TEST_F(KVEngineHybridTest, BatchGetMixed) {
  kv_->Put(1, "a", 0);
  kv_->Put(3, "ccc", 0);
  kv_->Put(5, "eeeee", 0);

  std::vector<uint64_t> keys = {1,2,3,4,5,6};
  base::ConstArray<uint64_t> ka(keys.data(), keys.size());
  std::vector<base::ConstArray<float>> vals;
  kv_->BatchGet(ka, &vals, 0);

  ASSERT_EQ(vals.size(), keys.size());
  EXPECT_GT(vals[0].Size(), 0); 
  EXPECT_EQ(vals[1].Size(), 0); 
  EXPECT_GT(vals[2].Size(), 0); 
  EXPECT_EQ(vals[3].Size(), 0); 
  EXPECT_GT(vals[4].Size(), 0); 
  EXPECT_EQ(vals[5].Size(), 0); 
}

// 边界值：空串 / 长串 / 特殊字符与\0
TEST_F(KVEngineHybridTest, BoundaryValues) {
  std::string got;

  kv_->Put(1, "", 0);
  kv_->Get(1, got, 0);
  EXPECT_EQ(got, "");

  std::string longv(512, 'x');
  kv_->Put(2, longv, 0);
  kv_->Get(2, got, 0);
  EXPECT_EQ(got, longv);

  std::string special = std::string("Hello\nWorld\t", 12) + std::string("\0ABC", 4);
  kv_->Put(3, special, 0);
  kv_->Get(3, got, 0);
  EXPECT_EQ(got.size(), special.size());
  EXPECT_EQ(memcmp(got.data(), special.data(), special.size()), 0);
}

// 特殊键：0 与超大
TEST_F(KVEngineHybridTest, SpecialKeys) {
  std::string v = "test_value";
  std::string got;

  kv_->Put(0, v, 0);
  kv_->Get(0, got, 0);
  EXPECT_EQ(got, v);

  uint64_t big = std::numeric_limits<uint64_t>::max() - 123;
  kv_->Put(big, v, 0);
  kv_->Get(big, got, 0);
  EXPECT_EQ(got, v);
}

// 随机数据：最后一次写入应为最终值
TEST_F(KVEngineHybridTest, RandomData) {
  std::mt19937_64 gen(12345);
  std::uniform_int_distribution<uint64_t> kdist(1, 1000);
  std::uniform_int_distribution<int> lenDist(1, 64);

  const int ops = 2000;
  std::unordered_map<uint64_t, std::string> expect;

  for (int i = 0; i < ops; ++i) {
    auto key = kdist(gen);
    int L = lenDist(gen);
    std::string base;
    base.reserve(L);
    for (int j = 0; j < L; ++j) base.push_back('a' + (gen() % 26));
    kv_->Put(key, base, 0);
    expect[key] = base;
  }

  for (auto &p : expect) {
    std::string got;
    kv_->Get(p.first, got, 0);
    EXPECT_EQ(got, p.second) << "key=" << p.first;
  }
}

// 并发 Put
TEST_F(KVEngineHybridTest, ConcurrentPut) {
  const int T = 8;
  const int per = 500;
  SimpleBarrier bar(T);
  std::vector<std::thread> ths;
  std::atomic<int> fail{0};

  for (int t = 0; t < T; ++t) {
    ths.emplace_back([&, t]{
      bar.wait();
      for (int i = 0; i < per; ++i) {
        uint64_t key = uint64_t(t) * per + i;
        std::string v = "thread_" + std::to_string(t) + "_" + std::to_string(i);
        try { kv_->Put(key, v, 0); }
        catch (...) { fail++; }
      }
    });
  }
  for (auto &th : ths) th.join();

  EXPECT_EQ(fail.load(), 0);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < per; ++i) {
      uint64_t key = uint64_t(t) * per + i;
      std::string got;
      kv_->Get(key, got, 0);
      EXPECT_NE(got.find("thread_" + std::to_string(t) + "_"), std::string::npos)
          << "key=" << key;
    }
  }
}

// 并发 Get
TEST_F(KVEngineHybridTest, ConcurrentGet) {
  const int N = 200;
  for (int i = 0; i < N; ++i) kv_->Put(i, "cg_" + std::to_string(i), 0);

  const int T = 8;
  const int R = 1000;
  SimpleBarrier bar(T);
  std::vector<std::thread> ths;
  std::atomic<int> ok{0}, bad{0};

  for (int t = 0; t < T; ++t) {
    ths.emplace_back([&, t]{
      std::mt19937 gen(777 + t);
      std::uniform_int_distribution<int> d(0, N - 1);
      bar.wait();
      for (int i = 0; i < R; ++i) {
        uint64_t k = d(gen);
        std::string got;
        try {
          kv_->Get(k, got, 0);
          if (got.empty()) bad++; else ok++;
        } catch (...) { bad++; }
      }
    });
  }
  for (auto &th : ths) th.join();

  EXPECT_EQ(bad.load(), 0);
  EXPECT_EQ(ok.load(), T * R);
}

// 并发读写混合
TEST_F(KVEngineHybridTest, ConcurrentReadWrite) {
  const int T = 8;
  const int OPS = 800;
  SimpleBarrier bar(T);
  std::vector<std::thread> ths;
  std::atomic<int> ok{0}, bad{0};

  for (int t = 0; t < T; ++t) {
    ths.emplace_back([&, t]{
      std::mt19937 gen(2024 + t);
      std::uniform_int_distribution<int> op(0,1);
      std::uniform_int_distribution<uint64_t> kd(0, 199);
      bar.wait();
      for (int i = 0; i < OPS; ++i) {
        uint64_t k = kd(gen);
        if (op(gen) == 0) {
          std::string v = "mix_t" + std::to_string(t) + "_" + std::to_string(i);
          try { kv_->Put(k, v, 0); ok++; } catch (...) { bad++; }
        } else {
          std::string got;
          try { kv_->Get(k, got, 0); ok++; } catch (...) { bad++; }
        }
      }
    });
  }
  for (auto &th : ths) th.join();

  EXPECT_EQ(bad.load(), 0);
  EXPECT_EQ(ok.load(), T * OPS);
}

// shm触发淘汰后的一致性
TEST_F(KVEngineHybridTest, EvictionConsistency) {
  const int N = 4000;
  for (int i = 0; i < N; ++i) {
    kv_->Put(i, "evict_" + std::to_string(i), 0);
  }
  // 读取检查
  for (int i = 0; i < N; ++i) {
    std::string got;
    kv_->Get(i, got, 0);
    EXPECT_EQ(got, "evict_" + std::to_string(i)) << "key=" << i;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}