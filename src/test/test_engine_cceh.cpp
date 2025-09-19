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
#include "memory/shm_file.h"
#include "storage/kv_engine/engine_cceh.h"

class KVEngineCCEHTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 创建临时测试目录
    test_dir_ = "/tmp/test_kv_engine_cceh_" + std::to_string(getpid());
    std::filesystem::create_directories(test_dir_);

    // 配置使用DRAM而不是持久内存
    base::PMMmapRegisterCenter::GetConfig().use_dram = false;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;

    // 创建配置
    config_.num_threads_ = 16;
    config_.json_config_ = {
        {"path", test_dir_}, {"capacity", 100000}, {"value_size", 128}};

    // 创建KV引擎实例
    kv_engine_ = std::make_unique<KVEngineCCEH>(config_);
    LOG(INFO) << "done!";
  }

  void TearDown() override {
    // 清理测试
    kv_engine_.reset();

    // 删除临时测试目录
    std::filesystem::remove_all(test_dir_);
  }

  // 辅助函数：创建固定长度的value
  std::string CreateFixedLengthValue(const std::string &base_value) {
    std::string value = base_value;
    value.resize(128); // 确保value长度为128字节
    return value;
  }

  // 简单的barrier实现，用于同步多线程测试
  class SimpleBarrier {
  public:
    explicit SimpleBarrier(int count) : count_(count), current_(0) {}

    void wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      ++current_;
      if (current_ == count_) {
        condition_.notify_all();
      } else {
        condition_.wait(lock, [this] { return current_ == count_; });
      }
    }

  private:
    int count_;
    int current_;
    std::mutex mutex_;
    std::condition_variable condition_;
  };

  std::string test_dir_;
  BaseKVConfig config_;
  std::unique_ptr<KVEngineCCEH> kv_engine_;
};

// 基本的Put和Get测试
TEST_F(KVEngineCCEHTest, BasicPutAndGet) {
  uint64_t key = 123;
  std::string value = CreateFixedLengthValue("test_value_123");
  std::string retrieved_value;

  // 测试Put操作
  kv_engine_->Put(key, value, 0);

  // 测试Get操作
  kv_engine_->Get(key, retrieved_value, 0);

  EXPECT_EQ(retrieved_value, value);
}

// 测试多个键值对
TEST_F(KVEngineCCEHTest, MultiplePutAndGet) {
  const int num_pairs = 50;
  std::vector<std::pair<uint64_t, std::string>> test_data;

  // 准备测试数据
  for (int i = 1; i <= num_pairs; i++) {
    test_data.emplace_back(
        i, CreateFixedLengthValue("value_" + std::to_string(i)));
  }

  // 插入数据
  for (const auto &pair : test_data) {
    kv_engine_->Put(pair.first, pair.second, 0);
  }

  // 验证数据
  for (const auto &pair : test_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}

// 测试BatchGet功能
TEST_F(KVEngineCCEHTest, BatchGet) {
  const int num_keys = 10000;
  int cnt = 0;
  std::vector<uint64_t> keys;
  std::vector<std::string> expected_values;

  // 准备测试数据
  for (int i = 0; i < num_keys; i++) {
    keys.push_back(i);
    expected_values.push_back(
        CreateFixedLengthValue("batch_value_" + std::to_string(i)));
    kv_engine_->Put(i, expected_values[i], 0);
  }

  // 创建keys数组
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  // 执行BatchGet
  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  // 验证结果
  EXPECT_EQ(batch_values.size(), num_keys) << "Failed size\n";

  for (int i = 0; i < num_keys; i++) {
    if (batch_values[i].Size() > 0) {
      // 将float数组转换回字符串进行比较
      std::string retrieved_value((char *)batch_values[i].Data(),
                                  batch_values[i].Size() * sizeof(float));
      // 由于存储的是字符串，我们需要截断到实际字符串长度
      size_t null_pos = retrieved_value.find('\0');
      if (null_pos != std::string::npos) {
        retrieved_value = retrieved_value.substr(0, null_pos);
      }

      // 创建期望值的原始字符串（不包含填充）
      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ(retrieved_value, expected_original) << "Failed for key " << i;
    } else {
      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ("", expected_original) << "Failed for key " << i;
    }
  }
}

TEST_F(KVEngineCCEHTest, ConcurrentBatchGet) {
  const int num_keys_per_thread = 1000;
  const int num_threads = 16;
  const int total_keys = num_keys_per_thread * num_threads;

  // 准备测试数据 - 先插入所有数据
  for (int i = 0; i < total_keys; i++) {
    std::string value =
        CreateFixedLengthValue("concurrent_value_" + std::to_string(i));
    kv_engine_->Put(i, value, 0);
  }

  // 用于收集结果和错误
  std::vector<std::vector<std::string>> thread_results(num_threads);
  std::vector<std::string> thread_errors(num_threads);
  std::vector<std::thread> threads;
  SimpleBarrier barrier(num_threads);

  // 启动多个线程并发执行BatchGet
  for (int tid = 0; tid < num_threads; tid++) {
    threads.emplace_back([&, tid]() {
      try {
        // 等待所有线程准备就绪
        barrier.wait();
        // 每个线程处理不同的key范围
        std::vector<uint64_t> keys;
        for (int i = tid * num_keys_per_thread;
             i < (tid + 1) * num_keys_per_thread; i++) {
          keys.push_back(i);
        }

        // 创建keys数组并执行BatchGet
        base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
        std::vector<base::ConstArray<float>> batch_values;
        kv_engine_->BatchGet(keys_array, &batch_values, 0);

        // 验证结果
        for (int i = 0; i < num_keys_per_thread; i++) {
          if (batch_values[i].Size() > 0) {
            std::string retrieved_value((char *)batch_values[i].Data(),
                                        batch_values[i].Size() * sizeof(float));
            size_t null_pos = retrieved_value.find('\0');
            if (null_pos != std::string::npos) {
              retrieved_value = retrieved_value.substr(0, null_pos);
            }
            thread_results[tid].push_back(retrieved_value);
          } else {
            thread_results[tid].push_back("");
          }
        }
      } catch (const std::exception &e) {
        thread_errors[tid] = e.what();
      }
    });
  }
  // 等待所有线程完成
  for (auto &t : threads) {
    t.join();
  }

  // 验证结果
  for (int tid = 0; tid < num_threads; tid++) {
    EXPECT_TRUE(thread_errors[tid].empty())
        << "Thread " << tid << " error: " << thread_errors[tid];
    EXPECT_EQ(thread_results[tid].size(), num_keys_per_thread)
        << "Thread " << tid << " result count mismatch";

    for (int i = 0; i < num_keys_per_thread; i++) {
      int global_key = tid * num_keys_per_thread + i;
      std::string expected = "concurrent_value_" + std::to_string(global_key);
      EXPECT_EQ(thread_results[tid][i], expected)
          << "Thread " << tid << " key " << global_key << " value mismatch";
    }
  }
}