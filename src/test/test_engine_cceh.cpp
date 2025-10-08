#include <condition_variable>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <string>

#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/engine_cceh.h"

class KVEngineCCEHTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = "/tmp/test_kv_engine_cceh_" + std::to_string(getpid());
    std::filesystem::create_directories(test_dir_);
    base::PMMmapRegisterCenter::GetConfig().use_dram = true;
    base::PMMmapRegisterCenter::GetConfig().numa_id  = 0;
    config_.num_threads_                             = 16;
    config_.json_config_                             = {
        {"path", test_dir_},
        {"capacity", 100000},
        {"value_size", 128},
        {"queue_size", 512}};
    kv_engine_ = std::make_unique<KVEngineCCEH>(config_);
  }

  void TearDown() override {
    kv_engine_.reset();
    std::filesystem::remove_all(test_dir_);
  }

  std::string CreateFixedLengthValue(const std::string& base_value) {
    std::string value = base_value;
    value.resize(128);
    return value;
  }

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

// // 基本的Put和Get测试
// TEST_F(KVEngineCCEHTest, BasicPutAndGet) {
//   uint64_t key      = 123;
//   std::string value = CreateFixedLengthValue("test_value_123");
//   std::string retrieved_value;
//   kv_engine_->Put(key, value, 0);
//   kv_engine_->Get(key, retrieved_value, 0);
//   EXPECT_EQ(retrieved_value, value);
// }

// // 测试多个键值对
// TEST_F(KVEngineCCEHTest, MultiplePutAndGet) {
//   const int num_pairs = 500;
//   std::vector<std::pair<uint64_t, std::string>> test_data;
//   for (int i = 1; i <= num_pairs; i++)
//     test_data.emplace_back(i, CreateFixedLengthValue("value_" +
//     std::to_string(i)));
//   for (const auto& pair : test_data)
//     kv_engine_->Put(pair.first, pair.second, 0);
//   for (const auto& pair : test_data) {
//     std::string retrieved_value;
//     kv_engine_->Get(pair.first, retrieved_value, 0);
//     EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " <<
//     pair.first;
//   }
// }

// // 测试BatchGet功能
// TEST_F(KVEngineCCEHTest, BatchGet) {
//   const int num_keys = 512;
//   int cnt            = 0;
//   std::vector<uint64_t> keys;
//   std::vector<std::string> expected_values;
//   for (int i = 0; i < num_keys; i++) {
//     keys.push_back(i);
//     expected_values.push_back(CreateFixedLengthValue("batch_value_" +
//     std::to_string(i))); kv_engine_->Put(i, expected_values[i], 0);
//   }
//   base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
//   std::vector<base::ConstArray<float>> batch_values;
//   kv_engine_->BatchGet(keys_array, &batch_values, 0);
//   EXPECT_EQ(batch_values.size(), num_keys) << "Failed size\n";
//   for (int i = 0; i < num_keys; i++) {
//     if (batch_values[i].Size() > 0) {
//       std::string retrieved_value((char*)batch_values[i].Data(),
//       batch_values[i].Size() * sizeof(float)); size_t null_pos =
//       retrieved_value.find('\0'); if (null_pos != std::string::npos)
//         retrieved_value = retrieved_value.substr(0, null_pos);
//       std::string expected_original = "batch_value_" + std::to_string(i);
//       EXPECT_EQ(retrieved_value, expected_original) << "Failed for key " <<
//       i;
//     } else {
//       std::string expected_original = "batch_value_" + std::to_string(i);
//       EXPECT_EQ("", expected_original) << "Failed for key " << i;
//     }
//   }
// }

// TEST_F(KVEngineCCEHTest, ConcurrentBatchGet) {
//   const int num_keys_per_thread = 512;
//   const int num_threads         = 16;
//   const int total_keys          = num_keys_per_thread * num_threads;
//   for (int i = 0; i < total_keys; i++) {
//     std::string value =
//         CreateFixedLengthValue("concurrent_value_" + std::to_string(i));
//     kv_engine_->Put(i, value, 0);
//   }
//   std::vector<std::vector<std::string>> thread_results(num_threads);
//   std::vector<std::string> thread_errors(num_threads);
//   std::vector<std::thread> threads;
//   SimpleBarrier barrier(num_threads);
//   for (int tid = 0; tid < num_threads; tid++) {
//     threads.emplace_back([&, tid]() {
//       try {
//         barrier.wait();
//         std::vector<uint64_t> keys;
//         for (int i = tid * num_keys_per_thread;
//              i < (tid + 1) * num_keys_per_thread;
//              i++)
//           keys.push_back(i);
//         base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
//         std::vector<base::ConstArray<float>> batch_values;
//         kv_engine_->BatchGet(keys_array, &batch_values, 0);
//         for (int i = 0; i < num_keys_per_thread; i++) {
//           if (batch_values[i].Size() > 0) {
//             std::string retrieved_value((char*)batch_values[i].Data(),
//                                         batch_values[i].Size() *
//                                         sizeof(float));
//             size_t null_pos = retrieved_value.find('\0');
//             if (null_pos != std::string::npos)
//               retrieved_value = retrieved_value.substr(0, null_pos);
//             thread_results[tid].push_back(retrieved_value);
//           } else {
//             thread_results[tid].push_back("");
//           }
//         }
//       } catch (const std::exception& e) {
//         thread_errors[tid] = e.what();
//       }
//     });
//   }
//   for (auto& t : threads) {
//     t.join();
//   }
//   for (int tid = 0; tid < num_threads; tid++) {
//     EXPECT_TRUE(thread_errors[tid].empty())
//         << "Thread " << tid << " error: " << thread_errors[tid];
//     EXPECT_EQ(thread_results[tid].size(), num_keys_per_thread)
//         << "Thread " << tid << " result count mismatch";

//     for (int i = 0; i < num_keys_per_thread; i++) {
//       int global_key       = tid * num_keys_per_thread + i;
//       std::string expected = "concurrent_value_" +
//       std::to_string(global_key); EXPECT_EQ(thread_results[tid][i], expected)
//           << "Thread " << tid << " key " << global_key << " value mismatch";
//     }
//   }
// }

TEST_F(KVEngineCCEHTest, UpdateOverwritesExistingValue) {
  uint64_t key         = 424242;
  std::string original = CreateFixedLengthValue("original_value");
  std::string updated  = CreateFixedLengthValue("updated_value");
  std::string retrieved;

  kv_engine_->Put(key, original, 0);
  kv_engine_->Get(key, retrieved, 0);
  EXPECT_EQ(retrieved, original);

  kv_engine_->Put(key, updated, 0);
  kv_engine_->Get(key, retrieved, 0);
  EXPECT_EQ(retrieved, updated);
}