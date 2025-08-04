#include "storage/dram/extendible_hash.h"
#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>


class IndexTest : public ::testing::Test {
protected:
  void SetUp() override {
    IndexConfig config;  // 假设默认配置
    index_ = new ExtendibleHash(config);  // 初始化为 ExtendibleHash 实例
  }

  void TearDown() override {
    delete index_;  
  }

  Index* index_;  
};

// 基本 Put 和 Get 测试
TEST_F(IndexTest, BasicPutAndGet) {
  uint64_t key = 123;
  uint64_t value = 456;
  unsigned tid = 0;

  index_->Put(key, value, tid);  // 插入
  uint64_t retrieved;
  index_->Get(key, retrieved, tid);  // 检索

  EXPECT_EQ(retrieved, value) << "Failed to retrieve correct value for key " << key;
}

// 测试不存在的键
TEST_F(IndexTest, GetNonExistentKey) {
  uint64_t key = 999;
  uint64_t retrieved;
  unsigned tid = 0;

  index_->Get(key, retrieved, tid);

  EXPECT_EQ(retrieved, 0) << "Non-existent key should return 0";  // 假设 0 为无效
}

// 测试键值覆盖
TEST_F(IndexTest, PutOverwrite) {
  uint64_t key = 100;
  uint64_t value1 = 200;
  uint64_t value2 = 300;
  unsigned tid = 0;

  index_->Put(key, value1, tid);
  index_->Put(key, value2, tid);

  uint64_t retrieved;
  index_->Get(key, retrieved, tid);
  EXPECT_EQ(retrieved, value2) << "Failed to overwrite value for key " << key;
}

// 测试 BatchGet (有协程版)
TEST_F(IndexTest, BatchGetWithCoroutine) {
  const int num_pairs = 100;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> values(num_pairs);
  unsigned tid = 0;

  // 插入数据
  for (int i = 0; i < num_pairs; i++) {
    keys[i] = i + 5000;
    values[i] = i * 50;
    index_->Put(keys[i], values[i], tid);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  std::vector<uint64_t> retrieved_values(num_pairs);
  bool sink_called = false;
  int count = 0;
  // 创建 push_type 作为 sink
  boost::coroutines2::coroutine<void>::push_type sink([this, &sink_called, &count](boost::coroutines2::coroutine<void>::pull_type& yield) {
    const int batch_size = 32;
    for(int i = 1 ; i <= 32 ; i ++ ){
      yield();  // 暂停
      sink_called = true;
      count ++;
    }
  });

  index_->BatchGet(sink, keys_array, retrieved_values.data(), tid);
  LOG(INFO) << "count: " << count;
  EXPECT_TRUE(sink_called) << "Sink should be called at least once";
  for (int i = 0; i < num_pairs; i++) {
    EXPECT_EQ(retrieved_values[i], values[i]) << "Failed for key " << keys[i];
  }
}

// 测试 BatchGet (无协程版)
TEST_F(IndexTest, BatchGetWithoutSink) {
  const int num_pairs = 50;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> values(num_pairs);
  unsigned tid = 0;

  // 插入数据
  for (int i = 0; i < num_pairs; i++) {
    keys[i] = i + 2000;
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

// 测试 BatchPut (有协程版)
TEST_F(IndexTest, BatchPutWithCoroutine) {
  const int num_pairs = 100;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> pointers(num_pairs);
  unsigned tid = 0;

  // 准备数据
  for (int i = 0; i < num_pairs; i++) {
    keys[i] = i + 6000;
    pointers[i] = i * 60;  // 模拟 pointers 值
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  bool sink_called = false;
  int count = 0;

  // 创建 push_type 作为 sink
  boost::coroutines2::coroutine<void>::push_type sink([this, &sink_called, &count](boost::coroutines2::coroutine<void>::pull_type& yield) {
    const int batch_size = 32;
    for(int i = 1 ; i <= 10 ; i ++ ){  // 计算预期批次次数
      yield();  // 暂停
      sink_called = true;
      count ++;
    }
  });

  index_->BatchPut(sink, keys_array, pointers.data(), tid);

  EXPECT_TRUE(sink_called) << "Sink should be called at least once";
  LOG(INFO) << "count: " << count;
  // 验证插入数据
  for (int i = 0; i < num_pairs; i++) {
    uint64_t retrieved;
    index_->Get(keys[i], retrieved, tid);
    EXPECT_EQ(retrieved, pointers[i]) << "Failed for key " << keys[i];
  }
}

// 测试 BulkLoad
TEST_F(IndexTest, BulkLoadTest) {
  const int num_keys = 50;
  std::vector<uint64_t> keys(num_keys);
  std::vector<Value_t> values(num_keys); 
  unsigned tid = 0;

  // 准备连续内存值
  for (int i = 0; i < num_keys; i++) {
    keys[i] = i + 3000;
    values[i] = i;  // 示例值
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_keys);
  index_->BulkLoad(keys_array, values.data());

  // 验证
  for (int i = 0; i < num_keys; i++) {
    uint64_t retrieved;
    index_->Get(keys[i], retrieved, tid);
    EXPECT_EQ(retrieved, values[i]) << "Failed for key " << keys[i];
  }
}

// 测试 LoadFakeData
TEST_F(IndexTest, LoadFakeDataTest) {
  const int64_t key_capacity = 50;
  const int value_size = sizeof(uint64_t);  // 假设 value_size 为指针大小
  unsigned tid = 0;

  index_->LoadFakeData(key_capacity, value_size);

  // 验证部分数据（0 到 key_capacity-1）
  for (int64_t i = 0; i < key_capacity; i++) {
    uint64_t retrieved;
    index_->Get(i, retrieved, tid);
    EXPECT_EQ(retrieved, i) << "Failed to load fake data for key " << i;
  }
}