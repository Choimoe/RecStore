#include "storage/ssd/CCEH.h"
#include "storage/ssd/file.h"
#include "storage/ssd/hash.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

class CCEHTest : public ::testing::Test {
protected:
  void SetUp() override { std::filesystem::remove(test_file_path); }

  void TearDown() override { std::filesystem::remove(test_file_path); }

  const std::string test_file_path = "cceh_test.db";
};

TEST_F(CCEHTest, SimpleInsertAndGet) {
  CCEH cceh(test_file_path);

  Key_t key = 100;
  Value_t value = 200;
  EXPECT_TRUE(cceh.Insert(key, value));

  Value_t ret_val = cceh.Get(key);
  EXPECT_EQ(ret_val, value);

  Key_t not_exist_key = 101;
  ret_val = cceh.Get(not_exist_key);
  EXPECT_EQ(ret_val, NONE);
}

TEST_F(CCEHTest, PersistenceTest) {
  const int num_to_insert = 10000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    keys.push_back(key);
  }

  {
    CCEH cceh(test_file_path);
    for (auto key : keys) {
      EXPECT_TRUE(cceh.Insert(key, key * 2));
    }
  }

  {
    CCEH cceh(test_file_path);
    for (const auto &key : keys) {
      Value_t ret_val = cceh.Get(key);
      EXPECT_EQ(ret_val, key * 2);
    }
  }
}

TEST_F(CCEHTest, SplitTest) {
  CCEH cceh(test_file_path);

  const int num_to_insert = 10000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    keys.push_back(key);
    EXPECT_TRUE(cceh.Insert(key, key * 2));
  }

  for (const auto &key : keys) {
    Value_t ret_val = cceh.Get(key);
    EXPECT_EQ(ret_val, key * 2);
  }
}

TEST_F(CCEHTest, DirectoryExpansionTest) {
  CCEH cceh(test_file_path);

  const int num_to_insert = 100000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i * 3;
    keys.push_back(key);
    EXPECT_TRUE(cceh.Insert(key, key * 2));
  }

  for (const auto &key : keys) {
    Value_t ret_val = cceh.Get(key);
    if (ret_val != key * 2) {
      EXPECT_EQ(ret_val, key * 2) << "Failed for key: " << key;
    }
  }
}

TEST_F(CCEHTest, UtilizationTest) {
  CCEH cceh(test_file_path);

  const int num_to_insert = 100000;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    EXPECT_TRUE(cceh.Insert(key, key * 2));
  }

  double util = cceh.Utilization();
  size_t cap = cceh.Capacity();

  EXPECT_GT(util, 0.0);
  EXPECT_LT(util, 100.0);
  EXPECT_GT(cap, 0);
  EXPECT_GE(cap, num_to_insert);
}

TEST_F(CCEHTest, ConcurrentInsertTest) {
  CCEH cceh(test_file_path);

  const int kNumThreads = 16;
  const int kInsertsPerThread = 1000;
  std::vector<std::thread> threads;

  auto inserter_func = [&](int thread_id) {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      Key_t key = thread_id * kInsertsPerThread + i;
      EXPECT_TRUE(cceh.Insert(key, key * 2));
    }
  };

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(inserter_func, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  // Verification
  for (int i = 0; i < kNumThreads * kInsertsPerThread; ++i) {
    Key_t key = i;
    Value_t ret_val = cceh.Get(key);
    EXPECT_EQ(ret_val, key * 2);
  }
}