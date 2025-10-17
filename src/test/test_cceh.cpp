#include "storage/ssd/CCEH.h"
#include "gtest/gtest.h"
#include <thread>
#include <vector>

class CCEHTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CCEHTest, SimpleInsertAndGet) {
  CCEH cceh;

  Key_t key     = 100;
  Value_t value = 200;
  cceh.Insert(key, value);

  Value_t ret_val = cceh.Get(key);
  EXPECT_EQ(ret_val, value);

  Key_t not_exist_key = 101;
  ret_val             = cceh.Get(not_exist_key);
  EXPECT_EQ(ret_val, NONE);
}

TEST_F(CCEHTest, SplitTest) {
  CCEH cceh;

  const int num_to_insert = 10000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    keys.push_back(key);
    cceh.Insert(key, key * 2);
  }

  for (const auto& key : keys) {
    Value_t ret_val = cceh.Get(key);
    EXPECT_EQ(ret_val, key * 2);
  }
}

TEST_F(CCEHTest, DirectoryExpansionTest) {
  CCEH cceh;

  const int num_to_insert = 100000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i * 3;
    keys.push_back(key);
    cceh.Insert(key, key * 2);
  }

  for (const auto& key : keys) {
    Value_t ret_val = cceh.Get(key);
    if (ret_val != key * 2) {
      EXPECT_EQ(ret_val, key * 2) << "Failed for key: " << key;
    }
  }
}

TEST_F(CCEHTest, ConcurrentInsertTest) {
  CCEH cceh;

  const int kNumThreads       = 64;
  const int kInsertsPerThread = 1000;
  std::vector<std::thread> threads;

  auto inserter_func = [&](int thread_id) {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      Key_t key = thread_id * kInsertsPerThread + i;
      cceh.Insert(key, key * 2);
    }
  };

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(inserter_func, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  // Verification
  for (int i = 0; i < kNumThreads * kInsertsPerThread; ++i) {
    Key_t key       = i;
    Value_t ret_val = cceh.Get(key);
    EXPECT_EQ(ret_val, key * 2);
  }
}