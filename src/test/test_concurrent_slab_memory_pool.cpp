#include <algorithm>

#include "memory/allocators/concurrent_slab_memory_pool.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace {

constexpr int kNumThreads               = 32;
constexpr int kAllocsPerThread          = 200;
constexpr int kMultiSlabAllocsPerThread = 40;
constexpr int64_t kPoolBytes            = 256 * 1024 * 1024;

std::string TempPoolPath(const std::string& name) {
  return "/tmp/recstore_concurrent_slab_pool_" + name;
}

std::vector<int> FactoryPayloadSizes() { return {32, 64, 128, 512, 1024}; }

bool OwnsAllocation(
    const base::MallocApi& pool, const char* ptr, size_t bytes) {
  const auto begin = reinterpret_cast<uintptr_t>(pool.BackingData());
  const auto addr  = reinterpret_cast<uintptr_t>(ptr);
  const auto size  = static_cast<uintptr_t>(pool.BackingSize());
  return ptr != nullptr && size >= bytes && addr >= begin &&
         addr - begin <= size - bytes;
}

} // namespace

TEST(ConcurrentSlabMemoryPoolOwnershipTest, AlternatingPoolsRetainCapacity) {
  constexpr int64_t kChunkBytes = 1024 * 1024;
  base::ConcurrentSlabMemoryPoolMalloc a(
      TempPoolPath("owner_a"), kChunkBytes, "DRAM");
  base::ConcurrentSlabMemoryPoolMalloc b(
      TempPoolPath("owner_b"), kChunkBytes, "DRAM");
  for (int round = 0; round < 2; ++round) {
    std::vector<char*> a_ptrs, b_ptrs;
    // Each pool has only one chunk, so every TLS switch must rediscover it.
    for (int i = 0; i < 1900; ++i) {
      char* pa = a.New(512);
      char* pb = b.New(512);
      ASSERT_TRUE(OwnsAllocation(a, pa, 512));
      ASSERT_TRUE(OwnsAllocation(b, pb, 512));
      a_ptrs.push_back(pa);
      b_ptrs.push_back(pb);
    }
    EXPECT_EQ(std::set<char*>(a_ptrs.begin(), a_ptrs.end()).size(),
              a_ptrs.size());
    EXPECT_EQ(std::set<char*>(b_ptrs.begin(), b_ptrs.end()).size(),
              b_ptrs.size());
    for (size_t i = 0; i < a_ptrs.size(); ++i) {
      EXPECT_TRUE(a.Free(a_ptrs[i]));
      EXPECT_TRUE(b.Free(b_ptrs[i]));
    }
    EXPECT_EQ(a.total_malloc(), 0u);
    EXPECT_EQ(b.total_malloc(), 0u);
  }
}

TEST(ConcurrentSlabMemoryPoolOwnershipTest,
     AlternatingPoolsKeepChunksForOtherClasses) {
  constexpr int64_t kChunkBytes = 1024 * 1024;
  base::ConcurrentSlabMemoryPoolMalloc a(
      TempPoolPath("owner_classes_a"), 2 * kChunkBytes, "DRAM");
  base::ConcurrentSlabMemoryPoolMalloc b(
      TempPoolPath("owner_classes_b"), 2 * kChunkBytes, "DRAM");
  for (int i = 0; i < 2; ++i) {
    for (auto* pool : {&a, &b}) {
      char* ptr = pool->New(512);
      ASSERT_TRUE(OwnsAllocation(*pool, ptr, 512));
      EXPECT_LT(pool->GetMallocOffset(ptr), kChunkBytes);
    }
  }
  for (auto* pool : {&a, &b}) {
    char* ptr = pool->New(64);
    ASSERT_TRUE(OwnsAllocation(*pool, ptr, 64));
    EXPECT_GE(pool->GetMallocOffset(ptr), kChunkBytes);
    EXPECT_EQ(pool->total_malloc(), 3u);
  }
}

TEST(ConcurrentSlabMemoryPoolOwnershipTest, FreshWorkersUseSeparateChunks) {
  constexpr int64_t kChunkBytes = 1024 * 1024;
  base::ConcurrentSlabMemoryPoolMalloc pool(
      TempPoolPath("owner_fresh_workers"), 2 * kChunkBytes, "DRAM");
  char* ptrs[2] = {};
  std::vector<std::thread> workers;
  for (int i = 0; i < 2; ++i) {
    workers.emplace_back([&, i] { ptrs[i] = pool.New(512); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  ASSERT_TRUE(OwnsAllocation(pool, ptrs[0], 512));
  ASSERT_TRUE(OwnsAllocation(pool, ptrs[1], 512));
  EXPECT_NE(pool.GetMallocOffset(ptrs[0]) / kChunkBytes,
            pool.GetMallocOffset(ptrs[1]) / kChunkBytes);
  EXPECT_TRUE(pool.Free(ptrs[0]));
  EXPECT_TRUE(pool.Free(ptrs[1]));
  EXPECT_EQ(pool.total_malloc(), 0u);
}

TEST(ConcurrentSlabMemoryPoolOwnershipTest, InitializeInvalidatesWorkerCache) {
  constexpr int64_t kChunkBytes = 1024 * 1024;
  base::ConcurrentSlabMemoryPoolMalloc pool(
      TempPoolPath("owner_reset"), 2 * kChunkBytes, "DRAM");
  std::promise<void> cached, resume;
  auto resume_future = resume.get_future();
  char* result       = nullptr;
  std::thread worker([&] {
    pool.New(512);
    cached.set_value();
    resume_future.wait();
    result = pool.New(512);
  });
  cached.get_future().wait();
  pool.Initialize();
  // Rebind the worker's former chunk to another class.
  char* small = pool.New(64);
  resume.set_value();
  worker.join();

  ASSERT_TRUE(OwnsAllocation(pool, result, 512));
  EXPECT_GE(pool.GetMallocOffset(result), kChunkBytes);
  EXPECT_EQ(pool.total_malloc(), 2u);
  EXPECT_TRUE(pool.Free(small));
  EXPECT_TRUE(pool.Free(result));
}

TEST(ConcurrentSlabMemoryPoolOwnershipTest,
     SameAddressRebuildInvalidatesWorkerCache) {
  using Pool                    = base::ConcurrentSlabMemoryPoolMalloc;
  constexpr int64_t kChunkBytes = 1024 * 1024;
  std::optional<Pool> pool(
      std::in_place, TempPoolPath("owner_old"), 2 * kChunkBytes, "DRAM");
  Pool* original = &*pool;
  std::promise<void> cached, resume;
  auto resume_future = resume.get_future();
  char* result       = nullptr;
  std::thread worker([&] {
    pool->New(512);
    cached.set_value();
    resume_future.wait();
    result = pool->New(512);
  });
  cached.get_future().wait();
  pool.reset();
  pool.emplace(TempPoolPath("owner_new"), 2 * kChunkBytes, "DRAM");
  EXPECT_EQ(&*pool, original);
  char* small = pool->New(64);
  resume.set_value();
  worker.join();

  ASSERT_TRUE(OwnsAllocation(*pool, result, 512));
  EXPECT_GE(pool->GetMallocOffset(result), kChunkBytes);
  EXPECT_EQ(pool->total_malloc(), 2u);
  EXPECT_TRUE(pool->Free(small));
  EXPECT_TRUE(pool->Free(result));
}

class ConcurrentSlabMemoryPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    path_ = TempPoolPath("gtest");
    base::file_util::Delete(path_, false);
    pool_ = std::make_unique<base::ConcurrentSlabMemoryPoolMalloc>(
        path_, kPoolBytes, "DRAM");
  }

  void TearDown() override {
    pool_.reset();
    base::file_util::Delete(path_, false);
  }

  std::string path_;
  std::unique_ptr<base::ConcurrentSlabMemoryPoolMalloc> pool_;
};

TEST_F(ConcurrentSlabMemoryPoolTest, FactoryCreatesInstance) {
  auto* from_factory = base::
      Factory<base::MallocApi, const std::string&, int64, const std::string&>::
          NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                      path_ + "_factory",
                      8 * 1024 * 1024,
                      "DRAM");
  ASSERT_NE(from_factory, nullptr);
  char* p = from_factory->New(32);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(from_factory->GetMallocSize(p), 32);
  EXPECT_TRUE(from_factory->Free(p));
  delete from_factory;
  base::file_util::Delete(path_ + "_factory", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SingleSlabMultithreadedNewFree) {
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &failures]() {
      std::vector<char*> ptrs;
      ptrs.reserve(kAllocsPerThread);
      for (int i = 0; i < kAllocsPerThread; ++i) {
        char* p = pool_->New(64);
        if (p == nullptr) {
          failures.fetch_add(1);
          return;
        }
        if (pool_->GetMallocSize(p) != 64) {
          failures.fetch_add(1);
          return;
        }
        ptrs.push_back(p);
      }
      for (char* p : ptrs) {
        if (!pool_->Free(p)) {
          failures.fetch_add(1);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, MultiSlabPerThreadHotChunks) {
  const auto payload_sizes = FactoryPayloadSizes();
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &payload_sizes, &failures]() {
      std::vector<char*> ptrs;
      ptrs.reserve(kMultiSlabAllocsPerThread * payload_sizes.size());
      for (int i = 0; i < kMultiSlabAllocsPerThread; ++i) {
        for (int payload : payload_sizes) {
          char* p = pool_->New(payload);
          if (p == nullptr) {
            failures.fetch_add(1);
            return;
          }
          if (pool_->GetMallocSize(p) != payload) {
            failures.fetch_add(1);
            return;
          }
          ptrs.push_back(p);
        }
      }
      for (char* p : ptrs) {
        if (!pool_->Free(p)) {
          failures.fetch_add(1);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, AllocatesBeyondSingleHotChunk) {
  std::vector<char*> ptrs;
  ptrs.reserve(9000);

  for (int i = 0; i < 9000; ++i) {
    char* p = pool_->New(128);
    ASSERT_NE(p, nullptr) << "allocation " << i;
    ASSERT_EQ(pool_->GetMallocSize(p), 128);
    ptrs.push_back(p);
  }

  for (char* p : ptrs) {
    ASSERT_TRUE(pool_->Free(p));
  }
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, ConcurrentAllocatesBeyondSingleChunk) {
  constexpr int kThreads = 32;
  constexpr int kAllocs  = 400;

  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::vector<char*>> ptrs(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([this, t, &start, &failures, &ptrs]() {
      ptrs[t].reserve(kAllocs);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kAllocs; ++i) {
        char* p = pool_->New(128);
        if (p == nullptr || pool_->GetMallocSize(p) != 128) {
          failures.fetch_add(1);
          return;
        }
        ptrs[t].push_back(p);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(failures.load(), 0);
  for (auto& local : ptrs) {
    for (char* p : local) {
      ASSERT_TRUE(pool_->Free(p));
    }
  }
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, AllocatedPointersDoNotOverlap) {
  std::set<char*> unique_ptrs;
  std::mutex ptr_mu;
  std::vector<std::vector<char*>> ptrs(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &ptrs, &unique_ptrs, &ptr_mu]() {
      auto& local = ptrs[t];
      for (int i = 0; i < 50; ++i) {
        local.push_back(pool_->New(128));
      }
      {
        std::lock_guard<std::mutex> lock(ptr_mu);
        for (char* p : local) {
          EXPECT_TRUE(unique_ptrs.insert(p).second) << "duplicate allocation";
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  // Keep all allocations alive until every thread has checked uniqueness.
  for (const auto& local : ptrs) {
    for (char* p : local) {
      EXPECT_TRUE(pool_->Free(p));
    }
  }
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, DefaultClassesIncludeEmbeddingWidths) {
  auto* pool = base::
      Factory<base::MallocApi, const std::string&, int64, const std::string&>::
          NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                      path_ + "_widths",
                      8 * 1024 * 1024,
                      "DRAM");
  ASSERT_NE(pool, nullptr);

  // 8 + dim * 4 classes keep dim 20 / dim 68 rows on an exact size class; both
  // would otherwise round up to 136 / 520 and waste most of the slot.
  char* row80 = pool->New(80);
  ASSERT_NE(row80, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row80), 80);

  char* row272 = pool->New(272);
  ASSERT_NE(row272, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row272), 272);

  EXPECT_TRUE(pool->Free(row80));
  EXPECT_TRUE(pool->Free(row272));
  delete pool;
  base::file_util::Delete(path_ + "_widths", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SizedFactoryUsesConfiguredClasses) {
  auto* pool = base::Factory<
      base::MallocApi,
      const std::string&,
      int64,
      const std::string&,
      const std::vector<int>&>::NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                                            path_ + "_sized",
                                            8 * 1024 * 1024,
                                            "DRAM",
                                            std::vector<int>{8 + 96});
  ASSERT_NE(pool, nullptr);

  char* row96 = pool->New(96);
  ASSERT_NE(row96, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row96), 96);
  EXPECT_TRUE(pool->Free(row96));
  delete pool;
  base::file_util::Delete(path_ + "_sized", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SizedFactorySupportsRowsBeyondDefaultCap) {
  // The fixed default table stops at 8+1024, so rows above 1024 bytes have no
  // class and hit a CHECK failure at the first insert. An explicit list is the
  // supported way to serve wider rows.
  const std::vector<int> classes{8 + 32, 8 + 2048};
  auto* pool = base::Factory<
      base::MallocApi,
      const std::string&,
      int64,
      const std::string&,
      const std::vector<int>&>::NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                                            path_ + "_wide",
                                            64 * 1024 * 1024,
                                            "DRAM",
                                            classes);
  ASSERT_NE(pool, nullptr);

  char* wide = pool->New(1032);
  ASSERT_NE(wide, nullptr);
  EXPECT_EQ(pool->GetMallocSize(wide), 1032);
  EXPECT_TRUE(pool->Free(wide));
  delete pool;
  base::file_util::Delete(path_ + "_wide", false);
}
