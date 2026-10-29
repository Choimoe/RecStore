#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/allocators/persist_loop_slab_allocator.h"
#include "memory/shm_file.h"

namespace {

class PersistLoopSlabBitmapTest : public ::testing::Test {
protected:
  void SetUp() override {
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;
  }

  std::unique_ptr<base::PersistLoopShmMalloc>
  MakeAllocator(const std::string& name) {
    const std::string path =
        "/tmp/recstore-persist-bitmap-" + std::to_string(getpid()) + "-" + name;
    return std::make_unique<base::PersistLoopShmMalloc>(
        path, 64 * 1024, "DRAM");
  }
};

TEST_F(PersistLoopSlabBitmapTest, PreservesNeighborsAcrossEveryWordOffset) {
  for (int blocks : {63, 64, 65}) {
    for (int offset = 0; offset < 64; ++offset) {
      SCOPED_TRACE("blocks=" + std::to_string(blocks) +
                   " offset=" + std::to_string(offset));
      auto allocator =
          MakeAllocator(std::to_string(blocks) + "-" + std::to_string(offset));
      // Each allocation has one header block before its payload. Padding
      // places the target payload at every possible bit offset in a word.
      int padding_blocks = (offset + 62) % 64;
      if (padding_blocks == 0)
        padding_blocks = 64;
      const int padding_bytes = padding_blocks * 8;
      const int value_bytes   = blocks * 8;
      char* before            = allocator->New(padding_bytes);
      char* target            = allocator->New(value_bytes);
      char* after             = allocator->New(64);
      ASSERT_NE(before, nullptr);
      ASSERT_NE(target, nullptr);
      ASSERT_NE(after, nullptr);
      ASSERT_EQ((allocator->GetMallocOffset(target) / 8) % 64, offset);
      std::memset(before, 0x11, padding_bytes);
      std::memset(target, 0x22, value_bytes);
      std::memset(after, 0x33, 64);

      ASSERT_TRUE(allocator->Free(target));
      EXPECT_FALSE(allocator->Free(target));
      EXPECT_EQ(allocator->total_malloc(), 2);
      std::vector<char*> live;
      allocator->GetMallocsAppend(&live);
      ASSERT_EQ(live.size(), 2);
      EXPECT_NE(std::find(live.begin(), live.end(), before), live.end());
      EXPECT_NE(std::find(live.begin(), live.end(), after), live.end());

      char* reused = allocator->New(value_bytes);
      ASSERT_EQ(reused, target);
      EXPECT_EQ(allocator->GetMallocSize(reused), value_bytes);
      std::memset(reused, 0x44, value_bytes);
      EXPECT_TRUE(std::all_of(before, before + padding_bytes, [](char byte) {
        return byte == 0x11;
      }));
      EXPECT_TRUE(std::all_of(after, after + 64, [](char byte) {
        return byte == 0x33;
      }));
      EXPECT_EQ(allocator->total_malloc(), 3);
      ASSERT_TRUE(allocator->Free(before));
      ASSERT_TRUE(allocator->Free(reused));
      ASSERT_TRUE(allocator->Free(after));
      EXPECT_EQ(allocator->total_malloc(), 0);
      live.clear();
      allocator->GetMallocsAppend(&live);
      EXPECT_TRUE(live.empty());
    }
  }
}

TEST_F(PersistLoopSlabBitmapTest, ZeroSizeSentinelPreservesLiveAllocation) {
  auto allocator = MakeAllocator("sentinel");
  char* value    = allocator->New(512);
  ASSERT_NE(value, nullptr);
  std::memset(value, 0x55, 512);
  char* sentinel = allocator->New(0);
  EXPECT_EQ(sentinel, allocator->BackingData() + allocator->BackingSize());
  EXPECT_EQ(allocator->GetMallocSize(sentinel), 0);
  EXPECT_TRUE(allocator->Free(sentinel));
  EXPECT_TRUE(allocator->Free(sentinel));
  EXPECT_EQ(allocator->total_malloc(), 1);
  EXPECT_TRUE(
      std::all_of(value, value + 512, [](char byte) { return byte == 0x55; }));
  EXPECT_TRUE(allocator->Free(value));
  EXPECT_FALSE(allocator->Free(value));
  EXPECT_EQ(allocator->total_malloc(), 0);
}

} // namespace
