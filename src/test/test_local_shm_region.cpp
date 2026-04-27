#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <sys/mman.h>

#include "ps/local_shm/local_shm_queue.h"
#include "ps/local_shm/local_shm_region.h"

namespace recstore {
namespace {

std::string UniqueRegionName() {
  static std::atomic<uint64_t> counter{0};
  return "recstore_local_shm_region_test_" + std::to_string(::getpid()) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         "_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

std::string NormalizeRegionNameForShm(const std::string& region_name) {
  if (!region_name.empty() && region_name.front() == '/') {
    return region_name;
  }
  return "/" + region_name;
}

TEST(LocalShmRegionTest, CreateAndAttachRegion) {
  const std::string region_name       = UniqueRegionName();
  constexpr uint32_t kSlotCount       = 4;
  constexpr uint32_t kSlotBytes       = 4096;
  constexpr uint32_t kReadyQueueCount = 3;

  ::shm_unlink(NormalizeRegionNameForShm(region_name).c_str());

  LocalShmRegion server_region;
  ASSERT_TRUE(server_region.Create(
      region_name, kSlotCount, kSlotBytes, kReadyQueueCount));
  ASSERT_TRUE(server_region.IsOpen());
  EXPECT_EQ(server_region.control()->magic, kLocalShmMagic);
  EXPECT_EQ(server_region.control()->slot_count, kSlotCount);
  EXPECT_EQ(server_region.control()->slot_buffer_bytes, kSlotBytes);
  EXPECT_EQ(server_region.control()->ready_queue_count, kReadyQueueCount);

  LocalShmRegion client_region;
  ASSERT_TRUE(client_region.Attach(
      region_name, kSlotCount, kSlotBytes, kReadyQueueCount));
  ASSERT_TRUE(client_region.IsOpen());
  EXPECT_EQ(client_region.control()->magic, kLocalShmMagic);
  EXPECT_NE(client_region.slot_header(0), nullptr);
  EXPECT_NE(client_region.slot_payload(0), nullptr);

  std::vector<uint32_t> seen;
  seen.reserve(kSlotCount);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    uint32_t slot_id = 0;
    ASSERT_TRUE(LocalShmQueueDequeue(
        client_region.queue_header(LocalQueueKind::kFree),
        client_region.queue_cells(LocalQueueKind::kFree),
        &slot_id));
    seen.push_back(slot_id);
  }
  EXPECT_FALSE(LocalShmQueueDequeue(
      client_region.queue_header(LocalQueueKind::kFree),
      client_region.queue_cells(LocalQueueKind::kFree),
      &seen[0]));
  EXPECT_EQ(seen.size(), static_cast<size_t>(kSlotCount));

  for (uint32_t ready_queue_id = 0; ready_queue_id < kReadyQueueCount;
       ++ready_queue_id) {
    uint32_t slot_id = 0;
    EXPECT_FALSE(LocalShmQueueDequeue(
        client_region.ready_queue_header(ready_queue_id),
        client_region.ready_queue_cells(ready_queue_id),
        &slot_id));
  }

  client_region.Close();
  server_region.Close();
  ::shm_unlink(NormalizeRegionNameForShm(region_name).c_str());
}

} // namespace
} // namespace recstore
