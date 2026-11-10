#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <limits>

#include "framework/gpu/host_memory.h"

TEST(HostMemoryTest, RejectsEmptyAndOverflowingRanges) {
  using recstore::framework::EnsurePinnedLocalShmPayload;
  EXPECT_FALSE(EnsurePinnedLocalShmPayload(nullptr, 4096));
  EXPECT_FALSE(EnsurePinnedLocalShmPayload(reinterpret_cast<void*>(1), 0));
  const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
  EXPECT_FALSE(EnsurePinnedLocalShmPayload(
      reinterpret_cast<void*>(max_address - 7), 16));
  EXPECT_FALSE(
      EnsurePinnedLocalShmPayload(reinterpret_cast<void*>(max_address - 7), 1));
}

TEST(HostMemoryTest, ClearsHandledRegistrationErrorOnAnActualCudaDevice) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    (void)cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const size_t bytes = static_cast<size_t>(page_size);
  void* memory       = mmap(
      nullptr,
      bytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  ASSERT_NE(memory, MAP_FAILED);
  struct Mapping {
    void* ptr;
    size_t bytes;
    bool registered = false;
    ~Mapping() {
      if (registered)
        (void)cudaHostUnregister(ptr);
      munmap(ptr, bytes);
    }
  } mapping{memory, bytes};
  ASSERT_EQ(cudaHostRegister(memory, bytes, cudaHostRegisterPortable),
            cudaSuccess);
  mapping.registered = true;
  (void)cudaGetLastError();
  EXPECT_TRUE(recstore::framework::EnsurePinnedLocalShmPayload(memory, bytes));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_TRUE(recstore::framework::EnsurePinnedLocalShmPayload(memory, bytes));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
