#include "framework/cache/gpu_embedding_cache.h"
#include "framework/cache/cache_aware_op.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <stdexcept>

namespace {
namespace cache = recstore::framework::gpu;
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T>
struct Buffer {
  T* data = nullptr;
  explicit Buffer(size_t count) {
    Check(cudaMallocManaged(&data, count * sizeof(T)));
  }
  ~Buffer() { cudaFree(data); }
  T& operator[](size_t i) { return data[i]; }
};
class GpuCacheCore : public ::testing::Test {
protected:
  cudaStream_t first = nullptr, second = nullptr;
  cache::CacheContext context{};
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
      GTEST_SKIP() << "CUDA device required";
    Check(cudaSetDevice(0));
    Check(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking));
    Check(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking));
    context = {0, first};
    cache::EnableGpuCache(128, 4, 0);
  }
  void TearDown() override {
    cache::DisableGpuCache();
    if (first)
      cudaStreamDestroy(first);
    if (second)
      cudaStreamDestroy(second);
  }
};

TEST_F(GpuCacheCore, QueryAndNoEvictMasksUseCallerBuffers) {
  Buffer<int64_t> keys(3), missing(3);
  Buffer<float> values(12), output(12);
  Buffer<bool> mask(3);
  Buffer<uint64_t> positions(3);
  Buffer<size_t> count(1);
  keys[0] = 11;
  keys[1] = 13;
  keys[2] = 999;
  for (int i = 0; i < 12; ++i)
    values[i] = float(i);
  cache::FillGpuCacheNoEvict(keys.data, 2, values.data, 4, mask.data, context);
  Check(cudaStreamSynchronize(first));
  EXPECT_TRUE(mask[0] && mask[1]);
  // Existing rows must survive a stale no-evict prefill.
  values[0] = -100;
  cache::FillGpuCacheNoEvict(keys.data, 2, values.data, 4, mask.data, context);
  auto n = cache::QueryGpuCache(
      keys.data,
      3,
      4,
      {output.data, positions.data, missing.data, count.data},
      context);
  ASSERT_EQ(n, 1);
  EXPECT_EQ(positions[0], 2);
  EXPECT_EQ(missing[0], 999);
  EXPECT_FLOAT_EQ(output[0], 0);
  cache::InvalidateGpuCacheWithMask(keys.data, 3, mask.data, context);
  Check(cudaStreamSynchronize(first));
  EXPECT_TRUE(mask[0] && mask[1]);
  EXPECT_FALSE(mask[2]);
  cache::ContainsGpuCache(keys.data, 3, mask.data, context);
  Check(cudaStreamSynchronize(first));
  EXPECT_FALSE(mask[0] || mask[1] || mask[2]);
}

TEST_F(GpuCacheCore, StrictSgdSumsDuplicatesAndRejectsPartialHits) {
  Buffer<int64_t> keys(3);
  Buffer<float> values(12), grads(12), output(12);
  Buffer<bool> miss(3);
  keys[0] = 5;
  keys[1] = 7;
  keys[2] = 5;
  std::fill_n(values.data, 12, 10.0f);
  for (int i = 0; i < 12; ++i)
    grads[i] = float(i + 1);
  cache::FillGpuCache(keys.data, 2, values.data, 4, context);
  ASSERT_TRUE(cache::ApplySgdUpdateGpuCache(
      keys.data, 3, grads.data, 4, 0.125, context));
  cache::LookupGpuCacheAssumingHits(
      keys.data, 3, 4, output.data, miss.data, context);
  Check(cudaStreamSynchronize(first));
  for (int j = 0; j < 4; ++j) {
    EXPECT_FLOAT_EQ(output[j], 10 - 0.125f * (grads[j] + grads[8 + j]));
    EXPECT_FLOAT_EQ(output[4 + j], 10 - 0.125f * grads[4 + j]);
    EXPECT_FLOAT_EQ(output[8 + j], output[j]);
  }
  EXPECT_FALSE(miss[0] || miss[1] || miss[2]);
  keys[2] = 999;
  ASSERT_FALSE(cache::ApplySgdUpdateGpuCache(
      keys.data, 3, grads.data, 4, 0.125, context));
  cache::ContainsGpuCache(keys.data, 3, miss.data, context);
  Check(cudaStreamSynchronize(first));
  EXPECT_FALSE(miss[0] || miss[1] || miss[2]);
}

TEST_F(GpuCacheCore, CrossStreamUpdatesFinishBeforeDisable) {
  Buffer<int64_t> keys(2);
  Buffer<float> values(8), grads(8), output(8);
  Buffer<bool> miss(2);
  keys[0] = 5;
  keys[1] = 999;
  std::fill_n(values.data, 8, 10.0f);
  std::fill_n(grads.data, 8, 2.0f);
  for (int iteration = 0; iteration < 4; ++iteration) {
    cache::EnableGpuCache(128, 4, 0);
    cache::FillGpuCache(keys.data, 1, values.data, 4, context);
    cache::ApplySgdUpdateBestEffortGpuCache(
        keys.data, 2, grads.data, 4, 0.5, {0, second});
    cache::LookupGpuCacheAssumingHits(
        keys.data, 2, 4, output.data, miss.data, context);
    cache::DisableGpuCache();
    EXPECT_FALSE(miss[0]);
    EXPECT_TRUE(miss[1]);
    for (int j = 0; j < 4; ++j)
      EXPECT_FLOAT_EQ(output[j], 9);
  }
}

TEST_F(GpuCacheCore, LifecycleOwnsPolicyResetAndValidatesDevice) {
  using recstore::framework::g_cache_aware_op;
  recstore::framework::CacheLookupActions io;
  io.usable       = true;
  io.query        = [] { return 1024; };
  io.read_missing = io.fill = io.scatter = io.read_all = io.clear = [] {};
  g_cache_aware_op.SetBypassEnabled(true);
  g_cache_aware_op.Lookup(1024, io);
  g_cache_aware_op.Lookup(1024, io);
  ASSERT_TRUE(g_cache_aware_op.Bypassed());
  cache::ClearGpuCache();
  EXPECT_TRUE(g_cache_aware_op.Bypassed());
  auto generation = cache::GetGpuCacheGeneration();
  cache::ResetGpuCache();
  EXPECT_FALSE(g_cache_aware_op.Bypassed());
  EXPECT_GT(cache::GetGpuCacheGeneration(), generation);
  EXPECT_TRUE(cache::CanUseGpuCache(0, 4));
  EXPECT_FALSE(cache::CanUseGpuCache(1, 4));
  EXPECT_THROW(cache::ContainsGpuCache(nullptr, 0, nullptr, {1, first}),
               std::runtime_error);
  EXPECT_THROW(cache::FillGpuCache(nullptr, 0, nullptr, 3, context),
               std::runtime_error);
  EXPECT_NO_THROW(cache::ContainsGpuCache(nullptr, 0, nullptr, context));
}
} // namespace
