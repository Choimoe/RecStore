#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace recstore::framework::gpu {

struct CacheContext {
  int device;
  cudaStream_t stream;
};

// All buffers are contiguous device allocations on context.device. Callers
// own them and keep them alive through completion on context.stream. Mutations
// and reads are ordered across streams; clear/disable wait before freeing.
struct LookupBuffers {
  float* values;
  uint64_t* missing_positions;
  int64_t* missing_keys;
  size_t* missing_count;
};

bool EnableGpuCache(int64_t capacity, int64_t dim, int device);
void DisableGpuCache();
// Explicit reset also resets bypass policy; internal eviction preserves it.
void ResetGpuCache();
void ClearGpuCache();
bool IsGpuCacheEnabled();
uint64_t GetGpuCacheGeneration();
bool CanUseGpuCache(int device, int64_t dim);

// Query waits for the miss count; hit-only reads and mutations enqueue work.
size_t QueryGpuCache(const int64_t* keys,
                     size_t count,
                     int64_t dim,
                     LookupBuffers output,
                     CacheContext context);
void LookupGpuCacheAssumingHits(
    const int64_t* keys,
    size_t count,
    int64_t dim,
    float* values,
    bool* miss,
    CacheContext context);
void ContainsGpuCache(
    const int64_t* keys, size_t count, bool* resident, CacheContext context);
void FillGpuCache(const int64_t* keys,
                  size_t count,
                  const float* values,
                  int64_t dim,
                  CacheContext context);
void FillGpuCacheNoEvict(
    const int64_t* keys,
    size_t count,
    const float* values,
    int64_t dim,
    bool* inserted,
    CacheContext context);
void UpdateGpuCache(const int64_t* keys,
                    size_t count,
                    const float* values,
                    int64_t dim,
                    CacheContext context);
void InvalidateGpuCache(
    const int64_t* keys, size_t count, CacheContext context);
void InvalidateGpuCacheWithMask(
    const int64_t* keys, size_t count, bool* removed, CacheContext context);

// Sum duplicate gradients before updating. If any key is absent, invalidate
// the requested keys and return false without applying any gradients.
bool ApplySgdUpdateGpuCache(
    const int64_t* keys,
    size_t count,
    const float* grads,
    int64_t dim,
    double lr,
    CacheContext context);
// Updates resident rows only, skipping missing keys, without a host wait.
void ApplySgdUpdateBestEffortGpuCache(
    const int64_t* keys,
    size_t count,
    const float* grads,
    int64_t dim,
    double lr,
    CacheContext context);

} // namespace recstore::framework::gpu
