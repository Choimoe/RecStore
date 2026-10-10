#pragma once

// Internal shared state and helpers for the recstore_ops translation units.
// This header owns the process-wide profile counters and the GPU cache
// adaptive-bypass / pending-update state so the per-family .cc files do not
// each carry their own copy.

#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/tensor.h"
#include "framework/op.h"
#include "framework/pytorch/op_cuda_runtime.h"
#include "ps/local_shm/local_shm_client.h"

#include <glog/logging.h>

#ifdef RECSTORE_ENABLE_GPU_CACHE
#  include "framework/gpu/gpu_embedding_cache.h"
#endif

namespace recstore {
namespace framework {

// --- Shared predicates and tensor helpers ----------------------------------

inline bool IsLocalFastPathBackend(const std::string& backend) {
  return backend == "local_shm" || backend == "hierkv";
}

inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

inline torch::TensorOptions PinnedCpuOptions(torch::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCPU)
      .dtype(dtype)
      .pinned_memory(true);
}

inline torch::Tensor StageCudaTensorToPinnedCpu(const torch::Tensor& tensor,
                                                torch::ScalarType dtype) {
  auto cpu_tensor = torch::empty(tensor.sizes(), PinnedCpuOptions(dtype));
  cpu_tensor.copy_(tensor.to(dtype), /*non_blocking=*/false);
  return cpu_tensor;
}

inline torch::Tensor
StageCudaTensorToPinnedCpuAsyncNoCast(const torch::Tensor& tensor) {
  auto cpu_tensor =
      torch::empty(tensor.sizes(), PinnedCpuOptions(tensor.scalar_type()));
  cpu_tensor.copy_(tensor, /*non_blocking=*/true);
  return cpu_tensor;
}

inline std::shared_ptr<KVClientOp> GetConcreteKVClientOp() {
  auto op    = GetKVClientOp();
  auto kv_op = std::dynamic_pointer_cast<KVClientOp>(op);
  TORCH_CHECK(kv_op != nullptr, "storage backend is not KVClientOp");
  return kv_op;
}

// --- Local fast-path profile state -----------------------------------------

enum LookupProfileIndex : std::size_t {
  kLookupTotalMs = 0,
  kLookupKeysStageMs,
  kLookupSubmitMs,
  kLookupWaitMs,
  kLookupPayloadPinMs,
  kLookupFallbackCopyMs,
  kLookupValuesH2DEnqueueMs,
  kLookupProfileSize,
};

enum UpdateProfileIndex : std::size_t {
  kUpdateTotalMs = 0,
  kUpdateKeysStageMs,
  kUpdateGradsStageMs,
  kUpdateShmCallMs,
  kUpdateStageWaitMs,
  kUpdateProfileSize,
};

inline thread_local std::vector<double>
    g_last_local_lookup_flat_profile(kLookupProfileSize, 0.0);
inline thread_local std::vector<double>
    g_last_local_update_flat_profile(kUpdateProfileSize, 0.0);

inline std::chrono::steady_clock::time_point SteadyNow() {
  return std::chrono::steady_clock::now();
}

inline double ElapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
             SteadyNow() - start)
      .count();
}

inline void ResetLocalLookupFlatProfile() {
  std::fill(g_last_local_lookup_flat_profile.begin(),
            g_last_local_lookup_flat_profile.end(),
            0.0);
}

inline void ResetLocalUpdateFlatProfile() {
  std::fill(g_last_local_update_flat_profile.begin(),
            g_last_local_update_flat_profile.end(),
            0.0);
}

#ifdef RECSTORE_ENABLE_GPU_CACHE

// --- GPU cache adaptive-bypass and pending-update state --------------------

struct PendingGpuCacheUpdate {
  torch::Tensor keys;
  torch::Tensor grads;
};

inline std::mutex g_pending_gpu_cache_updates_mu;
inline std::unordered_map<uint64_t, PendingGpuCacheUpdate>
    g_pending_gpu_cache_updates;

constexpr int64_t kGpuCacheBypassMinRows                   = 1024;
constexpr int kGpuCacheLowHitLimit                         = 1;
constexpr double kGpuCacheLowHitRatio                      = 0.05;
inline thread_local int g_gpu_cache_low_hit_streak         = 0;
inline thread_local bool g_gpu_cache_lookup_bypassed       = false;
inline thread_local bool g_gpu_cache_lookup_bypass_enabled = true;

void SafeClearGpuCacheNoThrow();

inline void ResetGpuCacheBypassState() {
  g_gpu_cache_low_hit_streak  = 0;
  g_gpu_cache_lookup_bypassed = false;
}

inline bool ShouldBypassGpuCacheLookup(int64_t num_keys) {
  return g_gpu_cache_lookup_bypass_enabled &&
         num_keys >= kGpuCacheBypassMinRows &&
         g_gpu_cache_low_hit_streak >= kGpuCacheLowHitLimit;
}

inline void RecordGpuCacheLookupOutcome(
    int64_t num_keys, double hit_count, double request_count) {
  if (num_keys < kGpuCacheBypassMinRows || request_count <= 0.0) {
    return;
  }
  const double hit_ratio = hit_count / request_count;
  if (hit_ratio < kGpuCacheLowHitRatio) {
    ++g_gpu_cache_low_hit_streak;
  } else {
    g_gpu_cache_low_hit_streak  = 0;
    g_gpu_cache_lookup_bypassed = false;
  }
}

inline bool ShouldBypassGpuCacheMaintenance(int64_t num_keys) {
  return g_gpu_cache_lookup_bypass_enabled &&
         num_keys >= kGpuCacheBypassMinRows && g_gpu_cache_lookup_bypassed;
}

inline void MarkGpuCacheLookupBypassed() {
  if (!g_gpu_cache_lookup_bypassed) {
    SafeClearGpuCacheNoThrow();
    g_gpu_cache_low_hit_streak = kGpuCacheLowHitLimit;
  }
  g_gpu_cache_lookup_bypassed = true;
}

inline void EnsureGpuCacheSafeForLookup() {
  if (g_gpu_cache_lookup_bypassed) {
    SafeClearGpuCacheNoThrow();
    ResetGpuCacheBypassState();
  }
}

inline void SafeClearGpuCacheNoThrow() {
  try {
    gpu::ClearGpuCache();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to clear GPU cache: " << e.what();
  } catch (...) {
    LOG(WARNING) << "Failed to clear GPU cache: unknown exception";
  }
}

inline void SetGpuCacheLookupBypassEnabled(bool enabled) {
  g_gpu_cache_lookup_bypass_enabled = enabled;
  if (!enabled) {
    ResetGpuCacheBypassState();
  }
}

inline void MaintainGpuCacheAfterUpdateNoThrow(
    const torch::Tensor& keys,
    const torch::Tensor& grads,
    int64_t embedding_dim) {
  (void)grads;
  if (!gpu::IsGpuCacheEnabled()) {
    return;
  }
  // When the adaptive bypass system is explicitly disabled, an external
  // controller (e.g. BagPipe) owns cache invalidation: it pushes deltas the
  // cache has already applied, so invalidating here (or clearing the whole
  // cache when keys are on CPU) would needlessly evict valid entries.
  if (!g_gpu_cache_lookup_bypass_enabled) {
    return;
  }
  if (ShouldBypassGpuCacheMaintenance(keys.numel())) {
    gpu::ResetLastGpuCacheProfile();
    return;
  }
  if (gpu::CanUseGpuCache(keys, embedding_dim)) {
    try {
      gpu::InvalidateGpuCache(keys);
      return;
    } catch (const std::exception& e) {
      LOG(WARNING) << "GPU cache invalidation failed after backend update "
                      "succeeded; clearing cache and continuing: "
                   << e.what();
    } catch (...) {
      LOG(WARNING) << "GPU cache invalidation failed after backend update "
                      "succeeded; clearing cache and continuing: "
                   << "unknown exception";
    }
  }
  SafeClearGpuCacheNoThrow();
  gpu::ResetLastGpuCacheProfile();
}

#endif // RECSTORE_ENABLE_GPU_CACHE

} // namespace framework
} // namespace recstore
