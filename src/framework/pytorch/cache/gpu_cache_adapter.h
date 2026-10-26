#pragma once

#include <torch/extension.h>

namespace recstore::framework::torch_cache {

bool EnableGpuCache(int64_t capacity, int64_t embedding_dim);
void DisableGpuCache();
void ClearGpuCache();
void ResetGpuCache();
bool IsGpuCacheEnabled();
uint64_t GetGpuCacheGeneration();

bool CanUseGpuCache(const torch::Tensor& keys, int64_t embedding_dim);

struct GpuCacheLookupResult {
  torch::Tensor values;
  torch::Tensor missing_keys_cpu;
  torch::Tensor missing_positions_cpu;
  int64_t missing_count = 0;
};

GpuCacheLookupResult
QueryGpuCache(const torch::Tensor& keys, int64_t embedding_dim);
// Returns (values, miss_mask). A true entry in miss_mask means the key was not
// resident when the kernel ran, and the corresponding values row is undefined;
// the caller must backfill those keys through the ordinary lookup path.
std::tuple<torch::Tensor, torch::Tensor>
LookupGpuCacheAssumingHits(const torch::Tensor& keys, int64_t embedding_dim);
torch::Tensor ContainsGpuCache(const torch::Tensor& keys);
void FillGpuCache(const torch::Tensor& keys_cuda,
                  const torch::Tensor& values_cuda);
torch::Tensor FillGpuCacheNoEvict(const torch::Tensor& keys_cuda,
                                  const torch::Tensor& values_cuda);
void ScatterMissValues(torch::Tensor* output_values,
                       const torch::Tensor& missing_positions_cpu,
                       const torch::Tensor& miss_values_cuda);
bool ApplySgdUpdateGpuCache(const torch::Tensor& keys_cuda,
                            const torch::Tensor& grads_cuda,
                            double learning_rate);
// Best-effort in-place SGD: value -= lr * grad on present keys only.
// Missing keys are silently skipped — no Query, no missing report, no
// device synchronization, no cache invalidation.
void ApplySgdUpdateBestEffortGpuCache(const torch::Tensor& keys_cuda,
                                      const torch::Tensor& grads_cuda,
                                      double learning_rate);
void UpdateGpuCache(const torch::Tensor& keys_cuda,
                    const torch::Tensor& values_cuda);
void InvalidateGpuCache(const torch::Tensor& keys_cuda);
torch::Tensor InvalidateGpuCacheWithMask(const torch::Tensor& keys_cuda);

} // namespace recstore::framework::torch_cache
