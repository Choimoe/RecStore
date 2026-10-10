#include "framework/pytorch/op_bagpipe.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

// GPU-cache-accelerated flat lookup that works with ANY backend (BRPC, GRPC,
// RDMA, local_shm).  Cache hits are served from the GPU cache; misses are
// fetched via EmbRead and filled back into the cache.  This is the forward
// path used by the BagPipe controller when the local_shm fast path is
// unavailable, so the GPU cache is actually queried instead of bypassed.
torch::Tensor
gpu_cache_lookup_flat_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  ResetLocalLookupFlatProfile();
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
#endif
  const auto total_start = SteadyNow();
  const bool is_cuda     = keys.is_cuda();
  auto orig_device       = keys.device();

  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool can_use_gpu_cache = gpu::CanUseGpuCache(keys, embedding_dim);
  const bool bypass_gpu_cache_lookup =
      can_use_gpu_cache && ShouldBypassGpuCacheLookup(num_keys);
  if (bypass_gpu_cache_lookup) {
    MarkGpuCacheLookupBypassed();
  }
  if (can_use_gpu_cache && !bypass_gpu_cache_lookup) {
    EnsureGpuCacheSafeForLookup();
    try {
      auto cache_result = gpu::QueryGpuCache(keys, embedding_dim);
      RecordGpuCacheLookupOutcome(
          num_keys,
          static_cast<double>(num_keys - cache_result.missing_count),
          static_cast<double>(num_keys));
      if (cache_result.missing_count == 0) {
        g_last_local_lookup_flat_profile[kLookupTotalMs] =
            ElapsedMs(total_start);
        return cache_result.values;
      }

      // Fetch misses via EmbRead (works with BRPC / GRPC / RDMA).
      const auto backend_start = SteadyNow();
      auto miss_cpu_keys       = cache_result.missing_keys_cpu.contiguous();
      const int64_t miss_count = miss_cpu_keys.size(0);
      auto miss_cpu_values     = torch::empty(
          {miss_count, embedding_dim},
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      auto op = GetKVClientOp();
      base::RecTensor rec_miss_keys =
          ToRecTensor(miss_cpu_keys, base::DataType::UINT64);
      base::RecTensor rec_miss_values =
          ToRecTensor(miss_cpu_values, base::DataType::FLOAT32);
      op->EmbRead(rec_miss_keys, rec_miss_values);
      gpu::AddGpuCacheBackendLookupMs(ElapsedMs(backend_start));

      auto miss_keys_cuda =
          miss_cpu_keys.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          miss_cpu_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "gpu_cache_lookup_flat: cache lookup failed; falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    } catch (...) {
      LOG(WARNING)
          << "gpu_cache_lookup_flat: cache lookup failed; falling back";
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    }
  }
#endif

  // Fallback: direct EmbRead (no GPU cache).
  torch::Tensor cpu_keys = keys;
  if (is_cuda) {
    const auto stage_start = SteadyNow();
    cpu_keys               = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
    g_last_local_lookup_flat_profile[kLookupKeysStageMs] =
        ElapsedMs(stage_start);
  }
  auto op         = GetKVClientOp();
  auto cpu_values = torch::empty(
      {cpu_keys.size(0), embedding_dim},
      is_cuda
          ? PinnedCpuOptions(torch::kFloat32)
          : torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);
  op->EmbRead(rec_keys, rec_values);
  g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
  if (is_cuda) {
    return cpu_values.to(orig_device, /*non_blocking=*/true);
  }
  return cpu_values;
}

std::tuple<torch::Tensor, torch::Tensor> gpu_cache_lookup_flat_no_evict_torch(
    const torch::Tensor& keys, int64_t embedding_dim) {
  ResetLocalLookupFlatProfile();
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
#endif
  const auto total_start = SteadyNow();
  const auto orig_device = keys.device();

  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = keys.size(0);
  auto resident = torch::ones({num_keys}, keys.options().dtype(torch::kBool));
  if (num_keys == 0) {
    return {
        torch::empty({0, embedding_dim}, keys.options().dtype(torch::kFloat32)),
        resident};
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(gpu::CanUseGpuCache(keys, embedding_dim),
              "gpu_cache_lookup_flat_no_evict requires an enabled GPU cache "
              "on the keys' CUDA device");
  auto cache_result = gpu::QueryGpuCache(keys, embedding_dim);
  RecordGpuCacheLookupOutcome(
      num_keys,
      static_cast<double>(num_keys - cache_result.missing_count),
      static_cast<double>(num_keys));
  if (cache_result.missing_count == 0) {
    g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
    return {cache_result.values, resident};
  }

  const auto backend_start = SteadyNow();
  auto miss_cpu_keys       = cache_result.missing_keys_cpu.contiguous();
  const int64_t miss_count = miss_cpu_keys.size(0);
  auto miss_cpu_values     = torch::empty(
      {miss_count, embedding_dim},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto op = GetKVClientOp();
  base::RecTensor rec_miss_keys =
      ToRecTensor(miss_cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_miss_values =
      ToRecTensor(miss_cpu_values, base::DataType::FLOAT32);
  op->EmbRead(rec_miss_keys, rec_miss_values);
  gpu::AddGpuCacheBackendLookupMs(ElapsedMs(backend_start));

  auto miss_keys_cuda = miss_cpu_keys.to(orig_device, /*non_blocking=*/false);
  auto miss_values_cuda =
      miss_cpu_values.to(orig_device, /*non_blocking=*/false);
  auto inserted = gpu::FillGpuCacheNoEvict(miss_keys_cuda, miss_values_cuda);
  gpu::ScatterMissValues(&cache_result.values,
                         cache_result.missing_positions_cpu,
                         miss_values_cuda);
  auto positions_cuda = cache_result.missing_positions_cpu.to(orig_device);
  resident.index_put_({positions_cuda}, inserted);
  g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
  return {cache_result.values, resident};
#else
  (void)total_start;
  (void)orig_device;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

void emb_write_values_torch(const torch::Tensor& keys,
                            const torch::Tensor& values) {
  // Direct value-set to the PS for a subset of keys, with *per-key* GPU
  // cache invalidation (not a full clear).  Used by the BagPipe eviction
  // writeback path to push locally-updated cache values back to the PS
  // without disturbing other cached entries.  Mirrors emb_write_torch but
  // replaces SafeClearGpuCacheNoThrow() with InvalidateGpuCache(keys).
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must be float32");
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "Keys and Values tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  if (values.is_cuda()) {
    cpu_values = values.cpu();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  op->EmbWrite(rec_keys, rec_values);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (gpu::IsGpuCacheEnabled()) {
    int64_t embedding_dim = values.size(1);
    if (keys.is_cuda() && gpu::CanUseGpuCache(keys, embedding_dim)) {
      try {
        gpu::InvalidateGpuCache(keys);
      } catch (...) {
        SafeClearGpuCacheNoThrow();
      }
    } else {
      SafeClearGpuCacheNoThrow();
    }
    gpu::ResetLastGpuCacheProfile();
  }
#endif
}

bool enable_gpu_cache_torch(int64_t capacity, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool enabled = gpu::EnableGpuCache(capacity, embedding_dim);
  if (enabled) {
    ResetGpuCacheBypassState();
  }
  return enabled;
#else
  (void)capacity;
  (void)embedding_dim;
  return false;
#endif
}

void disable_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::DisableGpuCache();
  ResetGpuCacheBypassState();
#endif
}

void clear_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ClearGpuCache();
  ResetGpuCacheBypassState();
#endif
}

void prefill_gpu_cache_torch(const torch::Tensor& keys,
                             const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  TORCH_CHECK(values.dim() == 2, "values must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must have dtype float32");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "keys and values must have the same number of rows");
  if (keys.numel() == 0) {
    return;
  }
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "prefill_gpu_cache requires keys or values on CUDA");
  const auto cache_device = values.is_cuda() ? values.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto values_cuda        = values.is_cuda() ? values : values.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!values_cuda.is_contiguous()) {
    values_cuda = values_cuda.contiguous();
  }
  gpu::FillGpuCache(keys_cuda, values_cuda);
#else
  (void)keys;
  (void)values;
#endif
}

torch::Tensor prefill_gpu_cache_no_evict_torch(const torch::Tensor& keys,
                                               const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  TORCH_CHECK(values.dim() == 2, "values must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must have dtype float32");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "keys and values must have the same number of rows");
  if (keys.numel() == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool));
  }
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "prefill_gpu_cache_no_evict requires keys or values on CUDA");
  const auto cache_device = values.is_cuda() ? values.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto values_cuda        = values.is_cuda() ? values : values.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!values_cuda.is_contiguous()) {
    values_cuda = values_cuda.contiguous();
  }
  return gpu::FillGpuCacheNoEvict(keys_cuda, values_cuda);
#else
  (void)keys;
  (void)values;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

torch::Tensor contains_gpu_cache_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return gpu::ContainsGpuCache(keys);
#else
  (void)keys;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

int64_t get_gpu_cache_generation_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return static_cast<int64_t>(gpu::GetGpuCacheGeneration());
#else
  return 0;
#endif
}

void invalidate_gpu_cache_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  if (keys.numel() == 0) {
    return;
  }
  TORCH_CHECK(keys.is_cuda(), "invalidate_gpu_cache requires keys on CUDA");
  auto keys_cuda = keys;
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  gpu::InvalidateGpuCache(keys_cuda);
#else
  (void)keys;
#endif
}

torch::Tensor invalidate_gpu_cache_with_mask_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  if (keys.numel() == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool));
  }
  TORCH_CHECK(keys.is_cuda(),
              "invalidate_gpu_cache_with_mask requires CUDA keys");
  auto keys_cuda = keys;
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  return gpu::InvalidateGpuCacheWithMask(keys_cuda);
#else
  (void)keys;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
gpu_cache_lookup_flat_assuming_hits_torch(const torch::Tensor& keys,
                                          int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(keys.is_cuda(),
              "gpu_cache_lookup_flat_assuming_hits requires CUDA keys");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");
  return gpu::LookupGpuCacheAssumingHits(keys, embedding_dim);
#else
  (void)keys;
  (void)embedding_dim;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

bool apply_sgd_update_gpu_cache_torch(const torch::Tensor& keys,
                                      const torch::Tensor& grads,
                                      double learning_rate) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  TORCH_CHECK(grads.dim() == 2, "grads must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "grads must have dtype float32");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "keys and grads must have the same number of rows");
  if (keys.numel() == 0) {
    return true;
  }
  TORCH_CHECK(keys.is_cuda() || grads.is_cuda(),
              "apply_sgd_update_gpu_cache requires keys or grads on CUDA");
  const auto cache_device = grads.is_cuda() ? grads.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto grads_cuda         = grads.is_cuda() ? grads : grads.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!grads_cuda.is_contiguous()) {
    grads_cuda = grads_cuda.contiguous();
  }
  return gpu::ApplySgdUpdateGpuCache(keys_cuda, grads_cuda, learning_rate);
#else
  (void)keys;
  (void)grads;
  (void)learning_rate;
  return false;
#endif
}

// Best-effort in-place SGD on the GPU cache: value -= lr * grad for keys
// present in the cache; missing keys silently skipped.  No Query, no
// missing report, no device synchronization.  This is the cache-side SGD
// primitive for controllers (BagPipe) that own their own cache policy.
void apply_sgd_update_gpu_cache_best_effort_torch(
    const torch::Tensor& keys,
    const torch::Tensor& grads,
    double learning_rate) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must be int64");
  TORCH_CHECK(grads.dim() == 2, "grads must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32, "grads must be float32");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "keys and grads must have the same number of rows");
  if (keys.numel() == 0) {
    return;
  }
  TORCH_CHECK(
      keys.is_cuda() || grads.is_cuda(),
      "apply_sgd_update_gpu_cache_best_effort requires keys or grads on CUDA");
  const auto cache_device = grads.is_cuda() ? grads.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto grads_cuda         = grads.is_cuda() ? grads : grads.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!grads_cuda.is_contiguous()) {
    grads_cuda = grads_cuda.contiguous();
  }
  gpu::ApplySgdUpdateBestEffortGpuCache(keys_cuda, grads_cuda, learning_rate);
#else
  (void)keys;
  (void)grads;
  (void)learning_rate;
#endif
}

void set_gpu_cache_lookup_bypass_enabled_torch(bool enabled) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  SetGpuCacheLookupBypassEnabled(enabled);
#else
  (void)enabled;
#endif
}

bool is_gpu_cache_lookup_bypass_enabled_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_gpu_cache_lookup_bypass_enabled;
#else
  return false;
#endif
}

bool is_gpu_cache_lookup_bypassed_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_gpu_cache_lookup_bypassed;
#else
  return false;
#endif
}

void reset_gpu_cache_bypass_state_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  ResetGpuCacheBypassState();
#endif
}

// ---- BagPipe-style GPU cache ops (query / update / invalidate / sgd) ----
std::tuple<torch::Tensor, torch::Tensor>
query_gpu_cache_torch(const torch::Tensor& keys, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (!gpu::IsGpuCacheEnabled() || keys.numel() == 0) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto dev  = keys.is_cuda() ? keys.device() : torch::kCPU;
    return {torch::empty({0, embedding_dim}, opts.device(dev)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))};
  }
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  auto keys_contig = keys.is_contiguous() ? keys : keys.contiguous();
  auto result      = gpu::QueryGpuCache(keys_contig, embedding_dim);
  return {result.values, result.missing_keys_cpu};
#else
  (void)keys;
  (void)embedding_dim;
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  return {torch::empty({0, 1}, opts),
          torch::empty({0}, opts.dtype(torch::kInt64))};
#endif
}

void update_gpu_cache_torch(const torch::Tensor& keys,
                            const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (keys.numel() == 0)
    return;
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must be int64");
  TORCH_CHECK(values.dim() == 2, "values must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must be float32");
  TORCH_CHECK(keys.size(0) == values.size(0), "row count mismatch");
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "update_gpu_cache requires keys or values on CUDA");
  const auto dev   = values.is_cuda() ? values.device() : keys.device();
  auto keys_cuda   = keys.is_cuda() ? keys : keys.to(dev);
  auto values_cuda = values.is_cuda() ? values : values.to(dev);
  if (!keys_cuda.is_contiguous())
    keys_cuda = keys_cuda.contiguous();
  if (!values_cuda.is_contiguous())
    values_cuda = values_cuda.contiguous();
  gpu::UpdateGpuCache(keys_cuda, values_cuda);
#else
  (void)keys;
  (void)values;
#endif
}

} // namespace

void RegisterBagpipeOps(torch::Library& m) {
  m.def("gpu_cache_lookup_flat", gpu_cache_lookup_flat_torch);
  m.def("gpu_cache_lookup_flat_no_evict", gpu_cache_lookup_flat_no_evict_torch);
  m.def("emb_write_values", emb_write_values_torch);
  m.def("enable_gpu_cache", enable_gpu_cache_torch);
  m.def("disable_gpu_cache", disable_gpu_cache_torch);
  m.def("clear_gpu_cache", clear_gpu_cache_torch);
  m.def("prefill_gpu_cache", prefill_gpu_cache_torch);
  m.def("prefill_gpu_cache_no_evict", prefill_gpu_cache_no_evict_torch);
  m.def("contains_gpu_cache", contains_gpu_cache_torch);
  m.def("get_gpu_cache_generation", get_gpu_cache_generation_torch);
  m.def("invalidate_gpu_cache", invalidate_gpu_cache_torch);
  m.def("invalidate_gpu_cache_with_mask", invalidate_gpu_cache_with_mask_torch);
  m.def("gpu_cache_lookup_flat_assuming_hits",
        gpu_cache_lookup_flat_assuming_hits_torch);
  m.def("apply_sgd_update_gpu_cache", apply_sgd_update_gpu_cache_torch);
  m.def("apply_sgd_update_gpu_cache_best_effort",
        apply_sgd_update_gpu_cache_best_effort_torch);
  m.def("set_gpu_cache_lookup_bypass_enabled",
        set_gpu_cache_lookup_bypass_enabled_torch);
  m.def("is_gpu_cache_lookup_bypass_enabled",
        is_gpu_cache_lookup_bypass_enabled_torch);
  m.def("is_gpu_cache_lookup_bypassed", is_gpu_cache_lookup_bypassed_torch);
  m.def("reset_gpu_cache_bypass_state", reset_gpu_cache_bypass_state_torch);
  m.def("query_gpu_cache", query_gpu_cache_torch);
  m.def("update_gpu_cache", update_gpu_cache_torch);
}

} // namespace framework
} // namespace recstore
