#include "framework/pytorch/op_internal.h"

namespace recstore::framework {

void CheckKeys(const torch::Tensor& keys, bool contiguous) {
  TORCH_CHECK(keys.dim() == 1 && keys.scalar_type() == torch::kInt64 &&
                  (!contiguous || keys.is_contiguous()),
              "Keys must be contiguous 1-D int64");
}

void CheckRows(const torch::Tensor& keys,
               const torch::Tensor& values,
               bool contiguous) {
  CheckKeys(keys, contiguous);
  TORCH_CHECK(values.dim() == 2 && values.scalar_type() == torch::kFloat32 &&
                  (!contiguous || values.is_contiguous()) &&
                  values.size(0) == keys.numel(),
              "Values must be contiguous 2-D float32 with one row per key");
}

std::pair<torch::Tensor, torch::Tensor>
CudaCacheRows(const torch::Tensor& keys, const torch::Tensor& values) {
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "Cache rows require a CUDA tensor");
  auto device = values.is_cuda() ? values.device() : keys.device();
  return {(keys.is_cuda() ? keys : keys.to(device)).contiguous(),
          (values.is_cuda() ? values : values.to(device)).contiguous()};
}

CacheUpdateActions CacheActions(const torch::Tensor& keys, int64_t dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return {[] { return torch_cache::IsGpuCacheEnabled(); },
          [keys, dim] {
            return keys.defined() && torch_cache::CanUseGpuCache(keys, dim);
          },
          [keys] { torch_cache::InvalidateGpuCache(keys); },
          [] {
            torch_cache::ClearGpuCache();
            gpu::ResetLastGpuCacheProfile();
          }};
#else
  return {};
#endif
}

namespace {
// Allocation and transport-buffer wrapping only. The common op owns the lease
// and validates its payload. The CUDA event retains it through the H2D copy.
torch::Tensor ReadLocalValues(const std::shared_ptr<KVClientOp>& op,
                              const torch::Tensor& cpu_keys,
                              int64_t dim,
                              const torch::Device& device) {
  auto keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  if (device.is_cuda() && op->SupportsBorrowedLocalLookup()) {
    auto lease        = op->AcquireLocalLookupFlat(keys, dim);
    auto start        = SteadyNow();
    const auto bytes  = lease->num_elements() * sizeof(float);
    const bool pinned = EnsurePinnedLocalShmPayload(lease->data(), bytes);
    g_last_local_lookup_flat_profile[kLookupPayloadPinMs] = ElapsedMs(start);
    auto view                                             = torch::from_blob(
        lease->data(),
        {cpu_keys.numel(), dim},
        [op, lease](void*) {},
        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(pinned));
    torch::Tensor staging = view;
    if (!pinned) {
      start   = SteadyNow();
      staging = torch::empty(view.sizes(), PinnedCpuOptions(torch::kFloat32));
      staging.copy_(view);
      g_last_local_lookup_flat_profile[kLookupFallbackCopyMs] =
          ElapsedMs(start);
    }
    start       = SteadyNow();
    auto result = staging.to(device, /*non_blocking=*/true);
    // from_blob storage is not owned by the PyTorch pinned allocator. Retain
    // the transport lease explicitly until this stream finishes reading it.
    RetainUntilCudaComplete(result, [op, lease] {});
    g_last_local_lookup_flat_profile[kLookupValuesH2DEnqueueMs] =
        ElapsedMs(start);
    return result;
  }
  auto values = torch::empty(
      {cpu_keys.numel(), dim},
      device.is_cuda() ? PinnedCpuOptions(torch::kFloat32)
                       : torch::TensorOptions().dtype(torch::kFloat32));
  auto out = ToRecTensor(values, base::DataType::FLOAT32);
  op->LocalLookupFlat(keys, out);
  return values.to(device, /*non_blocking=*/true);
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor> CachedReadTorch(
    const torch::Tensor& keys, int64_t dim, bool local, bool no_evict) {
  CheckKeys(keys);
  TORCH_CHECK(dim > 0, "Embedding dimension must be positive");
  ResetLocalLookupFlatProfile();
  const auto start = SteadyNow();
  std::shared_ptr<KVClientOp> op;
  if (local) {
    op = GetConcreteKVClientOp();
    op->RequireLocalBackend();
  }
  auto resident =
      no_evict ? torch::ones({keys.numel()}, keys.options().dtype(torch::kBool))
               : torch::Tensor();
  if (!keys.numel())
    return {
        torch::empty({0, dim},
                     no_evict ? keys.options().dtype(torch::kFloat32)
                              : torch::TensorOptions().dtype(torch::kFloat32)),
        resident};
  auto read = [&](const torch::Tensor& cpu_keys, const torch::Device& device) {
    if (!op)
      op = GetConcreteKVClientOp();
    if (local)
      return ReadLocalValues(op, cpu_keys, dim, device);
    auto values = torch::empty(
        {cpu_keys.numel(), dim},
        device.is_cuda() ? PinnedCpuOptions(torch::kFloat32)
                         : torch::TensorOptions().dtype(torch::kFloat32));
    auto k = ToRecTensor(cpu_keys, base::DataType::UINT64);
    auto v = ToRecTensor(values, base::DataType::FLOAT32);
    op->EmbRead(k, v);
    return values.to(device, /*non_blocking=*/true);
  };
  torch::Tensor output;
  CacheLookupActions io;
  io.read_all = [&] {
    auto stage_start = SteadyNow();
    auto cpu_keys =
        keys.is_cuda() ? StageCudaTensorToPinnedCpu(keys, torch::kInt64) : keys;
    if (keys.is_cuda())
      g_last_local_lookup_flat_profile[kLookupKeysStageMs] =
          ElapsedMs(stage_start);
    output = read(cpu_keys, keys.device());
  };
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
  torch_cache::GpuCacheLookupResult cached;
  torch::Tensor miss_keys, miss_values, inserted;
  io.usable = torch_cache::CanUseGpuCache(keys, dim);
  io.query  = [&] {
    cached = torch_cache::QueryGpuCache(keys, dim);
    output = cached.values;
    return cached.missing_count;
  };
  io.read_missing = [&] {
    auto begin  = SteadyNow();
    auto values = read(cached.missing_keys_cpu.contiguous(), torch::kCPU);
    gpu::AddGpuCacheBackendLookupMs(ElapsedMs(begin));
    miss_keys   = cached.missing_keys_cpu.to(keys.device());
    miss_values = values.to(keys.device());
  };
  io.fill = [&] {
    if (no_evict)
      inserted = torch_cache::FillGpuCacheNoEvict(miss_keys, miss_values);
    else
      torch_cache::FillGpuCache(miss_keys, miss_values);
  };
  io.scatter = [&] {
    torch_cache::ScatterMissValues(
        &output, cached.missing_positions_cpu, miss_values);
    if (no_evict)
      resident.index_put_(
          {cached.missing_positions_cpu.to(keys.device())}, inserted);
  };
  io.clear = [] {
    torch_cache::ClearGpuCache();
    gpu::ResetLastGpuCacheProfile();
  };
#endif
  g_cache_aware_op.Lookup(keys.numel(), io, no_evict);
  g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(start);
  return {output, resident};
}

} // namespace recstore::framework
