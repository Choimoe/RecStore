#include "framework/pytorch/op_flat.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

static torch::Tensor BackendLocalLookupFlat(
    const std::shared_ptr<KVClientOp>& kv_op,
    const torch::Tensor& cpu_keys,
    const torch::Device& result_device,
    bool result_on_cuda,
    int64_t embedding_dim,
    const std::chrono::steady_clock::time_point& total_start,
    bool record_profile = true) {
  const int64_t num_keys   = cpu_keys.size(0);
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  if (kv_op->CurrentPSBackend() != "local_shm") {
    auto cpu_values =
        result_on_cuda
            ? torch::empty({num_keys, embedding_dim},
                           PinnedCpuOptions(torch::kFloat32))
            : torch::empty({num_keys, embedding_dim},
                           torch::TensorOptions()
                               .device(torch::kCPU)
                               .dtype(torch::kFloat32));
    base::RecTensor rec_values =
        ToRecTensor(cpu_values, base::DataType::FLOAT32);
    kv_op->LocalLookupFlat(rec_keys, rec_values);
    if (record_profile) {
      g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
    }
    if (result_on_cuda) {
      return cpu_values.to(result_device, /*non_blocking=*/true);
    }
    return cpu_values;
  }

  if (!result_on_cuda) {
    auto cpu_values = torch::empty(
        {num_keys, embedding_dim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    base::RecTensor rec_values =
        ToRecTensor(cpu_values, base::DataType::FLOAT32);
    kv_op->LocalLookupFlat(rec_keys, rec_values);
    if (record_profile) {
      g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
    }
    return cpu_values;
  }

  LocalShmFlatGetHandle handle;
  const auto submit_start = SteadyNow();
  TORCH_CHECK(
      kv_op->SubmitLocalLookupFlat(rec_keys, embedding_dim, &handle) == 0,
      "Failed to submit local_shm flat lookup.");
  if (record_profile) {
    g_last_local_lookup_flat_profile[kLookupSubmitMs] = ElapsedMs(submit_start);
  }
  const auto wait_start = SteadyNow();
  const int wait_ret    = kv_op->WaitLocalLookupFlat(&handle);
  if (record_profile) {
    g_last_local_lookup_flat_profile[kLookupWaitMs] = ElapsedMs(wait_start);
  }
  if (wait_ret != 0) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(false, "Failed to wait for local_shm flat lookup.");
  }
  const float* payload_values = handle.values;
  const int64_t payload_rows  = handle.num_rows;
  const int64_t payload_dim   = handle.embedding_dim;
  const std::size_t payload_bytes =
      static_cast<std::size_t>(handle.output_bytes);
  const int64_t expected_bytes =
      num_keys * embedding_dim * static_cast<int64_t>(sizeof(float));
  if (payload_values == nullptr || payload_rows != num_keys ||
      payload_dim != embedding_dim ||
      static_cast<int64_t>(payload_bytes) != expected_bytes) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(false,
                "local_shm flat lookup returned unexpected payload metadata.");
  }
  const auto pin_start = SteadyNow();
  const bool payload_is_pinned =
      EnsurePinnedLocalShmPayload(payload_values, payload_bytes);
  if (record_profile) {
    g_last_local_lookup_flat_profile[kLookupPayloadPinMs] =
        ElapsedMs(pin_start);
  }
  if (payload_is_pinned) {
    try {
      LocalShmFlatGetHandle handle_for_release = handle;
      auto cpu_view                            = torch::from_blob(
          const_cast<float*>(payload_values),
          {num_keys, embedding_dim},
          [kv_op, handle_for_release](void* /*unused*/) mutable {
            kv_op->ReleaseLocalLookupFlat(&handle_for_release);
          },
          PinnedCpuOptions(torch::kFloat32));
      const auto h2d_start = SteadyNow();
      auto result          = cpu_view.to(result_device, /*non_blocking=*/true);
      if (record_profile) {
        g_last_local_lookup_flat_profile[kLookupValuesH2DEnqueueMs] =
            ElapsedMs(h2d_start);
        g_last_local_lookup_flat_profile[kLookupTotalMs] =
            ElapsedMs(total_start);
      }
      return result;
    } catch (...) {
      kv_op->ReleaseLocalLookupFlat(&handle);
      throw;
    }
  }

  auto cpu_values = torch::empty(
      {num_keys, embedding_dim}, PinnedCpuOptions(torch::kFloat32));
  const auto fallback_copy_start = SteadyNow();
  std::memcpy(cpu_values.data_ptr<float>(), payload_values, payload_bytes);
  if (record_profile) {
    g_last_local_lookup_flat_profile[kLookupFallbackCopyMs] =
        ElapsedMs(fallback_copy_start);
  }
  kv_op->ReleaseLocalLookupFlat(&handle);
  const auto h2d_start = SteadyNow();
  auto result          = cpu_values.to(result_device, /*non_blocking=*/true);
  if (record_profile) {
    g_last_local_lookup_flat_profile[kLookupValuesH2DEnqueueMs] =
        ElapsedMs(h2d_start);
    g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
  }
  return result;
}

torch::Tensor
local_lookup_flat_torch(const torch::Tensor& keys, int64_t embedding_dim) {
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

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(IsLocalFastPathBackend(kv_op->CurrentPSBackend()),
              "local_lookup_flat requires local_shm or hierkv backend, but "
              "current backend is ",
              kv_op->CurrentPSBackend());

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

      const auto backend_start = SteadyNow();
      auto miss_values         = BackendLocalLookupFlat(
          kv_op,
          cache_result.missing_keys_cpu.contiguous(),
          orig_device,
          /*result_on_cuda=*/false,
          embedding_dim,
          total_start);
      const double backend_ms = ElapsedMs(backend_start);
      gpu::AddGpuCacheBackendLookupMs(backend_ms);
      auto miss_keys_cuda =
          cache_result.missing_keys_cpu.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          miss_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      g_last_local_lookup_flat_profile[kLookupTotalMs] = ElapsedMs(total_start);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "GPU cache lookup failed; clearing cache and falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    } catch (...) {
      LOG(WARNING)
          << "GPU cache lookup failed; clearing cache and falling back: "
          << "unknown exception";
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    }
  }
#endif

  torch::Tensor cpu_keys = keys;
  if (is_cuda) {
    const auto stage_start = SteadyNow();
    cpu_keys               = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
    g_last_local_lookup_flat_profile[kLookupKeysStageMs] =
        ElapsedMs(stage_start);
  }

  return BackendLocalLookupFlat(
      kv_op, cpu_keys, orig_device, is_cuda, embedding_dim, total_start);
}

void local_update_flat_torch(const std::string& table_name,
                             const torch::Tensor& keys,
                             const torch::Tensor& grads) {
  ResetLocalUpdateFlatProfile();
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
#endif
  const auto total_start = SteadyNow();
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and grads tensors must have the same number of entries");

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(IsLocalFastPathBackend(kv_op->CurrentPSBackend()),
              "local_update_flat requires local_shm or hierkv backend, but "
              "current backend is ",
              kv_op->CurrentPSBackend());

  if (keys.size(0) == 0) {
    g_last_local_update_flat_profile[kUpdateTotalMs] = ElapsedMs(total_start);
    return;
  }

  torch::Tensor cpu_keys = keys;
  const bool can_async_stage_cuda =
      (keys.is_cuda() || grads.is_cuda()) &&
      (!keys.is_cuda() || !grads.is_cuda() || keys.device() == grads.device());
  bool staged_cuda_async = false;
  if (keys.is_cuda()) {
    const auto keys_stage_start = SteadyNow();
    if (can_async_stage_cuda) {
      cpu_keys          = StageCudaTensorToPinnedCpuAsyncNoCast(keys);
      staged_cuda_async = true;
    } else {
      cpu_keys = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
    }
    g_last_local_update_flat_profile[kUpdateKeysStageMs] =
        ElapsedMs(keys_stage_start);
  }
  torch::Tensor cpu_grads = grads;
  if (grads.is_cuda()) {
    const auto grads_stage_start = SteadyNow();
    if (can_async_stage_cuda) {
      cpu_grads         = StageCudaTensorToPinnedCpuAsyncNoCast(grads);
      staged_cuda_async = true;
    } else {
      cpu_grads = StageCudaTensorToPinnedCpu(grads, torch::kFloat32);
    }
    g_last_local_update_flat_profile[kUpdateGradsStageMs] =
        ElapsedMs(grads_stage_start);
  }
  if (staged_cuda_async) {
    const auto stage_wait_start = SteadyNow();
    SynchronizeCurrentCudaStreamForTensor(keys.is_cuda() ? keys : grads);
    g_last_local_update_flat_profile[kUpdateStageWaitMs] =
        ElapsedMs(stage_wait_start);
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  const auto shm_call_start = SteadyNow();
  try {
    kv_op->LocalUpdateFlat(table_name, rec_keys, rec_grads);
  } catch (...) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
    if (gpu::IsGpuCacheEnabled()) {
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    }
#endif
    throw;
  }
  g_last_local_update_flat_profile[kUpdateShmCallMs] =
      ElapsedMs(shm_call_start);

#ifdef RECSTORE_ENABLE_GPU_CACHE
  MaintainGpuCacheAfterUpdateNoThrow(keys, grads, grads.size(1));
#endif

  g_last_local_update_flat_profile[kUpdateTotalMs] = ElapsedMs(total_start);
}

bool warmup_local_lookup_flat_cuda_region_torch() {
  auto kv_op                = GetConcreteKVClientOp();
  const void* payload_base  = nullptr;
  std::size_t payload_bytes = 0;
  if (!kv_op->GetLocalLookupFlatPayloadRegion(&payload_base, &payload_bytes)) {
    return false;
  }
  return EnsurePinnedLocalShmPayload(payload_base, payload_bytes);
}

} // namespace

void RegisterFlatOps(torch::Library& m) {
  m.def("local_lookup_flat", local_lookup_flat_torch);
  m.def("local_update_flat", local_update_flat_torch);
  m.def("warmup_local_lookup_flat_cuda_region",
        warmup_local_lookup_flat_cuda_region_torch);
}

} // namespace framework
} // namespace recstore
