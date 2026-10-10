#include "framework/pytorch/op_base.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  bool is_cuda     = keys.is_cuda();
  auto orig_device = keys.device();

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

  auto op = GetKVClientOp();

#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
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
        return cache_result.values;
      }

      auto missing_cpu_values = torch::empty(
          {cache_result.missing_count, embedding_dim},
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      base::RecTensor rec_missing_keys = ToRecTensor(
          cache_result.missing_keys_cpu.contiguous(), base::DataType::UINT64);
      base::RecTensor rec_missing_values =
          ToRecTensor(missing_cpu_values, base::DataType::FLOAT32);
      const auto backend_start = SteadyNow();
      op->EmbRead(rec_missing_keys, rec_missing_values);
      gpu::AddGpuCacheBackendLookupMs(ElapsedMs(backend_start));

      auto miss_keys_cuda =
          cache_result.missing_keys_cpu.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          missing_cpu_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "GPU cache emb_read failed; clearing cache and falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    } catch (...) {
      LOG(WARNING)
          << "GPU cache emb_read failed; clearing cache and falling back: "
          << "unknown exception";
      SafeClearGpuCacheNoThrow();
      gpu::ResetLastGpuCacheProfile();
    }
  }
#endif

  torch::Tensor cpu_keys = is_cuda ? keys.cpu() : keys;

  auto cpu_values = torch::empty(
      {num_keys, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  op->EmbRead(rec_keys, rec_values);

  if (is_cuda) {
    return cpu_values.to(orig_device);
  }
  return cpu_values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse "
      "optimizer.");
}

void emb_update_table_torch(const std::string& table_name,
                            const torch::Tensor& keys,
                            const torch::Tensor& grads) {
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

  if (keys.size(0) == 0) {
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys  = keys;
  torch::Tensor cpu_grads = grads;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  if (grads.is_cuda()) {
    cpu_grads = grads.cpu();
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  op->EmbUpdate(table_name, rec_keys, rec_grads);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  MaintainGpuCacheAfterUpdateNoThrow(keys, grads, grads.size(1));
#endif
}

int64_t emb_update_async_torch(const std::string& table_name,
                               const torch::Tensor& keys,
                               const torch::Tensor& grads) {
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

  torch::Tensor cpu_keys  = keys.is_cuda() ? keys.cpu() : keys;
  torch::Tensor cpu_grads = grads.is_cuda() ? grads.cpu() : grads;
  auto kv_op              = GetConcreteKVClientOp();
  TORCH_CHECK(kv_op->CurrentPSBackend() == "rdma",
              "emb_update_async requires the RDMA backend");
  const uint64_t update_id = kv_op->EmbUpdateAsync(
      table_name,
      ToRecTensor(cpu_keys, base::DataType::UINT64),
      ToRecTensor(cpu_grads, base::DataType::FLOAT32));

#ifdef RECSTORE_ENABLE_GPU_CACHE
  std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
  const auto [it, inserted] = g_pending_gpu_cache_updates.emplace(
      update_id, PendingGpuCacheUpdate{keys, grads});
  TORCH_CHECK(inserted, "Duplicate asynchronous RDMA update handle");
#endif
  return static_cast<int64_t>(update_id);
}

void emb_update_wait_torch(int64_t update_id) {
  TORCH_CHECK(update_id > 0, "update_id must be positive");
  auto kv_op = GetConcreteKVClientOp();
  try {
    kv_op->WaitForEmbUpdate(static_cast<uint64_t>(update_id));
  } catch (...) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
    {
      std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
      g_pending_gpu_cache_updates.erase(static_cast<uint64_t>(update_id));
    }
    SafeClearGpuCacheNoThrow();
#endif
    throw;
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  PendingGpuCacheUpdate pending;
  {
    std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
    const auto it =
        g_pending_gpu_cache_updates.find(static_cast<uint64_t>(update_id));
    TORCH_CHECK(it != g_pending_gpu_cache_updates.end(),
                "Missing GPU cache state for asynchronous RDMA update");
    pending = std::move(it->second);
    g_pending_gpu_cache_updates.erase(it);
  }
  MaintainGpuCacheAfterUpdateNoThrow(
      pending.keys, pending.grads, pending.grads.size(1));
#endif
}

int64_t init_embedding_table_torch(
    const std::string& table_name,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t table_id = 0) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");
  TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");
  TORCH_CHECK(table_id >= 0, "table_id must be non-negative");

  EmbeddingTableConfig cfg{};
  cfg.num_embeddings = static_cast<uint64_t>(num_embeddings);
  cfg.embedding_dim  = static_cast<uint64_t>(embedding_dim);
  cfg.table_id       = static_cast<uint64_t>(table_id);

  auto kv_op        = GetConcreteKVClientOp();
  const int64_t tag = kv_op->InitEmbeddingTable(table_name, cfg);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (tag >= 0 && gpu::IsGpuCacheEnabled()) {
    SafeClearGpuCacheNoThrow();
    gpu::ResetLastGpuCacheProfile();
  }
#endif
  return tag;
}

bool save_checkpoint_torch(const std::string& path,
                           const std::string& metadata) {
  TORCH_CHECK(!path.empty(), "checkpoint path must be non-empty");
  TORCH_CHECK(!metadata.empty(), "checkpoint metadata must be non-empty");
  return GetKVClientOp()->SaveCheckpoint(path, metadata);
}

bool load_checkpoint_torch(const std::string& path,
                           const std::string& metadata) {
  TORCH_CHECK(!path.empty(), "checkpoint path must be non-empty");
  TORCH_CHECK(!metadata.empty(), "checkpoint metadata must be non-empty");
  const bool ok = GetKVClientOp()->LoadCheckpoint(path, metadata);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (ok && gpu::IsGpuCacheEnabled()) {
    SafeClearGpuCacheNoThrow();
    gpu::ResetLastGpuCacheProfile();
  }
#endif
  return ok;
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must have dtype float32");
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
    SafeClearGpuCacheNoThrow();
    gpu::ResetLastGpuCacheProfile();
  }
#endif
}

void set_ps_config_torch(const std::string& host, int64_t port) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSConfig(host, static_cast<int>(port));
}

void set_ps_backend_torch(const std::string& backend) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSBackend(backend);
}

std::string current_ps_backend_torch() {
  auto kv_op = GetConcreteKVClientOp();
  return kv_op->CurrentPSBackend();
}

} // namespace

void RegisterBaseOps(torch::Library& m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_update_table", emb_update_table_torch);
  m.def("emb_update_async", emb_update_async_torch);
  m.def("emb_update_wait", emb_update_wait_torch);
  m.def("init_embedding_table", init_embedding_table_torch);
  m.def("save_checkpoint", save_checkpoint_torch);
  m.def("load_checkpoint", load_checkpoint_torch);
  m.def("emb_write", emb_write_torch);
  m.def("set_ps_config", set_ps_config_torch);
  m.def("set_ps_backend", set_ps_backend_torch);
  m.def("current_ps_backend", current_ps_backend_torch);
}

} // namespace framework
} // namespace recstore
