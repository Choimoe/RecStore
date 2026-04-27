#include <torch/extension.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "base/tensor.h"
#include "framework/op.h"
#include "ps/local_shm/local_shm_client.h"
// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
#include <glog/logging.h>

namespace recstore {
namespace framework {

static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

static torch::TensorOptions PinnedCpuOptions(torch::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCPU)
      .dtype(dtype)
      .pinned_memory(true);
}

static torch::Tensor StageCudaTensorToPinnedCpu(
    const torch::Tensor& tensor, torch::ScalarType dtype) {
  auto cpu_tensor = torch::empty(
      tensor.sizes(),
      PinnedCpuOptions(dtype));
  cpu_tensor.copy_(tensor.to(dtype), /*non_blocking=*/false);
  return cpu_tensor;
}

static bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return false;
  }
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const std::size_t page_bytes = static_cast<std::size_t>(page_size);
  const uintptr_t raw_begin    = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t raw_end      = raw_begin + bytes;
  const uintptr_t page_begin   = raw_begin & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const uintptr_t page_end =
      (raw_end + page_bytes - 1U) & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const std::size_t required_bytes =
      static_cast<std::size_t>(page_end - page_begin);

  static std::mutex mu;
  static std::unordered_map<uintptr_t, std::size_t> registered_bytes_by_base;
  std::lock_guard<std::mutex> guard(mu);
  const std::size_t existing_bytes = registered_bytes_by_base[page_begin];
  if (existing_bytes >= required_bytes) {
    return true;
  }

  void* register_ptr = reinterpret_cast<void*>(page_begin + existing_bytes);
  const std::size_t register_bytes = required_bytes - existing_bytes;
  const cudaError_t err = cudaHostRegister(
      register_ptr,
      register_bytes,
      cudaHostRegisterPortable);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    LOG(WARNING) << "cudaHostRegister failed for local_shm payload: "
                 << cudaGetErrorString(err)
                 << " base=" << reinterpret_cast<void*>(page_begin)
                 << " bytes=" << required_bytes;
    return false;
  }
  registered_bytes_by_base[page_begin] = required_bytes;
  return true;
}

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  bool is_cuda           = keys.is_cuda();
  auto orig_device       = keys.device();
  torch::Tensor cpu_keys = is_cuda ? keys.cpu() : keys;

  TORCH_CHECK(cpu_keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(cpu_keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(cpu_keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = cpu_keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

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

static std::shared_ptr<KVClientOp> GetConcreteKVClientOp() {
  auto op    = GetKVClientOp();
  auto kv_op = std::dynamic_pointer_cast<KVClientOp>(op);
  TORCH_CHECK(
      kv_op != nullptr, "storage backend is not KVClientOp");
  return kv_op;
}

torch::Tensor local_lookup_flat_torch(const torch::Tensor& keys,
                                      int64_t embedding_dim) {
  bool is_cuda           = keys.is_cuda();
  auto orig_device       = keys.device();
  torch::Tensor cpu_keys =
      is_cuda ? StageCudaTensorToPinnedCpu(keys, torch::kInt64) : keys;

  TORCH_CHECK(cpu_keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(cpu_keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(cpu_keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(
      kv_op->CurrentPSBackend() == "local_shm",
      "local_lookup_flat requires local_shm backend, but current backend is ",
      kv_op->CurrentPSBackend());

  const int64_t num_keys = cpu_keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  if (!is_cuda) {
    auto cpu_values = torch::empty(
        {num_keys, embedding_dim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);
    kv_op->LocalLookupFlat(rec_keys, rec_values);
    return cpu_values;
  }

  LocalShmFlatGetHandle handle;
  TORCH_CHECK(
      kv_op->SubmitLocalLookupFlat(rec_keys, embedding_dim, &handle) == 0,
      "Failed to submit local_shm flat lookup.");
  const int wait_ret = kv_op->WaitLocalLookupFlat(&handle);
  if (wait_ret != 0) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(false, "Failed to wait for local_shm flat lookup.");
  }
  const float* payload_values   = handle.values;
  const int64_t payload_rows    = handle.num_rows;
  const int64_t payload_dim     = handle.embedding_dim;
  const std::size_t payload_bytes =
      static_cast<std::size_t>(handle.output_bytes);
  const int64_t expected_bytes =
      num_keys * embedding_dim * static_cast<int64_t>(sizeof(float));
  if (payload_values == nullptr || payload_rows != num_keys ||
      payload_dim != embedding_dim ||
      static_cast<int64_t>(payload_bytes) != expected_bytes) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(
        false,
        "local_shm flat lookup returned unexpected payload metadata.");
  }
  if (EnsurePinnedLocalShmPayload(payload_values, payload_bytes)) {
    try {
      LocalShmFlatGetHandle handle_for_release = handle;
      auto cpu_view = torch::from_blob(
          const_cast<float*>(payload_values),
          {num_keys, embedding_dim},
          [kv_op, handle_for_release](void* /*unused*/) mutable {
            kv_op->ReleaseLocalLookupFlat(&handle_for_release);
          },
          PinnedCpuOptions(torch::kFloat32));
      return cpu_view.to(orig_device, /*non_blocking=*/true);
    } catch (...) {
      kv_op->ReleaseLocalLookupFlat(&handle);
      throw;
    }
  }

  auto cpu_values = torch::empty(
      {num_keys, embedding_dim}, PinnedCpuOptions(torch::kFloat32));
  std::memcpy(cpu_values.data_ptr<float>(), payload_values, payload_bytes);
  kv_op->ReleaseLocalLookupFlat(&handle);
  return cpu_values.to(orig_device, /*non_blocking=*/true);
}

// Async prefetch: returns a unique prefetch id (uint64_t)
int64_t emb_prefetch_torch(const torch::Tensor& keys) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  auto op                = GetKVClientOp();
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  // Dummy values tensor (unused by backend prefetch implementation)
  auto dummy_vals = torch::empty({0, 0}, keys.options().dtype(torch::kFloat32));
  base::RecTensor rec_vals = ToRecTensor(dummy_vals, base::DataType::FLOAT32);
  uint64_t pid             = op->EmbPrefetch(rec_keys, rec_vals);
  return static_cast<int64_t>(pid);
}

// Wait for prefetch and return result tensor [N, embedding_dim] on CPU
torch::Tensor
emb_wait_result_torch(int64_t prefetch_id, int64_t embedding_dim) {
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");
  auto op = GetKVClientOp();
  op->WaitForPrefetch(static_cast<uint64_t>(prefetch_id));
  std::vector<float> flat_values;
  int64_t L = 0;
  op->GetPretchResultFlat(
      static_cast<uint64_t>(prefetch_id), &flat_values, &L, embedding_dim);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out = torch::empty({L, embedding_dim}, options);
  if (L > 0 && !flat_values.empty()) {
    std::memcpy(out.data_ptr<float>(),
                flat_values.data(),
                static_cast<size_t>(L) * static_cast<size_t>(embedding_dim) *
                    sizeof(float));
  }
  return out;
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
}

void local_update_flat_torch(const std::string& table_name,
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

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(
      kv_op->CurrentPSBackend() == "local_shm",
      "local_update_flat requires local_shm backend, but current backend is ",
      kv_op->CurrentPSBackend());

  if (keys.size(0) == 0) {
    return;
  }

  torch::Tensor cpu_keys =
      keys.is_cuda() ? StageCudaTensorToPinnedCpu(keys, torch::kInt64) : keys;
  torch::Tensor cpu_grads =
      grads.is_cuda() ? StageCudaTensorToPinnedCpu(grads, torch::kFloat32) : grads;

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  kv_op->LocalUpdateFlat(table_name, rec_keys, rec_grads);
}

bool init_embedding_table_torch(const std::string& table_name,
                                int64_t num_embeddings,
                                int64_t embedding_dim) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");
  TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");

  EmbeddingTableConfig cfg{};
  cfg.num_embeddings = static_cast<uint64_t>(num_embeddings);
  cfg.embedding_dim  = static_cast<uint64_t>(embedding_dim);

  auto op = GetKVClientOp();
  return op->InitEmbeddingTable(table_name, cfg);
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

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("local_lookup_flat", local_lookup_flat_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_update_table", emb_update_table_torch);
  m.def("local_update_flat", local_update_flat_torch);
  m.def("init_embedding_table", init_embedding_table_torch);
  m.def("emb_write", emb_write_torch);
  m.def("emb_prefetch", emb_prefetch_torch);
  m.def("emb_wait_result", emb_wait_result_torch);
  m.def("set_ps_config", set_ps_config_torch);
  m.def("set_ps_backend", set_ps_backend_torch);
  m.def("current_ps_backend", current_ps_backend_torch);
}

} // namespace framework
} // namespace recstore
