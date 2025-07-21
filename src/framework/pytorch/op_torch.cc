#include <torch/extension.h>
#include "framework/op.h"
#include "base/tensor.h"
#include "framework/shm_kv_store/shm_kv_store.h"
#include "framework/gpu/gpu_kv_store.h"

namespace recstore {
namespace framework {

static inline base::RecTensor ToRecTensor(const torch::Tensor& tensor) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()),
                         shape,
                         base::FromTorchDType(tensor.scalar_type()));
}

static inline torch::Device GetBackendDevice(std::shared_ptr<CommonOp>& op) {
  auto gpu_op = std::dynamic_pointer_cast<GPUKVStore>(op);
  if (gpu_op) {
    return gpu_op->device();
  }
  return torch::kCPU;
}

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  auto op              = GetKVClientOp();
  torch::Device device = GetBackendDevice(op);

  auto keys_on_device = keys.to(device, /*non_blocking=*/true);

  auto values = torch::empty({keys.size(0), embedding_dim},
                             torch::dtype(torch::kFloat32).device(device));

  base::RecTensor rec_keys   = ToRecTensor(keys_on_device);
  base::RecTensor rec_values = ToRecTensor(values);

  op->EmbRead(rec_keys, rec_values);

  return values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and Grads tensors must have the same number of entries");

  auto op              = GetKVClientOp();
  torch::Device device = GetBackendDevice(op);

  auto keys_on_device  = keys.to(device, /*non_blocking=*/true);
  auto grads_on_device = grads.to(device, /*non_blocking=*/true);

  base::RecTensor rec_keys  = ToRecTensor(keys_on_device);
  base::RecTensor rec_grads = ToRecTensor(grads_on_device);

  op->EmbUpdate(rec_keys, rec_grads);
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

  auto op              = GetKVClientOp();
  torch::Device device = GetBackendDevice(op);

  auto keys_on_device   = keys.to(device, /*non_blocking=*/true);
  auto values_on_device = values.to(device, /*non_blocking=*/true);

  base::RecTensor rec_keys   = ToRecTensor(keys_on_device);
  base::RecTensor rec_values = ToRecTensor(values_on_device);

  op->EmbWrite(rec_keys, rec_values);
}

void emb_barrier_torch() {
  auto op = GetKVClientOp();
  op->barrier();
}

void emb_reset_dimension_torch() {
  auto op = GetKVClientOp();
  // Cast to SharedMemoryKVStore to access reset_embedding_dimension
  auto shm_op = std::dynamic_pointer_cast<SharedMemoryKVStore>(op);
  if (shm_op) {
    shm_op->reset_embedding_dimension();
  }
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_write", emb_write_torch);
  m.def("emb_barrier", emb_barrier_torch);
  m.def("emb_reset_dimension", emb_reset_dimension_torch);
}

} // namespace framework
} // namespace recstore
