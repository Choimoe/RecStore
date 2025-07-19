#include <torch/extension.h>
#include "framework/op.h"
#include "base/tensor.h"
#include "framework/shm_kv_store/shm_kv_store.h"

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

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  auto op = GetKVClientOp();

  const int64_t num_keys = keys.size(0);
  auto values            = torch::empty(
      {num_keys, embedding_dim}, keys.options().dtype(torch::kFloat32));
  TORCH_CHECK(values.is_contiguous(),
              "Internal error: Created values tensor is not contiguous");

  base::RecTensor rec_keys   = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(values, base::DataType::FLOAT32);

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

  auto op = GetKVClientOp();

  base::RecTensor rec_keys  = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(grads, base::DataType::FLOAT32);

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

  auto op = GetKVClientOp();

  base::RecTensor rec_keys   = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(values, base::DataType::FLOAT32);

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
