#include <torch/extension.h>
#include "framework/op.h"
#include "base/tensor.h"
#include <cstdlib>
#include <iostream>
#include <string>

namespace recstore {
namespace framework {

// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
static int get_log_level() {
  static int level = []() {
    const char* env = std::getenv("RECSTORE_LOG_LEVEL");
    if (!env)
      return 2; // Default INFO
    return std::atoi(env);
  }();
  return level;
}
#define RECSTORE_LOG(level, msg)                                               \
  do {                                                                         \
    if (get_log_level() >= level) {                                            \
      std::cout << msg << std::endl;                                           \
    }                                                                          \
  } while (0)

static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  RECSTORE_LOG(2,
               "[INFO] emb_read_torch called: keys shape="
                   << keys.sizes() << ", dtype=" << keys.dtype()
                   << ", embedding_dim=" << embedding_dim);
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = keys.size(0);
  if (num_keys == 0) {
    RECSTORE_LOG(3, "[DEBUG] emb_read_torch: num_keys==0, returning empty");
    return torch::empty(
        {0, embedding_dim}, keys.options().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

  auto values = torch::empty(
      {num_keys, embedding_dim}, keys.options().dtype(torch::kFloat32));
  TORCH_CHECK(values.is_contiguous(),
              "Internal error: Created values tensor is not contiguous");

  base::RecTensor rec_keys   = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(values, base::DataType::FLOAT32);

  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: calling op->EmbRead");
  op->EmbRead(rec_keys, rec_values);
  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: EmbRead done");

  return values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  RECSTORE_LOG(2,
               "[INFO] emb_update_torch called: keys shape="
                   << keys.sizes() << ", grads shape=" << grads.sizes());
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

  if (keys.size(0) == 0) {
    RECSTORE_LOG(3, "[DEBUG] emb_update_torch: num_keys==0, early return");
    return;
  }

  auto op = GetKVClientOp();

  base::RecTensor rec_keys  = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(grads, base::DataType::FLOAT32);

  RECSTORE_LOG(3, "[DEBUG] emb_update_torch: calling op->EmbUpdate");
  op->EmbUpdate(rec_keys, rec_grads);
  RECSTORE_LOG(3, "[DEBUG] emb_update_torch: EmbUpdate done");
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  RECSTORE_LOG(2,
               "[INFO] emb_write_torch called: keys shape="
                   << keys.sizes() << ", values shape=" << values.sizes());
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
    RECSTORE_LOG(3, "[DEBUG] emb_write_torch: num_keys==0, early return");
    return;
  }

  auto op = GetKVClientOp();

  base::RecTensor rec_keys   = ToRecTensor(keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(values, base::DataType::FLOAT32);

  RECSTORE_LOG(3, "[DEBUG] emb_write_torch: calling op->EmbWrite");
  op->EmbWrite(rec_keys, rec_values);
  RECSTORE_LOG(3, "[DEBUG] emb_write_torch: EmbWrite done");
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_write", emb_write_torch);
}

} // namespace framework
} // namespace recstore
