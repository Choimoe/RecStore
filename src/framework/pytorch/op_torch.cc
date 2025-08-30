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
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_read_torch: keys shape="
                   << keys.sizes() << ", dtype=" << keys.dtype()
                   << ", data_ptr=" << keys.data_ptr());
  RECSTORE_LOG(
      0, "[DEBUG][op_torch] emb_read_torch: embedding_dim=" << embedding_dim);
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    RECSTORE_LOG(2, "[INFO] emb_read_torch: copying GPU keys to CPU");
    cpu_keys = keys.cpu();
  }
  if (cpu_keys.size(0) > 0) {
    auto cpu_keys_acc = cpu_keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_read_torch: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << cpu_keys_acc[i] << ", ";
    RECSTORE_LOG(0, oss.str());
  }
  RECSTORE_LOG(2,
               "[INFO] emb_read_torch called: keys shape="
                   << cpu_keys.sizes() << ", dtype=" << cpu_keys.dtype()
                   << ", embedding_dim=" << embedding_dim);
  TORCH_CHECK(cpu_keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(cpu_keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(cpu_keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = cpu_keys.size(0);
  if (num_keys == 0) {
    RECSTORE_LOG(3, "[DEBUG] emb_read_torch: num_keys==0, returning empty");
    return torch::empty(
        {0, embedding_dim}, cpu_keys.options().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

  auto values = torch::empty(
      {num_keys, embedding_dim}, keys.options().dtype(torch::kFloat32));
  torch::Tensor cpu_values = values;
  if (values.is_cuda()) {
    RECSTORE_LOG(
        2,
        "[INFO] emb_read_torch: copying GPU values to CPU for C++ operation");
    cpu_values = values.cpu();
  }
  TORCH_CHECK(cpu_values.is_contiguous(),
              "Internal error: Created values tensor is not contiguous");
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_read_torch: values shape="
                   << cpu_values.sizes() << ", dtype=" << cpu_values.dtype()
                   << ", data_ptr=" << cpu_values.data_ptr());
  if (cpu_values.size(0) > 0) {
    auto values_acc = cpu_values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_read_torch: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, cpu_values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, cpu_values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    RECSTORE_LOG(0, oss.str());
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: calling op->EmbRead");
  op->EmbRead(rec_keys, rec_values);
  RECSTORE_LOG(3, "[DEBUG] emb_read_torch: EmbRead done");

  if (values.is_cuda()) {
    RECSTORE_LOG(2, "[INFO] emb_read_torch: copying results back to GPU");
    values.copy_(cpu_values);
  }

  return values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse optimizer.");
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_write_torch: keys shape="
                   << keys.sizes() << ", dtype=" << keys.dtype()
                   << ", data_ptr=" << keys.data_ptr());
  RECSTORE_LOG(0,
               "[DEBUG][op_torch] emb_write_torch: values shape="
                   << values.sizes() << ", dtype=" << values.dtype()
                   << ", data_ptr=" << values.data_ptr());
  if (keys.size(0) > 0) {
    auto keys_acc = keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_write_torch: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << keys_acc[i] << ", ";
    RECSTORE_LOG(0, oss.str());
  }
  if (values.size(0) > 0) {
    auto values_acc = values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "[DEBUG][op_torch] emb_write_torch: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    RECSTORE_LOG(0, oss.str());
  }
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

  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    RECSTORE_LOG(2, "[INFO] emb_write_torch: copying GPU keys to CPU");
    cpu_keys = keys.cpu();
  }
  if (values.is_cuda()) {
    RECSTORE_LOG(2, "[INFO] emb_write_torch: copying GPU values to CPU");
    cpu_values = values.cpu();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

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
