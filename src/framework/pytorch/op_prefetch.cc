#include "framework/pytorch/op_prefetch.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

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
  auto owned = std::make_shared<base::RecTensor>(
      std::vector<int64_t>{0, embedding_dim}, base::DataType::FLOAT32);
  op->GetPretchResult(static_cast<uint64_t>(prefetch_id), *owned);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const int64_t L = owned->dim() == 2 ? owned->shape(0) : 0;
  if (L == 0) {
    return torch::empty({0, embedding_dim}, options);
  }
  return torch::from_blob(
      owned->data(), {L, embedding_dim}, [owned](void*) {}, options);
}

} // namespace

void RegisterPrefetchOps(torch::Library& m) {
  m.def("emb_prefetch", emb_prefetch_torch);
  m.def("emb_wait_result", emb_wait_result_torch);
}

} // namespace framework
} // namespace recstore
