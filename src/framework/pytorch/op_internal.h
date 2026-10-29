#pragma once

// Tensor/device conversion helpers shared by the PyTorch op adapters.

#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/tensor.h"
#include "framework/op.h"
#include "framework/common/op_profile.h"
#include "framework/cache/cache_profile.h"
#include "framework/pytorch/op_cuda_runtime.h"
#include "ps/local_shm/local_shm_client.h"

#include <glog/logging.h>

#ifdef RECSTORE_ENABLE_GPU_CACHE
#  include "framework/pytorch/cache/gpu_cache_adapter.h"
#endif

namespace recstore {
namespace framework {

// --- Shared predicates and tensor helpers ----------------------------------

inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

inline torch::TensorOptions PinnedCpuOptions(torch::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCPU)
      .dtype(dtype)
      .pinned_memory(true);
}

inline torch::Tensor StageCudaTensorToPinnedCpu(const torch::Tensor& tensor,
                                                torch::ScalarType dtype) {
  auto cpu_tensor = torch::empty(tensor.sizes(), PinnedCpuOptions(dtype));
  cpu_tensor.copy_(tensor.to(dtype), /*non_blocking=*/false);
  return cpu_tensor;
}

inline torch::Tensor
StageCudaTensorToPinnedCpuAsyncNoCast(const torch::Tensor& tensor) {
  auto cpu_tensor =
      torch::empty(tensor.sizes(), PinnedCpuOptions(tensor.scalar_type()));
  cpu_tensor.copy_(tensor, /*non_blocking=*/true);
  return cpu_tensor;
}

inline torch::Tensor ToCpuTensor(const torch::Tensor& tensor) {
  return tensor.is_cuda() ? tensor.cpu() : tensor;
}

inline std::shared_ptr<KVClientOp> GetConcreteKVClientOp() {
  auto op    = GetKVClientOp();
  auto kv_op = std::dynamic_pointer_cast<KVClientOp>(op);
  TORCH_CHECK(kv_op != nullptr, "storage backend is not KVClientOp");
  return kv_op;
}

// Tensor adapters for the framework-independent op/cache runtime.
void CheckKeys(const torch::Tensor& keys, bool contiguous = true);
void CheckRows(const torch::Tensor& keys,
               const torch::Tensor& values,
               bool contiguous = true);
std::pair<torch::Tensor, torch::Tensor>
CudaCacheRows(const torch::Tensor& keys, const torch::Tensor& values);
CacheUpdateActions
CacheActions(const torch::Tensor& keys = {}, int64_t dim = 0);
std::tuple<torch::Tensor, torch::Tensor> CachedReadTorch(
    const torch::Tensor& keys,
    int64_t dim,
    bool local    = false,
    bool no_evict = false);

} // namespace framework
} // namespace recstore
