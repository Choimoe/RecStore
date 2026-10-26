#include "framework/pytorch/cache/gpu_cache_adapter.h"
#include "framework/cache/gpu_embedding_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace recstore::framework::torch_cache {
namespace {
gpu::CacheContext Context(const torch::Tensor& keys) {
  TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.dim() == 1 &&
                  keys.scalar_type() == torch::kInt64,
              "keys must be a contiguous 1-D CUDA int64 tensor");
  return {keys.get_device(),
          at::cuda::getCurrentCUDAStream(keys.get_device()).stream()};
}
gpu::CacheContext Rows(const torch::Tensor& keys, const torch::Tensor& values) {
  auto context = Context(keys);
  TORCH_CHECK(
      values.device() == keys.device() && values.is_contiguous() &&
          values.scalar_type() == torch::kFloat32 && values.dim() == 2 &&
          values.size(0) == keys.numel(),
      "values must be contiguous CUDA float32 rows on the keys' device");
  return context;
}
torch::Tensor Mask(const torch::Tensor& keys) {
  return torch::empty({keys.numel()}, keys.options().dtype(torch::kBool));
}
} // namespace

bool EnableGpuCache(int64_t capacity, int64_t dim) {
  return gpu::EnableGpuCache(capacity, dim, c10::cuda::current_device());
}
void DisableGpuCache() { gpu::DisableGpuCache(); }
void ClearGpuCache() { gpu::ClearGpuCache(); }
void ResetGpuCache() { gpu::ResetGpuCache(); }
bool IsGpuCacheEnabled() { return gpu::IsGpuCacheEnabled(); }
uint64_t GetGpuCacheGeneration() { return gpu::GetGpuCacheGeneration(); }
bool CanUseGpuCache(const torch::Tensor& keys, int64_t dim) {
  return keys.is_cuda() && keys.is_contiguous() && keys.dim() == 1 &&
         keys.scalar_type() == torch::kInt64 &&
         gpu::CanUseGpuCache(keys.get_device(), dim);
}

GpuCacheLookupResult QueryGpuCache(const torch::Tensor& keys, int64_t dim) {
  const auto context = Context(keys);
  TORCH_CHECK(dim > 0, "embedding dimension must be positive");
  auto values =
      torch::empty({keys.numel(), dim}, keys.options().dtype(torch::kFloat32));
  auto positions = torch::empty_like(keys);
  auto missing   = torch::empty_like(keys);
  auto count     = torch::empty({1}, keys.options());
  static_assert(sizeof(size_t) == sizeof(int64_t));
  const int64_t n = gpu::QueryGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      dim,
      {values.data_ptr<float>(),
       reinterpret_cast<uint64_t*>(positions.data_ptr<int64_t>()),
       missing.data_ptr<int64_t>(),
       reinterpret_cast<size_t*>(count.data_ptr<int64_t>())},
      context);
  return {
      values, missing.slice(0, 0, n).cpu(), positions.slice(0, 0, n).cpu(), n};
}
std::tuple<torch::Tensor, torch::Tensor>
LookupGpuCacheAssumingHits(const torch::Tensor& keys, int64_t dim) {
  const auto context = Context(keys);
  TORCH_CHECK(dim > 0, "embedding dimension must be positive");
  auto values =
      torch::empty({keys.numel(), dim}, keys.options().dtype(torch::kFloat32));
  auto miss = Mask(keys);
  gpu::LookupGpuCacheAssumingHits(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      dim,
      values.data_ptr<float>(),
      miss.data_ptr<bool>(),
      context);
  return {values, miss};
}
torch::Tensor ContainsGpuCache(const torch::Tensor& keys) {
  const auto context = Context(keys);
  auto resident      = Mask(keys);
  gpu::ContainsGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      resident.data_ptr<bool>(),
      context);
  return resident;
}
void FillGpuCache(const torch::Tensor& keys, const torch::Tensor& values) {
  const auto context = Rows(keys, values);
  gpu::FillGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      values.data_ptr<float>(),
      values.size(1),
      context);
}
torch::Tensor
FillGpuCacheNoEvict(const torch::Tensor& keys, const torch::Tensor& values) {
  const auto context = Rows(keys, values);
  auto inserted      = Mask(keys);
  gpu::FillGpuCacheNoEvict(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      values.data_ptr<float>(),
      values.size(1),
      inserted.data_ptr<bool>(),
      context);
  return inserted;
}
void UpdateGpuCache(const torch::Tensor& keys, const torch::Tensor& values) {
  const auto context = Rows(keys, values);
  gpu::UpdateGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      values.data_ptr<float>(),
      values.size(1),
      context);
}
void InvalidateGpuCache(const torch::Tensor& keys) {
  const auto context = Context(keys);
  gpu::InvalidateGpuCache(keys.data_ptr<int64_t>(), keys.numel(), context);
}
torch::Tensor InvalidateGpuCacheWithMask(const torch::Tensor& keys) {
  const auto context = Context(keys);
  auto removed       = Mask(keys);
  gpu::InvalidateGpuCacheWithMask(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      removed.data_ptr<bool>(),
      context);
  return removed;
}
bool ApplySgdUpdateGpuCache(
    const torch::Tensor& keys, const torch::Tensor& grads, double lr) {
  const auto context = Rows(keys, grads);
  return gpu::ApplySgdUpdateGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      grads.data_ptr<float>(),
      grads.size(1),
      lr,
      context);
}
void ApplySgdUpdateBestEffortGpuCache(
    const torch::Tensor& keys, const torch::Tensor& grads, double lr) {
  const auto context = Rows(keys, grads);
  gpu::ApplySgdUpdateBestEffortGpuCache(
      keys.data_ptr<int64_t>(),
      keys.numel(),
      grads.data_ptr<float>(),
      grads.size(1),
      lr,
      context);
}
void ScatterMissValues(torch::Tensor* out,
                       const torch::Tensor& positions_cpu,
                       const torch::Tensor& values) {
  if (!values.numel())
    return;
  TORCH_CHECK(out && out->is_cuda() && out->is_contiguous(),
              "output must be contiguous CUDA");
  TORCH_CHECK(values.device() == out->device() && values.is_contiguous(),
              "miss values must be contiguous on the output device");
  TORCH_CHECK(positions_cpu.device().is_cpu() &&
                  positions_cpu.scalar_type() == torch::kInt64,
              "miss positions must be CPU int64");
  out->index_copy_(0, positions_cpu.to(out->device(), true), values);
}
} // namespace recstore::framework::torch_cache
