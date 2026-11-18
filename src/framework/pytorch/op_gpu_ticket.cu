#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
__global__ void InitializeOrdinals(int64_t n, int64_t* ordinal) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(gridDim.x) * blockDim.x) {
    ordinal[i] = i;
  }
}

__global__ void MarkHeads(int64_t n, const int64_t* sorted, int32_t* flags) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(gridDim.x) * blockDim.x) {
    flags[i] = i == 0 || sorted[i] != sorted[i - 1];
  }
}

__global__ void ScatterPlan(int64_t n, const int64_t* sorted,
                            const int64_t* permutation, const int32_t* groups,
                            int64_t* packed, int64_t* inverse, int64_t* offsets) {
  // The preceding scan completed on the same stream before this launch.
  const int64_t unique_count = groups[n - 1];
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t group = int64_t(groups[i]) - 1;
    inverse[permutation[i]] = group;
    if (i == 0 || sorted[i] != sorted[i - 1]) {
      // Head destinations are [0,U); tail initialization is [U,N).
      // Exactly one head writes each group; permutation is a bijection.
      packed[group + 1] = sorted[i];
      offsets[group] = i;
    }
    if (i >= unique_count) {
      packed[i + 1] = 0;
      offsets[i] = n;
    }
    if (i == 0) {
      packed[0] = unique_count;
      offsets[n] = n;
    }
  }
}

using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
Outputs GPUTicketFixed(const at::Tensor& ids) {
  TORCH_CHECK(ids.is_cuda() && ids.layout() == at::kStrided &&
              ids.scalar_type() == at::kLong && ids.dim() == 1,
              "ticket_fixed requires a one-dimensional strided CUDA int64 tensor");
  const int64_t n = ids.numel();
  // The installed CUB 2.5 DeviceScan overload accepts int num_items.
  TORCH_CHECK(n <= std::numeric_limits<int>::max(), "ticket_fixed N exceeds CUB scan limit");
  c10::cuda::CUDAGuard device_guard(ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream(ids.get_device());
  // The caller must establish producer -> current-stream ordering. This records
  // allocator lifetime only; it is not a cross-stream readiness barrier.
  c10::cuda::CUDACachingAllocator::recordStream(ids.storage().data_ptr(), stream);
  const auto options = ids.options();
  auto inverse = at::empty({n}, options);
  auto permutation = at::empty({n}, options);
  if (n == 0) {
    return {at::zeros({1}, options), inverse, permutation, at::zeros({1}, options)};
  }
  auto packed = at::empty({n + 1}, options);
  auto input = ids.contiguous();
  auto offsets = at::empty({n + 1}, options);
  auto sorted = at::empty({n}, options);
  auto ordinal = at::empty({n}, options);
  auto flags = at::empty({n}, options.dtype(at::kInt));
  auto groups = at::empty({n}, options.dtype(at::kInt));
  size_t sort_bytes = 0, scan_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, sort_bytes, input.data_ptr<int64_t>(), sorted.data_ptr<int64_t>(),
      ordinal.data_ptr<int64_t>(), permutation.data_ptr<int64_t>(),
      static_cast<int>(n), 0, 64, stream.stream()));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      nullptr, scan_bytes, flags.data_ptr<int32_t>(), groups.data_ptr<int32_t>(),
      static_cast<int>(n), stream.stream()));
  const size_t workspace_bytes = std::max<size_t>(1, std::max(sort_bytes, scan_bytes));
  TORCH_CHECK(workspace_bytes <= size_t(std::numeric_limits<int64_t>::max()),
              "ticket_fixed workspace exceeds Tensor size limit");
  auto workspace = at::empty({static_cast<int64_t>(workspace_bytes)}, options.dtype(at::kByte));
  const int blocks = static_cast<int>(std::min<int64_t>((n + 256) / 256, 65535));
  InitializeOrdinals<<<blocks, 256, 0, stream.stream()>>>(n, ordinal.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      workspace.data_ptr(), sort_bytes, input.data_ptr<int64_t>(), sorted.data_ptr<int64_t>(),
      ordinal.data_ptr<int64_t>(), permutation.data_ptr<int64_t>(),
      static_cast<int>(n), 0, 64, stream.stream()));
  MarkHeads<<<blocks, 256, 0, stream.stream()>>>(
      n, sorted.data_ptr<int64_t>(), flags.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      workspace.data_ptr(), scan_bytes, flags.data_ptr<int32_t>(), groups.data_ptr<int32_t>(),
      static_cast<int>(n), stream.stream()));
  ScatterPlan<<<blocks, 256, 0, stream.stream()>>>(
      n, sorted.data_ptr<int64_t>(), permutation.data_ptr<int64_t>(),
      groups.data_ptr<int32_t>(), packed.data_ptr<int64_t>(), inverse.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Temporaries were allocated and consumed on this stream. PyTorch's caching
  // allocator preserves stream order when their local Tensor owners are released.
  return {packed, inverse, permutation, offsets};
}
} // namespace

namespace recstore {
namespace framework {
void RegisterGPUTicketOps(torch::Library& m) {
  m.def("gpu_ticket_fixed(Tensor ids) -> (Tensor, Tensor, Tensor, Tensor)", &GPUTicketFixed);
}
} // namespace framework
} // namespace recstore
