#include "framework/pytorch/op_cuda_runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace recstore {
namespace framework {

std::function<void()>
RecordCudaCompletionForTensor(const torch::Tensor& tensor) {
  if (!tensor.is_cuda())
    return [] {};
  c10::cuda::CUDAGuard guard(tensor.device());
  auto event = std::make_shared<at::cuda::CUDAEvent>();
  event->record(at::cuda::getCurrentCUDAStream());
  return [event] { event->synchronize(); };
}

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor) {
  if (!tensor.is_cuda()) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(tensor.device());
  C10_CUDA_CHECK(
      cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
}

void RetainUntilCudaComplete(const torch::Tensor& tensor,
                             std::function<void()> release) {
  c10::cuda::CUDAGuard guard(tensor.device());
  RetainUntilStreamComplete(
      at::cuda::getCurrentCUDAStream().stream(), std::move(release));
}

} // namespace framework
} // namespace recstore
