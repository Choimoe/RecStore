#include "framework/pytorch/op_cuda_runtime.h"
#include <torch/library.h>

// CPU-only fallback for builds without a CUDA toolkit: the stream
// synchronization is a no-op and page-locking always reports failure so the
// callers take their non-pinned path.

namespace recstore {
namespace framework {

// Leave the optional integer CUDA operator absent in CPU-only builds.
void RegisterGPUTicketOps(torch::Library&) {}

std::function<void()>
RecordCudaCompletionForTensor(const torch::Tensor& tensor) {
  TORCH_CHECK(!tensor.is_cuda(), "CUDA staging requires a CUDA build");
  return [] {};
}

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor) {
  (void)tensor;
}

void RetainUntilCudaComplete(const torch::Tensor&,
                             std::function<void()> release) {
  release();
}

} // namespace framework
} // namespace recstore
