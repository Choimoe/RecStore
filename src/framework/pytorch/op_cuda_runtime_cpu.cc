#include "framework/pytorch/op_cuda_runtime.h"

// CPU-only fallback for builds without a CUDA toolkit: the stream
// synchronization is a no-op and page-locking always reports failure so the
// callers take their non-pinned path.

namespace recstore {
namespace framework {

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor) {
  (void)tensor;
}

bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes) {
  (void)ptr;
  (void)bytes;
  return false;
}

} // namespace framework
} // namespace recstore
