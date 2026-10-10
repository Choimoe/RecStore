#include "framework/pytorch/op_cuda_runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <glog/logging.h>

namespace recstore {
namespace framework {

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor) {
  if (!tensor.is_cuda()) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(tensor.device());
  C10_CUDA_CHECK(
      cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
}

bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return false;
  }
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const std::size_t page_bytes = static_cast<std::size_t>(page_size);
  const uintptr_t raw_begin    = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t raw_end      = raw_begin + bytes;
  const uintptr_t page_begin =
      raw_begin & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const uintptr_t page_end =
      (raw_end + page_bytes - 1U) & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const std::size_t required_bytes =
      static_cast<std::size_t>(page_end - page_begin);

  static std::mutex mu;
  static std::unordered_map<uintptr_t, std::size_t> registered_bytes_by_base;
  std::lock_guard<std::mutex> guard(mu);
  const std::size_t existing_bytes = registered_bytes_by_base[page_begin];
  if (existing_bytes >= required_bytes) {
    return true;
  }

  void* register_ptr = reinterpret_cast<void*>(page_begin + existing_bytes);
  const std::size_t register_bytes = required_bytes - existing_bytes;
  const cudaError_t err =
      cudaHostRegister(register_ptr, register_bytes, cudaHostRegisterPortable);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    LOG(WARNING) << "cudaHostRegister failed for local_shm payload: "
                 << cudaGetErrorString(err)
                 << " base=" << reinterpret_cast<void*>(page_begin)
                 << " bytes=" << required_bytes;
    return false;
  }
  registered_bytes_by_base[page_begin] = required_bytes;
  return true;
}

} // namespace framework
} // namespace recstore
