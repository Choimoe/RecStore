#pragma once

// CUDA-runtime host helpers used by the recstore_ops translation units.
// Declarations stay free of CUDA includes so the .cc files do not need
// __has_include guards; the implementation lives in op_cuda_runtime.cu
// (or op_cuda_runtime_cpu.cc for CPU-only builds).

#include <cstddef>

#include <torch/extension.h>

namespace recstore {
namespace framework {

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor);

bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes);

} // namespace framework
} // namespace recstore
