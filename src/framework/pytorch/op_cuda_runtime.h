#pragma once

// CUDA-runtime host helpers used by the recstore_ops translation units.
// Declarations stay free of CUDA includes so the .cc files do not need
// __has_include guards; the implementation lives in op_cuda_runtime.cu
// (or op_cuda_runtime_cpu.cc for CPU-only builds).

#include "framework/gpu/host_memory.h"
#include <cstddef>
#include <functional>

#include <torch/extension.h>

namespace recstore {
namespace framework {

void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor);
// Record on the caller's stream; the returned wait may run on a CPU worker.
std::function<void()>
RecordCudaCompletionForTensor(const torch::Tensor& tensor);

void RetainUntilCudaComplete(const torch::Tensor& tensor,
                             std::function<void()> release);

} // namespace framework
} // namespace recstore
