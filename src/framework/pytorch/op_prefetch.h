#pragma once

#include <torch/extension.h>

namespace recstore {
namespace framework {

// Registers the prefetch family of recstore_ops. Called from the single
// TORCH_LIBRARY(recstore_ops, ...) translation unit (op_torch.cc).
void RegisterPrefetchOps(torch::Library& m);

} // namespace framework
} // namespace recstore
