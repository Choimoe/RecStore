#pragma once

#include <torch/extension.h>

namespace recstore {
namespace framework {

// Registers the base family of recstore_ops. Called from the single
// TORCH_LIBRARY(recstore_ops, ...) translation unit (op_torch.cc).
void RegisterBaseOps(torch::Library& m);

} // namespace framework
} // namespace recstore
