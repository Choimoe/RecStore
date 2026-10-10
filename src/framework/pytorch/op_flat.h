#pragma once

#include <torch/extension.h>

namespace recstore {
namespace framework {

// Registers the flat family of recstore_ops. Called from the single
// TORCH_LIBRARY(recstore_ops, ...) translation unit (op_torch.cc).
void RegisterFlatOps(torch::Library& m);

} // namespace framework
} // namespace recstore
