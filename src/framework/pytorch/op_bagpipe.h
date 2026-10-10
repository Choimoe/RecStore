#pragma once

#include <torch/extension.h>

namespace recstore {
namespace framework {

// Registers the bagpipe family of recstore_ops. Called from the single
// TORCH_LIBRARY(recstore_ops, ...) translation unit (op_torch.cc).
void RegisterBagpipeOps(torch::Library& m);

} // namespace framework
} // namespace recstore
