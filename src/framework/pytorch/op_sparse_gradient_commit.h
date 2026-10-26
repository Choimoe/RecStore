#pragma once

#include <torch/library.h>

namespace recstore::framework {
void RegisterSparseGradientCommitOps(torch::Library& m);
} // namespace recstore::framework
