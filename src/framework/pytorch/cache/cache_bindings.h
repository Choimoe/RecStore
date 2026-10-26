#pragma once

#include <torch/library.h>

namespace recstore::framework {
void RegisterGpuCacheOps(torch::Library& m);
} // namespace recstore::framework
