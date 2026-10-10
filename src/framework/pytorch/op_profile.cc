#include "framework/pytorch/op_profile.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

std::vector<double> get_last_local_lookup_flat_profile_torch() {
  return g_last_local_lookup_flat_profile;
}

std::vector<double> get_last_local_update_flat_profile_torch() {
  return g_last_local_update_flat_profile;
}

std::vector<double> get_last_gpu_cache_profile_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  const auto profile = gpu::GetLastGpuCacheProfile();
  return {
      profile.query_ms,
      profile.backend_lookup_ms,
      profile.fill_ms,
      profile.update_ms,
      profile.hit_count,
      profile.invalidate_ms,
      profile.request_count,
      profile.miss_count,
  };
#else
  return {};
#endif
}

} // namespace

void RegisterProfileOps(torch::Library& m) {
  m.def("get_last_local_lookup_flat_profile",
        get_last_local_lookup_flat_profile_torch);
  m.def("get_last_local_update_flat_profile",
        get_last_local_update_flat_profile_torch);
  m.def("get_last_gpu_cache_profile", get_last_gpu_cache_profile_torch);
}

} // namespace framework
} // namespace recstore
