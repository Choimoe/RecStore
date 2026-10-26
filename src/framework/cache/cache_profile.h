#pragma once

namespace recstore::framework::gpu {

// Compatibility counters: collection was removed by the external-release
// cleanup. Keep the existing API returning zeros.
struct GpuCacheProfile {
  double query_ms          = 0.0;
  double backend_lookup_ms = 0.0;
  double fill_ms           = 0.0;
  double update_ms         = 0.0;
  double hit_count         = 0.0;
  double invalidate_ms     = 0.0;
  double request_count     = 0.0;
  double miss_count        = 0.0;
};

inline GpuCacheProfile GetLastGpuCacheProfile() { return {}; }
inline void ResetLastGpuCacheProfile() {}
inline void AddGpuCacheBackendLookupMs(double /*ms*/) {}

} // namespace recstore::framework::gpu
