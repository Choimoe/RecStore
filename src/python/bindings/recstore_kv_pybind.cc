#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "recstore_kv_bindings.h"

// The engine registers two primitive gflags. Hidden no-op constructors keep
// the extension preloadable in processes that already embed another gflags.
extern "C" {
__attribute__((visibility("hidden"))) void
_ZN6google14FlagRegistererC1IiEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, int*, int*) {}
__attribute__((visibility("hidden"))) void
_ZN6google14FlagRegistererC1IdEEPKcS3_S3_PT_S5_(
    const char*, const char*, const char*, double*, double*) {}
}

namespace {

// This extension must be present at process startup on glibc hosts where the
// engine's static TLS exceeds the late-dlopen surplus. Remove only this module
// from LD_PRELOAD as soon as the loader has mapped it so startup hooks and
// helper processes do not try to load a Python extension into non-Python
// executables. Other preloads (for example a filesystem client) are preserved.
__attribute__((constructor)) void ClearRecStoreKvFromLdPreload() {
  const char* raw_preload = std::getenv("LD_PRELOAD");
  if (raw_preload == nullptr || *raw_preload == '\0') {
    return;
  }

  std::string normalized(raw_preload);
  std::replace(normalized.begin(), normalized.end(), ' ', ':');
  std::stringstream entries(normalized);
  std::string entry;
  std::vector<std::string> remaining;
  while (std::getline(entries, entry, ':')) {
    if (!entry.empty() && std::filesystem::path(entry).filename().string().find(
                              "_recstore_kv") == std::string::npos) {
      remaining.push_back(std::move(entry));
    }
  }

  if (remaining.empty()) {
    unsetenv("LD_PRELOAD");
    return;
  }
  std::string updated = remaining.front();
  for (size_t i = 1; i < remaining.size(); ++i) {
    updated.append(":").append(remaining[i]);
  }
  setenv("LD_PRELOAD", updated.c_str(), 1);
}

} // namespace

PYBIND11_MODULE(_recstore_kv, module) {
  module.attr("RECSTORE_VERSION")    = RECSTORE_VERSION;
  module.attr("BINDING_API_VERSION") = RECSTORE_KV_API_VERSION;
  module.attr("BINDING_ABI_VERSION") = RECSTORE_KV_ABI_VERSION;

  recstore::python::BindLocalKV(module);
  recstore::python::BindLocalIndexedKV(module);
}
