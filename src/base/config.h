#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace base {

inline constexpr const char* kDefaultRecStoreConfigPath =
    "/app/RecStore/recstore_config.json";
inline constexpr const char* kRecStoreConfigFilename = "recstore_config.json";

namespace detail {

inline bool ConfigFileExists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

inline std::optional<std::filesystem::path> FindRecStoreConfigPath(
    const std::filesystem::path& default_path,
    std::filesystem::path start_dir) {
  if (ConfigFileExists(default_path)) {
    return default_path;
  }

  std::error_code ec;
  if (!start_dir.is_absolute()) {
    start_dir = std::filesystem::absolute(start_dir, ec);
    if (ec) {
      return std::nullopt;
    }
  }

  while (!start_dir.empty()) {
    const auto candidate = start_dir / kRecStoreConfigFilename;
    if (ConfigFileExists(candidate)) {
      return candidate;
    }

    const auto parent = start_dir.parent_path();
    if (parent == start_dir) {
      break;
    }
    start_dir = parent;
  }

  return std::nullopt;
}

inline std::filesystem::path ResolveRecStoreConfigPath(
    const std::filesystem::path& default_path,
    const std::filesystem::path& start_dir) {
  auto config_path = FindRecStoreConfigPath(default_path, start_dir);
  if (!config_path.has_value()) {
    throw std::runtime_error(
        "recstore_config.json not found at " + default_path.string() +
        " or in any parent directory of " + start_dir.string());
  }
  return *config_path;
}

} // namespace detail

inline std::optional<std::filesystem::path> FindRecStoreConfigPath() {
  return detail::FindRecStoreConfigPath(kDefaultRecStoreConfigPath,
                                        std::filesystem::current_path());
}

inline std::filesystem::path ResolveRecStoreConfigPath() {
  return detail::ResolveRecStoreConfigPath(kDefaultRecStoreConfigPath,
                                           std::filesystem::current_path());
}

} // namespace base
