#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>

#include "memory/allocators/allocator_factory.h"
#include "base_kv.h"

namespace base {

struct EngineResolved {
  std::string engine;
  BaseKVConfig cfg;
};

inline std::string EngineSelectorUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

inline bool HasLegacyLocalConfig(const json& j) {
  return j.contains("index_type") || j.contains("value_type") ||
         j.contains("VALUE_SIZE_BYTES") || j.contains("ENTRY_CAPACITY") ||
         j.contains("DATA_DIR") || j.contains("DRAM_SIZE") ||
         j.contains("SSD_SIZE") || j.contains("shmcapacity") ||
         j.contains("ssdcapacity");
}

inline void CanonicalizeAlias(json& j, const char* lhs, const char* rhs) {
  const bool has_lhs = j.contains(lhs);
  const bool has_rhs = j.contains(rhs);
  if (has_lhs && has_rhs && j.at(lhs) != j.at(rhs)) {
    throw std::invalid_argument(
        std::string(lhs) + " conflicts with alias " + rhs);
  }
  if (!has_lhs && has_rhs) {
    j[lhs] = j.at(rhs);
  }
}

inline uint64_t ValueSizeHint(const json& j) {
  if (j.contains("value_size")) {
    return j.at("value_size").get<uint64_t>();
  }
  if (j.contains("value") &&
      j.at("value").contains("default_value_size_hint")) {
    return j.at("value").at("default_value_size_hint").get<uint64_t>();
  }
  return 0;
}

inline json MakeSsdAllocator(
    const json& j, const std::string& file_path, uint64_t capacity_bytes) {
  const int queue_depth = j.value("queue_size", j.value("queue_cnt", 512));
  return {{"type", "SSD_BUDDY"},
          {"capacity_bytes", capacity_bytes},
          {"min_block_size", 128},
          {"max_block_size", 4096},
          {"io",
           {{"type", j.value("io_backend_type", std::string("IOURING"))},
            {"file_path", file_path},
            {"queue_depth", queue_depth},
            {"base_offset_bytes", static_cast<uint64_t>(4096)}}}};
}

inline void NormalizeLegacyLocalConfig(BaseKVConfig& cfg) {
  auto& j = cfg.json_config_;
  CanonicalizeAlias(j, "path", "DATA_DIR");
  CanonicalizeAlias(j, "capacity", "ENTRY_CAPACITY");
  CanonicalizeAlias(j, "value_size", "VALUE_SIZE_BYTES");
  CanonicalizeAlias(j, "shmcapacity", "DRAM_SIZE");
  CanonicalizeAlias(j, "ssdcapacity", "SSD_SIZE");
  CanonicalizeAlias(j, "queue_cnt", "IO_QUEUE_COUNT");
  base::allocators::NormalizeAllocatorConfig(j);

  if (!j.contains("path")) {
    throw std::invalid_argument("path is required");
  }

  std::string index_type = EngineSelectorUpper(j.value("index_type", "DRAM"));
  std::string value_type = EngineSelectorUpper(j.value("value_type", ""));
  if (index_type == "DRAM/SSD") {
    index_type = "DRAM";
  }
  if (value_type.empty()) {
    throw std::invalid_argument("value_type is required");
  }

  const uint64_t value_size = ValueSizeHint(j);
  if (value_type != "HYBRID" && value_size == 0) {
    throw std::invalid_argument("value_size is required");
  }

  if (!j.contains("capacity") && value_size > 0) {
    uint64_t capacity_bytes = 0;
    if (value_type == "DRAM" && j.contains("DRAM_SIZE")) {
      capacity_bytes = j.at("DRAM_SIZE").get<uint64_t>();
    } else if (value_type == "SSD" && j.contains("SSD_SIZE")) {
      capacity_bytes = j.at("SSD_SIZE").get<uint64_t>();
    }
    if (capacity_bytes > 0) {
      j["capacity"] = std::max<uint64_t>(1, capacity_bytes / value_size);
    }
  }
  if (!j.contains("capacity")) {
    throw std::invalid_argument("capacity is required");
  }

  const std::string path  = j.at("path").get<std::string>();
  const uint64_t capacity = j.at("capacity").get<uint64_t>();
  const uint64_t fallback_bytes =
      std::max<uint64_t>(1, capacity) * std::max<uint64_t>(1, value_size);

  if (!j.contains("index")) {
    if (index_type == "DRAM") {
      j["index"] = {{"type", "DRAM_EXTENDIBLE_HASH"}};
    } else if (index_type == "SSD") {
      const int queue_depth = j.value("queue_size", j.value("queue_cnt", 512));
      j["index"]            = {
          {"type", "SSD_EXTENDIBLE_HASH"},
          {"io",
                      {{"type", j.value("io_backend_type", std::string("IOURING"))},
                       {"file_path", path + "/index_pages.db"},
                       {"queue_depth", queue_depth},
                       {"base_offset_bytes", static_cast<uint64_t>(0)}}}};
    } else {
      throw std::invalid_argument("index_type must be DRAM or SSD");
    }
  }

  if (!j.contains("value")) {
    j["value"] = json::object();
  }
  auto& value = j["value"];
  if (value_size > 0 && !value.contains("default_value_size_hint")) {
    value["default_value_size_hint"] = value_size;
  }

  if (value_type == "DRAM") {
    value["type"] = "DRAM_VALUE_STORE";
    if (!value.contains("dram_allocator")) {
      const uint64_t bytes    = j.value("DRAM_SIZE", fallback_bytes);
      value["dram_allocator"] = {
          {"type", j.at("allocator_type").get<std::string>()},
          {"capacity_bytes", bytes}};
    }
  } else if (value_type == "SSD") {
    value["type"] = "SSD_VALUE_STORE";
    if (!value.contains("ssd_allocator")) {
      const uint64_t bytes = j.value("SSD_SIZE", fallback_bytes);
      value["ssd_allocator"] =
          MakeSsdAllocator(j, path + "/value_pages.db", bytes);
    }
  } else if (value_type == "HYBRID") {
    value["type"]             = "TIERED_VALUE_STORE";
    const uint64_t dram_bytes = j.value("DRAM_SIZE", j.value("shmcapacity", 0));
    const uint64_t ssd_bytes  = j.value("SSD_SIZE", j.value("ssdcapacity", 0));
    if (dram_bytes == 0 || ssd_bytes == 0) {
      throw std::invalid_argument(
          "HYBRID requires shmcapacity and ssdcapacity");
    }
    if (!value.contains("dram_allocator")) {
      value["dram_allocator"] = {
          {"type", j.at("allocator_type").get<std::string>()},
          {"capacity_bytes", dram_bytes}};
    }
    if (!value.contains("ssd_allocator")) {
      value["ssd_allocator"] =
          MakeSsdAllocator(j, path + "/tiered_value_pages.db", ssd_bytes);
    }
    if (!value.contains("tiering")) {
      value["tiering"] = {{"cache_policy", j.value("cache_policy", "LRU")}};
    }
  } else {
    throw std::invalid_argument("value_type must be DRAM, SSD, or HYBRID");
  }
}

inline EngineResolved ResolveEngine(BaseKVConfig cfg) {
  auto& j = cfg.json_config_;

  if (j.contains("engine_type")) {
    const std::string engine = j.at("engine_type").get<std::string>();
    static const std::set<std::string> kExplicitEngines = {
        "KVEngine",
        "KVEngineFasterKV",
        "KVEngineHPSHashMap",
        "KVEngineHPSRocksDB"};
    if (!kExplicitEngines.count(engine)) {
      throw std::invalid_argument("unknown engine_type: " + engine);
    }
    if (engine != "KVEngine") {
      for (const char* k : {"path", "capacity"}) {
        if (!j.contains(k)) {
          throw std::invalid_argument(
              std::string(k) + " is required for " + engine);
        }
      }
      const bool has_value_size =
          j.contains("value_size") ||
          (j.contains("value") &&
           j.at("value").contains("default_value_size_hint"));
      if (!has_value_size) {
        throw std::invalid_argument("value_size is required for " + engine);
      }
      return EngineResolved{engine, std::move(cfg)};
    }
  }

  if (HasLegacyLocalConfig(j)) {
    NormalizeLegacyLocalConfig(cfg);
  }

  for (const char* k : {"path", "capacity", "index", "value"}) {
    if (!j.contains(k)) {
      throw std::invalid_argument(std::string(k) + " is required");
    }
  }

  for (const char* k : {"mode", "value_layout"}) {
    if (j.contains(k)) {
      throw std::invalid_argument(
          "legacy field '" + std::string(k) +
          "' not allowed; use nested index/value config");
    }
  }

  const auto& idx_j = j.at("index");
  const auto& val_j = j.at("value");
  if (!idx_j.contains("type")) {
    throw std::invalid_argument("index.type is required");
  }
  if (!val_j.contains("type")) {
    throw std::invalid_argument("value.type is required");
  }

  const std::string idx_type = idx_j.at("type").get<std::string>();
  const std::string val_type = val_j.at("type").get<std::string>();

  static const std::set<std::string> kDramIndex = {
      "DRAM_EXTENDIBLE_HASH", "DRAM_UNORDERED_MAP", "DRAM_PET_HASH"};
  static const std::set<std::string> kSsdIndex = {"SSD", "SSD_EXTENDIBLE_HASH"};
  static const std::set<std::string> kValTypes = {
      "DRAM_VALUE_STORE", "SSD_VALUE_STORE", "TIERED_VALUE_STORE"};

  if (!kDramIndex.count(idx_type) && !kSsdIndex.count(idx_type)) {
    throw std::invalid_argument("unknown index.type: " + idx_type);
  }
  if (!kValTypes.count(val_type)) {
    throw std::invalid_argument("unknown value.type: " + val_type);
  }
  if (kSsdIndex.count(idx_type) && val_type != "SSD_VALUE_STORE") {
    throw std::invalid_argument("SSD index only supports SSD_VALUE_STORE");
  }
  if (kSsdIndex.count(idx_type) && !idx_j.contains("io")) {
    throw std::invalid_argument("SSD index requires index.io config");
  }

  if (val_type == "DRAM_VALUE_STORE") {
    if (!val_j.contains("dram_allocator")) {
      throw std::invalid_argument(
          "DRAM_VALUE_STORE requires value.dram_allocator");
    }
    if (val_j.contains("ssd_allocator")) {
      throw std::invalid_argument(
          "DRAM_VALUE_STORE must not have ssd_allocator");
    }
  } else if (val_type == "SSD_VALUE_STORE") {
    if (!val_j.contains("ssd_allocator")) {
      throw std::invalid_argument(
          "SSD_VALUE_STORE requires value.ssd_allocator");
    }
    if (val_j.contains("dram_allocator")) {
      throw std::invalid_argument(
          "SSD_VALUE_STORE must not have dram_allocator");
    }
  } else if (val_type == "TIERED_VALUE_STORE") {
    if (!val_j.contains("dram_allocator") || !val_j.contains("ssd_allocator")) {
      throw std::invalid_argument(
          "TIERED_VALUE_STORE requires dram_allocator and ssd_allocator");
    }
  }

  return EngineResolved{"KVEngine", std::move(cfg)};
}

} // namespace base
