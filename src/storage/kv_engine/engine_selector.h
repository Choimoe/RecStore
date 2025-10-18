// engine_selector.h
#pragma once
#include <algorithm>
#include <stdexcept>
#include "base_kv.h"

namespace base {

inline std::string Upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

struct EngineResolved {
  std::string engine;   // "KVEngineExtendibleHash" / "KVEngineCCEH" / "KVEngineHybrid"
  BaseKVConfig cfg;     // 归一化后的配置（写回 engine_type、补默认/剔无关）
};

inline EngineResolved ResolveEngine(BaseKVConfig cfg) {
  auto& j = cfg.json_config_;

  std::string idx = Upper(j.value("index_type", "DRAM"));   // 缺省按 DRAM 索引
  std::string val = Upper(j.value("value_type", ""));       // 必填：DRAM / SSD / HYBRID
  if (val.empty())
    throw std::invalid_argument("value_type is required (DRAM/SSD/HYBRID)");

  std::string engine;
  if (val == "HYBRID") {
    engine = "KVEngineHybrid";
    // Hybrid 需要 shmcapacity / ssdcapacity
    if (!j.contains("shmcapacity") || !j.contains("ssdcapacity")) {
      throw std::invalid_argument("Hybrid requires shmcapacity and ssdcapacity");
    }
    // 纯 Hybrid 不需要 value_size/capacity
  } else if (val == "DRAM" || val == "SSD") {
    if (idx == "DRAM") engine = "KVEngineExtendibleHash";
    else if (idx == "SSD") engine = "KVEngineCCEH";
    else throw std::invalid_argument("index_type must be DRAM or SSD");

    // 非 Hybrid 需要 capacity / value_size
    if (!j.contains("capacity") || !j.contains("value_size")) {
      throw std::invalid_argument("Non-HYBRID requires capacity and value_size");
    }

    j.erase("shmcapacity");
    j.erase("ssdcapacity");
  } else {
    throw std::invalid_argument("value_type must be DRAM/SSD/HYBRID");
  }

  // 若用户显式写了 engine_type，做一致性校验
  if (j.contains("engine_type")) {
    if (Upper(j["engine_type"].get<std::string>()) != Upper(engine)) {
      throw std::invalid_argument("engine_type conflicts with (index_type,value_type) deduction");
    }
  }
  j["engine_type"] = engine;  // 回写，保持向后兼容

  return EngineResolved{engine, std::move(cfg)};
}

} // namespace base
