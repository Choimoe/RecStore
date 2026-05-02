#pragma once
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

using boost::coroutines2::coroutine;

// #define XMH_SIMPLE_MALLOC

struct BaseKVConfig {
  int num_threads_ = 0;
  json json_config_; // add your custom config in this field
};
/*
===============================================================
使用指南：如何配置 KV 引擎（engine）与内存管理（allocator）
===============================================================

一、总体原则
-----------
1) 你只需要配置 **index_type** 与 **value_type**，具体用哪个引擎由
   `base::ResolveEngine(cfg)` 自动推导，再交给工厂创建。
   - value_type = HYBRID        → KVEngineHybrid
   - value_type ∈ {DRAM, SSD}
       且 index_type = DRAM     → KVEngineExtendibleHash
       且 index_type = SSD      → KVEngineCCEH
2) 内存管理实现（allocator）推荐通过
   `json_config_["allocator_type"]` 选择：
   - "PERSIST_LOOP_SLAB"（默认）
   - "R2_SLAB"
   兼容旧字段 `value_memory_management`。

二、必须/可选配置字段
-------------------
必填（非 HYBRID）：
  - "DATA_DIR"    : 工作目录（每个实例建议唯一，兼容旧字段 "path"）
  - "index_type"  : "DRAM" | "SSD"
  - "value_type"  : "DRAM" | "SSD"
  - "VALUE_SIZE_BYTES" : 每条 value 字节数（兼容旧字段 "value_size"，例如 128）
  - "DRAM_SIZE"/"SSD_SIZE" : 对应 value 介质容量预算（字节，二选一）
    或显式给 "ENTRY_CAPACITY"（兼容旧字段 "capacity"）

必填（HYBRID）：
  - "DATA_DIR"    : 兼容旧字段 "path"
  - "index_type"  : "DRAM" | "SSD"
  - "value_type"  : "HYBRID"
  - "DRAM_SIZE"   : DRAM/SHM 侧的字节数预算（兼容旧字段 "shmcapacity"）
  - "SSD_SIZE"    : SSD 侧的字节数预算（兼容旧字段 "ssdcapacity"）

可选：
  - "allocator_type" : "PERSIST_LOOP_SLAB" | "R2_SLAB"
  - "value_layout"   : "FIXED" | "VARIABLE"（VARIABLE 仅 HYBRID 支持）
  默认分配器是 PERSIST_LOOP_SLAB

三、推荐创建流程（示例 C++）
---------------------------
BaseKVConfig cfg;
cfg.num_threads_ = 16;
cfg.json_config_ = {
  {"DATA_DIR", "/tmp/recstore"},
  {"index_type", "DRAM"},
  {"value_type", "SSD"},
  {"DRAM_SIZE", 128000000},
  {"VALUE_SIZE_BYTES", 128},
  {"allocator_type", "PERSIST_LOOP_SLAB"} // 或 "R2_SLAB"
};

// 由 (index_type, value_type) 推导 engine_type，并补齐/校验必需字段
auto r = base::ResolveEngine(cfg);

// 用工厂创建（确保相应引擎与分配器已在本进程中注册/链接）
std::unique_ptr<BaseKV> kv(
  base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(r.engine, r.cfg)
);
if (!kv) throw std::runtime_error("Create KV engine failed: " + r.engine);

四、常见组合示例
---------------
1) DRAM 索引 + SSD 值（KVEngineExtendibleHash）
   cfg.json_config_ = {
     {"DATA_DIR", "/data/eh"},
     {"index_type", "DRAM"},
     {"value_type", "SSD"},
     {"SSD_SIZE", 256000000},
     {"VALUE_SIZE_BYTES", 128},
     {"allocator_type", "PERSIST_LOOP_SLAB"} // 或 "R2_SLAB"
   };

2) SSD 索引 + SSD 值（KVEngineCCEH）
   cfg.json_config_ = {
     {"DATA_DIR", "/data/cceh"},
     {"index_type", "SSD"},
     {"value_type", "SSD"},
     {"SSD_SIZE", 256000000},
     {"VALUE_SIZE_BYTES", 128},
     {"allocator_type", "PERSIST_LOOP_SLAB"}
   };

3) HYBRID 值（KVEngineHybrid)
   cfg.json_config_ = {
     {"DATA_DIR", "/data/hybrid"},
     {"index_type", "DRAM"},
     {"value_type", "HYBRID"},
     {"DRAM_SIZE",  128ull * 1'000'000},  // DRAM 侧字节数
     {"SSD_SIZE",   256ull * 1'000'000},  // SSD  侧字节数
     {"allocator_type", "PERSIST_LOOP_SLAB"}
   };

六、依赖
- 使用前请先include"storage/kv_engine/engine_factory"（注册所有引擎）
- 使用前请先include"memory/memory_factory.h"（注册所有内存管理机制）

五、校验与排错
-------------
- 若你仍显式配置了 "engine_type"，它必须与 ResolveEngine
推导结果一致；不一致会抛异常。
- HYBRID 缺 "DRAM_SIZE"/"SSD_SIZE"（或旧字段）→ ResolveEngine 会抛
std::invalid_argument。
- 非 HYBRID 缺 "VALUE_SIZE_BYTES" 且无法由 DRAM_SIZE/SSD_SIZE 推导容量
  （也未显式给 ENTRY_CAPACITY）→ 同上抛异常。


（完）
*/

/*
另：可查阅
- KV 引擎实现 https://recstore.github.io/RecStore/storage/kv_engines/
- 内存管理 https://recstore.github.io/RecStore/storage/memory/
*/

class BaseKV {
public:
  virtual ~BaseKV() { std::cout << "exit BaseKV" << std::endl; }

  explicit BaseKV(const BaseKVConfig& config){};

  virtual void Util() {
    std::cout << "BaseKV Util: no impl" << std::endl;
    return;
  }
  virtual void Get(const uint64_t key, std::string& value, unsigned tid) = 0;
  virtual void
  Put(const uint64_t key, const std::string_view& value, unsigned tid) = 0;

  virtual void BatchPut(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual void BatchPut(coroutine<void>::push_type& sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  };

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) = 0;

  virtual void BatchGet(coroutine<void>::push_type& sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual bool ApplySgdUpdateFlat(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      uint8_t tag,
      unsigned tid) {
    return false;
  }

  virtual void DebugInfo() const {}

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) {
    LOG(FATAL) << "not implemented";
  };

  virtual void LoadFakeData(int64_t key_capacity, int value_size) {
    std::vector<uint64_t> keys;
    float* values = new float[value_size / sizeof(float) * key_capacity];
    keys.reserve(key_capacity);
    for (int64_t i = 0; i < key_capacity; i++) {
      keys.push_back(i);
    }
    this->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  };

  virtual void clear() {
    LOG(WARNING) << "clear() not fully implemented for this KV engine";
  };

protected:
};
