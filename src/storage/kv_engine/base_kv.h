#pragma once
#include <boost/coroutine2/all.hpp>
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

using boost::coroutines2::coroutine;

// #define XMH_SIMPLE_MALLOC

struct BaseKVConfig {
  int num_threads_ = 0;
  json json_config_;  // add your custom config in this field
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
2) 内存管理实现（allocator）通过
   `json_config_["value_memory_management"]` 选择：
   - "PersistLoopShmMalloc"    （默认）
   - "R2ShmMalloc"

二、必须/可选配置字段
-------------------
必填（非 HYBRID）：
  - "path"        : 工作目录（每个实例建议唯一）
  - "index_type"  : "DRAM" | "SSD"
  - "value_type"  : "DRAM" | "SSD"
  - "capacity"    : 预估条目数（用于预分配/空间估算）
  - "value_size"  : 每条 value 的字节数（例如 128）

必填（HYBRID）（等待value.h同步，暂时不支持）：
  - "path"
  - "index_type"  : "DRAM" | "SSD"
  - "value_type"  : "HYBRID"
  - "shmcapacity" : DRAM/SHM 侧的字节数预算
  - "ssdcapacity" : SSD 侧的字节数预算

可选：
  - "value_memory_management" : "PersistLoopShmMalloc" | "R2ShmMalloc"
  默认是PersistLoopShmMalloc

三、推荐创建流程（示例 C++）
---------------------------
BaseKVConfig cfg;
cfg.num_threads_ = 16;
cfg.json_config_ = {
  {"path", "/tmp/recstore"},
  {"index_type", "DRAM"},
  {"value_type", "SSD"},
  {"capacity", 1000000},
  {"value_size", 128},
  {"value_memory_management", "PersistLoopShmMalloc"} // 或 "R2ShmMalloc"
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
     {"path", "/data/eh"},
     {"index_type", "DRAM"},
     {"value_type", "SSD"},
     {"capacity", 2'000'000},
     {"value_size", 128},
     {"value_memory_management", "PersistLoopShmMalloc"} // 或 "R2ShmMalloc"
   };

2) SSD 索引 + SSD 值（KVEngineCCEH）
   cfg.json_config_ = {
     {"path", "/data/cceh"},
     {"index_type", "SSD"},
     {"value_type", "SSD"},
     {"capacity", 2'000'000},
     {"value_size", 128},
     {"value_memory_management", "PersistLoopShmMalloc"}
   };

3) HYBRID 值（KVEngineHybrid)
   cfg.json_config_ = {
     {"path", "/data/hybrid"},
     {"index_type", "DRAM/SSD"},               
     {"value_type", "HYBRID"},
     {"shmcapacity",  128ull * 1'000'000},  // DRAM 侧字节数
     {"ssdcapacity",  256ull * 1'000'000},  // SSD  侧字节数
     {"value_memory_management", "PersistLoopShmMalloc"}
   };

六、依赖
- 使用前请先include"storage/kv_engine/engine_factory"（注册所有引擎）
- 使用前请先include"memory/memory_factory.h"（注册所有内存管理机制）

五、校验与排错
-------------
- 若你仍显式配置了 "engine_type"，它必须与 ResolveEngine 推导结果一致；不一致会抛异常。
- HYBRID 缺 "shmcapacity"/"ssdcapacity" → ResolveEngine 会抛 std::invalid_argument。
- 非 HYBRID 缺 "capacity"/"value_size" → 同上抛异常。


（完）
*/

class BaseKV {
 public:
  virtual ~BaseKV() { std::cout << "exit BaseKV" << std::endl; }

  explicit BaseKV(const BaseKVConfig &config){};

  virtual void Util() {
    std::cout << "BaseKV Util: no impl" << std::endl;
    return;
  }
  virtual void Get(const uint64_t key, std::string &value, unsigned tid) = 0;
  virtual void Put(const uint64_t key, const std::string_view &value,
                   unsigned tid) = 0;

  virtual void BatchPut(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  };

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) = 0;

  virtual void BatchGet(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual void DebugInfo() const {}

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) {
    LOG(FATAL) << "not implemented";
  };

  virtual void LoadFakeData(int64_t key_capacity, int value_size) {
    std::vector<uint64_t> keys;
    float *values = new float[value_size / sizeof(float) * key_capacity];
    keys.reserve(key_capacity);
    for (int64_t i = 0; i < key_capacity; i++) {
      keys.push_back(i);
    }
    this->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  };

  virtual void clear() { LOG(FATAL) << "not implemented"; };

 protected:
};
