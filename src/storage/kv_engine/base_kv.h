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
KVEngine uses the nested configuration format:
{
  "capacity": 1000000,
  "index": {"type": "DRAM_EXTENDIBLE_HASH"},
  "value": {
    "type": "DRAM_VALUE_STORE",
    "path": "/tmp/recstore/value",
    "default_value_size_hint": 128,
    "dram_allocator": {
      "type": "PERSIST_LOOP_SLAB",
      "capacity_bytes": 128000000
    }
  }
}

ResolveEngine returns "KVEngine" for every valid configuration. Legacy fields
such as top-level path/index_type/value_type and nested file_path are rejected.
SSD index uses index.path, single-layer value stores use value.path, and TIERED
value stores use value.dram_allocator.path/value.ssd_allocator.path.
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
  virtual bool Exists(const uint64_t key, unsigned tid) {
    std::string value;
    Get(key, value, tid);
    return !value.empty();
  }

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
