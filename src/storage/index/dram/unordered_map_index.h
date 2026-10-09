#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "base/factory.h"
#include "storage/index/index.h"

class DramUnorderedMapIndex : public Index {
public:
  explicit DramUnorderedMapIndex(const BaseKVConfig& config) : Index(config) {}

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    (void)tid;
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = map_.find(key);
    pointer = (it == map_.end()) ? NONE : it->second;
  }

  Value_t Put(Key_t key, Value_t pointer, unsigned tid) override {
    (void)tid;
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it            = map_.find(key);
    Value_t old_handle = (it == map_.end()) ? kValueHandleNone : it->second;
    map_[key]          = pointer;
    return old_handle;
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    for (int i = 0; i < keys.Size(); ++i) {
      Get(keys[i], pointers[i], tid);
    }
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    for (int i = 0; i < keys.Size(); ++i) {
      Put(keys[i], pointers[i], tid);
    }
  }

  bool Delete(Key_t& key) override {
    std::unique_lock<std::shared_mutex> lock(mu_);
    return map_.erase(key) > 0;
  }

  size_t Capacity() override { return map_.size(); }

private:
  mutable std::shared_mutex mu_;
  std::unordered_map<Key_t, Value_t> map_;
};

FACTORY_REGISTER(
    Index, DRAM_UNORDERED_MAP, DramUnorderedMapIndex, const BaseKVConfig&);
