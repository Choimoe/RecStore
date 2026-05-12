#pragma once

#include <memory>

#include "base/factory.h"
#include "storage/index/dram/extendible_hash.h"

class DramExtendibleHashIndex : public Index {
public:
  explicit DramExtendibleHashIndex(const BaseKVConfig& config)
      : Index(config), impl_(std::make_unique<ExtendibleHash>(config)) {}

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    impl_->Get(key, pointer, tid);
  }

  void Put(Key_t key, Value_t pointer, unsigned tid) override {
    impl_->Put(key, pointer, tid);
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    impl_->BatchGet(keys, pointers, tid);
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    impl_->BatchPut(keys, pointers, tid);
  }

  bool Delete(Key_t& key) override { return impl_->Delete(key); }
  double Utilization() override { return impl_->Utilization(); }
  size_t Capacity() override { return impl_->Capacity(); }

private:
  std::unique_ptr<ExtendibleHash> impl_;
};

FACTORY_REGISTER(Index, DRAM, DramExtendibleHashIndex, const BaseKVConfig&);
FACTORY_REGISTER(
    Index, DRAM_EXTENDIBLE_HASH, DramExtendibleHashIndex, const BaseKVConfig&);
