#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

#include "base/factory.h"
#include "storage/index/index.h"
#include "storage/nvm/pet_kv/pet_hash.h"

class DramPetHashIndex : public Index {
public:
  explicit DramPetHashIndex(const BaseKVConfig& config) : Index(config) {
    const uint64_t capacity =
        config.json_config_.at("capacity").get<uint64_t>();
    const size_t bytes =
        base::PetHash<Key_t, Value_t, false>::MemorySize(capacity);
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, bytes) != 0 || mem == nullptr) {
      throw std::bad_alloc();
    }
    impl_ = new (mem) base::PetHash<Key_t, Value_t, false>();
    impl_->Initialize(capacity);
  }

  ~DramPetHashIndex() override {
    if (impl_ != nullptr) {
      impl_->~PetHash<Key_t, Value_t, false>();
      std::free(impl_);
      impl_ = nullptr;
    }
  }

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    (void)tid;
    auto [value, exists] = impl_->Get(key);
    pointer              = exists ? value : NONE;
  }

  Value_t Put(Key_t key, Value_t pointer, unsigned tid) override {
    (void)tid;
    Value_t old_handle = kValueHandleNone;
    impl_->Set(key, pointer, nullptr, false, &old_handle);
    return old_handle;
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    (void)tid;
    for (int i = 0; i < keys.Size(); ++i) {
      if (i + 1 < keys.Size()) {
        impl_->HintPrefetch(keys[i + 1]);
      }
      auto [value, exists] = impl_->Get(keys[i]);
      pointers[i]          = exists ? value : NONE;
    }
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    for (int i = 0; i < keys.Size(); ++i) {
      Put(keys[i], pointers[i], tid);
    }
  }

  bool Delete(Key_t& key) override { return impl_->Delete(key); }
  size_t Capacity() override { return impl_->Capacity(); }

private:
  base::PetHash<Key_t, Value_t, false>* impl_ = nullptr;
};

FACTORY_REGISTER(Index, DRAM_PET_HASH, DramPetHashIndex, const BaseKVConfig&);

// PetHash variant with two opt-in knobs wired from the index config:
//   index.max_load_factor  (float, default 0.5)
//   index.prefetch_depth   (int,   default 1)
// Probing stays chunk-local and BatchGet prefetches `prefetch_depth` keys
// ahead instead of only the next key.
class DramPetHashLocalIndex : public Index {
public:
  using Impl = base::PetHash<Key_t, Value_t, false>;

  explicit DramPetHashLocalIndex(const BaseKVConfig& config) : Index(config) {
    const auto& json        = config.json_config_;
    const auto& index       = json.at("index");
    const uint64_t capacity = json.at("capacity").get<uint64_t>();

    const double load_factor = index.value("max_load_factor", 0.5);
    if (!(load_factor > 0.0) || load_factor > 1.0) {
      throw std::invalid_argument(
          "DramPetHashLocalIndex requires 0 < index.max_load_factor <= 1");
    }
    const long load_factor_pct = std::lround(load_factor * 100.0);
    if (load_factor_pct < 1) {
      throw std::invalid_argument(
          "DramPetHashLocalIndex requires index.max_load_factor >= 0.01");
    }
    options_.max_load_factor_pct = static_cast<uint32_t>(load_factor_pct);
    options_.chunk_local_probe   = true;

    const int prefetch_depth = index.value("prefetch_depth", 1);
    if (prefetch_depth < 0) {
      throw std::invalid_argument(
          "DramPetHashLocalIndex requires index.prefetch_depth >= 0");
    }
    prefetch_depth_ = static_cast<size_t>(prefetch_depth);

    const size_t bytes = Impl::MemorySize(capacity, false, options_);
    void* mem          = nullptr;
    if (posix_memalign(&mem, 64, bytes) != 0 || mem == nullptr) {
      throw std::bad_alloc();
    }
    impl_ = new (mem) Impl();
    impl_->Initialize(capacity, false, options_);
  }

  ~DramPetHashLocalIndex() override {
    if (impl_ != nullptr) {
      impl_->~Impl();
      std::free(impl_);
      impl_ = nullptr;
    }
  }

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    (void)tid;
    auto [value, exists] = impl_->Get(key);
    pointer              = exists ? value : NONE;
  }

  Value_t Put(Key_t key, Value_t pointer, unsigned tid) override {
    (void)tid;
    Value_t old_handle = kValueHandleNone;
    impl_->Set(key, pointer, nullptr, false, &old_handle);
    return old_handle;
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    (void)tid;
    for (int i = 0; i < keys.Size(); ++i) {
      const int lookahead = i + static_cast<int>(prefetch_depth_);
      if (lookahead < keys.Size()) {
        impl_->HintPrefetch(keys[lookahead]);
      }
      auto [value, exists] = impl_->Get(keys[i]);
      pointers[i]          = exists ? value : NONE;
    }
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    for (int i = 0; i < keys.Size(); ++i) {
      Put(keys[i], pointers[i], tid);
    }
  }

  bool Delete(Key_t& key) override { return impl_->Delete(key); }
  size_t Capacity() override { return impl_->Capacity(); }

private:
  Impl::Options options_;
  size_t prefetch_depth_ = 1;
  Impl* impl_            = nullptr;
};

FACTORY_REGISTER(
    Index, DRAM_PET_HASH_LOCAL, DramPetHashLocalIndex, const BaseKVConfig&);
