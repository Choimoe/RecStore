#pragma once

#include <memory>
#include <new>

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

  void Put(Key_t key, Value_t pointer, unsigned tid) override {
    (void)tid;
    impl_->Set(key, pointer);
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

  bool Delete(Key_t& key) override { return impl_->Delete(key); }
  size_t Capacity() override { return impl_->Capacity(); }

private:
  base::PetHash<Key_t, Value_t, false>* impl_ = nullptr;
};
