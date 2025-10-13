#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"

class KVEngineMap : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

 public:
  KVEngineMap(const BaseKVConfig &config)
      : BaseKV(config),
        kValueSize_(config.json_config_.value("value_size", 0)),
        kPreKnownValueSize_(
            config.json_config_.value("pre_known_value_size", false)) {
    if (kPreKnownValueSize_) CHECK_NE(kValueSize_, 0);
    auto capacity = config.json_config_.at("capacity").get<uint64_t>();
    auto path = config.json_config_.at("path").get<std::string>();

    // step1 init index pool
    hash_table_ = new std::unordered_map<uint64_t, uint64_t>();

    // step2 init value (malloc)
    uint64_t value_shm_size =
        config.json_config_.at("value_pool_size").get<uint64_t>();

    // shm_malloc_.reset(new base::PersistSimpleMalloc(
    //     path + "/value",
    // value_shm_size,
    //     config.json_config_.at("value_size").get<size_t>()));
    shm_malloc_.reset(
        new base::PersistLoopShmMalloc(path + "/value", value_shm_size));

    if (!valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize)) {
      base::file_util::Delete(path + "/valid", false);
      CHECK(
          valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize));
      shm_malloc_->Initialize();
    }
    LOG(INFO) << "After init: [shm_malloc] " << shm_malloc_->GetInfo();
  }

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    base::PetKVData shmkv_data;
    std::shared_lock<std::shared_mutex> _(lock_);
    auto iter = hash_table_->find(key);

    if (iter == hash_table_->end()) {
      value = std::string();
    } else {
      uint64_t &read_value = iter->second;
      shmkv_data = *(base::PetKVData *)(&read_value);
      char *data = shm_malloc_->GetMallocData(shmkv_data.shm_malloc_offset());

      int size;
      if (kPreKnownValueSize_) {
        size = kValueSize_;
      } else {
        size = shm_malloc_->GetMallocSize(shmkv_data.shm_malloc_offset());
      }
      value = std::string(data, size);
    }
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    base::PetKVData shmkv_data;
    char *sync_data = shm_malloc_->New(value.size());
    shmkv_data.SetShmMallocOffset(shm_malloc_->GetMallocOffset(sync_data));
    memcpy(sync_data, value.data(), value.size());
    _mm_mfence();
    asm volatile("" ::: "memory");
    std::unique_lock<std::shared_mutex> _(lock_);
    hash_table_->insert({key, shmkv_data.data_value});
  }

  void BatchGet(base::ConstArray<uint64> keys,
                std::vector<base::ConstArray<float>> *values,
                unsigned tid) override {
    values->clear();
    for (auto k : keys) {
      base::PetKVData shmkv_data;
      auto iter = hash_table_->find(k);
      if (iter == hash_table_->end()) {
        values->emplace_back(base::ConstArray<float>());
      } else {
        uint64_t &read_value = iter->second;
        shmkv_data = *(base::PetKVData *)(&read_value);
        char *data = shm_malloc_->GetMallocData(shmkv_data.shm_malloc_offset());

        int size;
        if (kPreKnownValueSize_) {
          size = kValueSize_;
        } else {
          size = shm_malloc_->GetMallocSize(shmkv_data.shm_malloc_offset());
        }
        values->emplace_back((float *)data, size / sizeof(float));
      }
    }
  }

  ~KVEngineMap() {
    std::cout << "exit KVEngineMap" << std::endl;
    // hash_table_->hash_name();
  }

  void clear() {
    std::unique_lock<std::shared_mutex> _(lock_);
    for (auto &item : *hash_table_) {
      base::PetKVData shmkv_data = *(base::PetKVData *)(&item.second);
      shm_malloc_->Free(
          shm_malloc_->GetMallocData(shmkv_data.shm_malloc_offset()));
    }
    hash_table_->clear();
  }

 private:
  std::unordered_map<uint64_t, uint64_t> *hash_table_;
  std::shared_mutex lock_;

  const bool kPreKnownValueSize_ = false;
  const int kValueSize_ = 0;
  std::unique_ptr<base::MallocApi> shm_malloc_;

  base::ShmFile valid_shm_file_;
};

FACTORY_REGISTER(BaseKV, KVEngineMap, KVEngineMap, const BaseKVConfig &);
