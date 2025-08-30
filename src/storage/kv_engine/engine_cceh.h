#pragma once

#include <cstring> // for std::memcpy
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../nvm/pet_kv/shm_common.h"
#include "../ssd/CCEH.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"

class KVEngineCCEH : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineCCEH(const BaseKVConfig &config)
      : BaseKV(config),
        shm_malloc_(config.json_config_.at("path").get<std::string>() +
                        "/value",
                    1.2 * config.json_config_.at("capacity").get<size_t>() *
                        config.json_config_.at("value_size").get<size_t>()) {
    value_size_ = config.json_config_.at("value_size").get<int>();

    // 初始化extendible
    // hash表，数据库文件放在配置的工作路径中，避免测试之间相互影响
    std::string path = config.json_config_.at("path").get<std::string>();
    std::string db_path = path + "/cceh_test.db";
    hash_table_ = new CCEH(db_path);

    // 初始化值存储区域
    uint64_t value_shm_size =
        config.json_config_.at("capacity").get<uint64_t>() *
        config.json_config_.at("value_size").get<uint64_t>();

    if (!valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize)) {
      base::file_util::Delete(path + "/valid", false);
      CHECK(
          valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize));
      shm_malloc_.Initialize();
    }
    LOG(INFO) << "After init: [shm_malloc] " << shm_malloc_.GetInfo();
  }

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    base::PetKVData shmkv_data;
    Key_t hash_key = key;
    Value_t read_value = hash_table_->Get(hash_key);

    if (read_value == NONE) {
      value = std::string();
    } else {
      shmkv_data.data_value = read_value;
      char *data = shm_malloc_.GetMallocData(shmkv_data.shm_malloc_offset());
      if (data == nullptr) {
        value = std::string();
        return;
      }
#ifdef XMH_VARIABLE_SIZE_KV
      int size = shm_malloc_.GetMallocSize(shmkv_data.shm_malloc_offset());
#else
      int size = value_size_;
#endif
      value = std::string(data, size);
    }
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    base::PetKVData shmkv_data;
    char *sync_data = shm_malloc_.New(value.size());
    shmkv_data.SetShmMallocOffset(shm_malloc_.GetMallocOffset(sync_data));
    std::memcpy(sync_data, value.data(), value.size());
    Key_t hash_key = key;
    hash_table_->Insert(hash_key, shmkv_data.data_value);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>> *values,
                unsigned tid) override {
    values->clear();
    int size = keys.Size();
    std::vector<std::unique_ptr<coroutine<Value_t>::pull_type>> coros;
    for (size_t i = 0; i < size; i++) {
      auto k = keys[i];
      coros.emplace_back(new coroutine<Value_t>::pull_type{
          [this, i, k](auto &yield) { hash_table_->Get(yield, i, k); }});
    }
    struct io_uring *ring = hash_table_->fm->get_thread_ring();
    struct io_uring_cqe *cqe;
    std::vector<Value_t> vals(size, NONE);
    while (active_coro) {
      if (!io_uring_peek_cqe(ring, &cqe)) {
        if (cqe->res < 0) {
          active_coro--;
          io_uring_cqe_seen(ring, cqe);
          throw std::runtime_error("Write operation failed: " +
                                   std::string(strerror(-cqe->res)));
        }
        if (cqe->res != PAGE_SIZE) {
          active_coro--;
          io_uring_cqe_seen(ring, cqe);
          throw std::runtime_error("Incomplete write: expected " +
                                   std::to_string(PAGE_SIZE) +
                                   " bytes, wrote " + std::to_string(cqe->res));
        }
        int id = cqe->user_data;
        active_coro--;
        io_uring_cqe_seen(ring, cqe);
        if (coros[id] && *coros[id]) {
          (*coros[id])();
          vals[id] = coros[id]->get();
        }
      }
    }
    for (auto v : vals) {
      base::PetKVData shmkv_data;
      if (v == NONE) {
        values->emplace_back();
      } else {
        shmkv_data.data_value = v;
        char *data = shm_malloc_.GetMallocData(shmkv_data.shm_malloc_offset());
        if (data == nullptr) {
          values->emplace_back();
          continue;
        }
        int size = value_size_;
        values->emplace_back((float *)data, size / sizeof(float));
      }
    }
  }

  ~KVEngineCCEH() {
    std::cout << "exit KVEngineCCEH" << std::endl;
    if (hash_table_) {
      delete hash_table_;
      hash_table_ = nullptr;
    }
  }

private:
  CCEH *hash_table_;
  // std::shared_mutex lock_;

  uint64_t counter = 0;
  std::string dict_pool_name_;
  size_t dict_pool_size_;
  int value_size_;
#ifdef XMH_SIMPLE_MALLOC
  base::PersistSimpleMalloc shm_malloc_;
#else
  base::PersistLoopShmMalloc shm_malloc_;
#endif
  base::ShmFile valid_shm_file_;
};

FACTORY_REGISTER(BaseKV, KVEngineCCEH, KVEngineCCEH, const BaseKVConfig &);