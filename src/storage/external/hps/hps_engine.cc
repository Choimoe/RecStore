#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "base/factory.h"
#include "storage/kv_engine/base_kv.h"
#include "third_party/HugeCTR/HugeCTR/include/hps/database_backend.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/hash_map_backend.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/rocksdb_backend.hpp"

namespace {

using HpsBackend = HugeCTR::DatabaseBackendBase<long long>;

size_t ConfigValueSize(const BaseKVConfig& config) {
  const auto& j = config.json_config_;
  if (j.contains("value_size")) {
    return j.at("value_size").get<size_t>();
  }
  if (j.contains("value")) {
    return j.at("value").value("default_value_size_hint", 0);
  }
  return 0;
}

size_t ConfigMaxBatchSize(const BaseKVConfig& config) {
  return config.json_config_.value("max_batch_size", 64UL * 1024UL);
}

int ConfigThreads(const BaseKVConfig& config) {
  if (config.num_threads_ > 0) {
    return config.num_threads_;
  }
  return config.json_config_.value("num_threads", 1);
}

std::string ConfigTableName(const BaseKVConfig& config) {
  return config.json_config_.value("table_name", std::string("default"));
}

void ValidateFloatAligned(size_t value_size, const char* operation) {
  if (value_size == 0 || value_size % sizeof(float) != 0) {
    throw std::invalid_argument(
        std::string(operation) +
        " requires a non-zero float-aligned value_size");
  }
}

std::vector<long long> ConvertKeys(base::ConstArray<uint64_t> keys) {
  std::vector<long long> out(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    out[i] = static_cast<long long>(keys[i]);
  }
  return out;
}

std::unique_ptr<HpsBackend> CreateHashMapBackend(const BaseKVConfig& config) {
  HugeCTR::HashMapBackendParams params;
  params.max_batch_size = ConfigMaxBatchSize(config);
  params.num_partitions =
      static_cast<size_t>(std::max(1, ConfigThreads(config)));
  params.overflow_margin = std::numeric_limits<size_t>::max();
  return std::make_unique<HugeCTR::HashMapBackend<long long>>(params);
}

std::unique_ptr<HpsBackend> CreateRocksDBBackend(const BaseKVConfig& config) {
  HugeCTR::RocksDBBackendParams params;
  const std::string path = config.json_config_.at("path").get<std::string>();
  std::filesystem::create_directories(path);
  params.path =
      config.json_config_.value("rocksdb_path", path + "/hps_rocksdb");
  params.max_batch_size = ConfigMaxBatchSize(config);
  params.num_threads = static_cast<size_t>(std::max(1, ConfigThreads(config)));
  return std::make_unique<HugeCTR::RocksDBBackend<long long>>(params);
}

class KVEngineHPSBase : public BaseKV {
public:
  KVEngineHPSBase(const BaseKVConfig& config,
                  std::unique_ptr<HpsBackend> backend)
      : BaseKV(config),
        value_size_(ConfigValueSize(config)),
        table_name_(ConfigTableName(config)),
        backend_(std::move(backend)) {
    if (value_size_ == 0) {
      throw std::invalid_argument("HPS BaseKV engine requires value_size");
    }
    if (!backend_) {
      throw std::runtime_error("HPS BaseKV engine backend is null");
    }
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    (void)tid;
    const long long hps_key = static_cast<long long>(key);
    value.assign(value_size_, '\0');
    bool missed = false;
    backend_->fetch(
        table_name_,
        1,
        &hps_key,
        value.data(),
        value_size_,
        [&](size_t) { missed = true; },
        std::chrono::nanoseconds::zero());
    if (missed) {
      value.clear();
    }
  }

  bool Exists(const uint64_t key, unsigned tid) override {
    (void)tid;
    const long long hps_key = static_cast<long long>(key);
    return backend_->contains(
               table_name_, 1, &hps_key, std::chrono::nanoseconds::zero()) == 1;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    (void)tid;
    if (value.size() != value_size_) {
      throw std::invalid_argument("HPS BaseKV engine requires fixed-size Put");
    }
    const long long hps_key = static_cast<long long>(key);
    backend_->insert(
        table_name_,
        1,
        &hps_key,
        value.data(),
        static_cast<uint32_t>(value_size_),
        value_size_);
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "HPS BaseKV BatchPut");
    if (values == nullptr || keys.Size() != static_cast<int>(values->size())) {
      throw std::invalid_argument("HPS BaseKV BatchPut size mismatch");
    }

    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    for (int i = 0; i < keys.Size(); ++i) {
      const auto& row = (*values)[i];
      if (row.Size() != floats_per_row) {
        throw std::invalid_argument("HPS BaseKV BatchPut row size mismatch");
      }
      std::memcpy(flat.data() + static_cast<size_t>(i) * value_size_,
                  row.Data(),
                  value_size_);
    }
    std::vector<long long> hps_keys = ConvertKeys(keys);
    backend_->insert(
        table_name_,
        hps_keys.size(),
        hps_keys.data(),
        flat.data(),
        static_cast<uint32_t>(value_size_),
        value_size_);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "HPS BaseKV BatchGet");
    values->resize(keys.Size());
    thread_local std::vector<std::vector<float>> buffers;
    buffers.assign(keys.Size(), {});

    std::vector<long long> hps_keys = ConvertKeys(keys);
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    std::vector<uint8_t> misses(keys.Size(), 0);
    backend_->fetch(
        table_name_,
        hps_keys.size(),
        hps_keys.data(),
        flat.data(),
        value_size_,
        [&](size_t index) { misses[index] = 1; },
        std::chrono::nanoseconds::zero());

    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    for (int i = 0; i < keys.Size(); ++i) {
      if (misses[i]) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      buffers[i].resize(floats_per_row);
      std::memcpy(buffers[i].data(),
                  flat.data() + static_cast<size_t>(i) * value_size_,
                  value_size_);
      (*values)[i] =
          base::ConstArray<float>(buffers[i].data(), buffers[i].size());
    }
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    if (value == nullptr) {
      throw std::invalid_argument("HPS BaseKV BulkLoad value is null");
    }
    std::vector<long long> hps_keys = ConvertKeys(keys);
    backend_->insert(
        table_name_,
        hps_keys.size(),
        hps_keys.data(),
        reinterpret_cast<const char*>(value),
        static_cast<uint32_t>(value_size_),
        value_size_);
  }

private:
  size_t value_size_;
  std::string table_name_;
  std::unique_ptr<HpsBackend> backend_;
};

} // namespace

class KVEngineHPSHashMap final : public KVEngineHPSBase {
public:
  explicit KVEngineHPSHashMap(const BaseKVConfig& config)
      : KVEngineHPSBase(config, CreateHashMapBackend(config)) {}
};

class KVEngineHPSRocksDB final : public KVEngineHPSBase {
public:
  explicit KVEngineHPSRocksDB(const BaseKVConfig& config)
      : KVEngineHPSBase(config, CreateRocksDBBackend(config)) {}
};

FACTORY_REGISTER(
    BaseKV, KVEngineHPSHashMap, KVEngineHPSHashMap, const BaseKVConfig&);
FACTORY_REGISTER(
    BaseKV, KVEngineHPSRocksDB, KVEngineHPSRocksDB, const BaseKVConfig&);
