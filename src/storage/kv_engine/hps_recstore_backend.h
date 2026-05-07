#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "third_party/HugeCTR/HugeCTR/include/hps/database_backend.hpp"

namespace recstore {
namespace storage {

struct HpsRecStoreBackendParams : public HugeCTR::VolatileBackendParams {
  std::string path;
  std::string index_type{"DRAM_EXTENDIBLE_HASH"};
  std::string value_store_type{"DRAM_VALUE_STORE"};
  std::string dram_allocator{"PERSIST_LOOP_SLAB"};
  uint64_t capacity{0};
  uint32_t value_size{0};
  uint64_t dram_capacity_bytes{0};
  int num_threads{1};
};

template <typename Key>
class HpsRecStoreBackend final
    : public HugeCTR::VolatileBackend<Key, HpsRecStoreBackendParams> {
public:
  using Base = HugeCTR::VolatileBackend<Key, HpsRecStoreBackendParams>;

  explicit HpsRecStoreBackend(const HpsRecStoreBackendParams& params)
      : Base(params) {
    base::PMMmapRegisterCenter::GetConfig().use_dram =
        params.value_store_type == "DRAM_VALUE_STORE" ||
        params.value_store_type == "TIERED_VALUE_STORE";

    BaseKVConfig cfg;
    cfg.num_threads_ = params.num_threads;
    const uint64_t value_size =
        params.value_size > 0 ? params.value_size : sizeof(float);
    const uint64_t dram_capacity =
        params.dram_capacity_bytes > 0
            ? params.dram_capacity_bytes
            : std::max<uint64_t>(1, params.capacity) * value_size * 6 / 5;
    cfg.json_config_ = {
        {"path", params.path},
        {"capacity", params.capacity},
        {"index", {{"type", params.index_type}}},
        {"value",
         {{"type", params.value_store_type},
          {"default_value_size_hint", value_size},
          {"dram_allocator",
           {{"type", params.dram_allocator},
            {"capacity_bytes", dram_capacity}}}}}};

    auto resolved = base::ResolveEngine(cfg);
    kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        resolved.engine, resolved.cfg));
    if (!kv_) {
      throw std::runtime_error("failed to create RecStore BaseKV for HPS");
    }
  }

  const char* get_name() const override { return "RecStoreBackend"; }
  bool is_shared() const override { return false; }

  size_t size(const std::string& table_name) const override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    const auto it = table_sizes_.find(table_name);
    return it == table_sizes_.end() ? 0 : it->second;
  }

  size_t contains(const std::string& table_name,
                  size_t num_keys,
                  const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override {
    (void)table_name;
    (void)time_budget;
    size_t hits = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (kv_->Exists(static_cast<uint64_t>(keys[i]), 0)) {
        ++hits;
      }
    }
    return hits;
  }

  size_t insert(const std::string& table_name,
                size_t num_pairs,
                const Key* keys,
                const char* values,
                uint32_t value_size,
                size_t value_stride) override {
    if (num_pairs == 0) {
      return 0;
    }
    std::vector<uint64_t> u64_keys(num_pairs);
    std::vector<std::vector<float>> rows(num_pairs);
    std::vector<base::ConstArray<float>> row_views;
    row_views.reserve(num_pairs);
    for (size_t i = 0; i < num_pairs; ++i) {
      u64_keys[i]         = static_cast<uint64_t>(keys[i]);
      const char* src     = values + i * value_stride;
      const size_t floats = (value_size + sizeof(float) - 1) / sizeof(float);
      rows[i].assign(floats, 0.0f);
      std::memcpy(rows[i].data(), src, value_size);
      row_views.emplace_back(rows[i].data(), rows[i].size());
    }
    kv_->BatchPut(base::ConstArray<uint64_t>(u64_keys), &row_views, 0);
    {
      std::lock_guard<std::mutex> lock(table_sizes_mu_);
      table_sizes_[table_name] += num_pairs;
    }
    return num_pairs;
  }

  size_t fetch(const std::string& table_name,
               size_t num_keys,
               const Key* keys,
               char* values,
               size_t value_stride,
               const HugeCTR::DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override {
    (void)table_name;
    (void)time_budget;
    std::vector<uint64_t> u64_keys(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      u64_keys[i] = static_cast<uint64_t>(keys[i]);
    }
    std::vector<base::ConstArray<float>> rows;
    kv_->BatchGet(base::ConstArray<uint64_t>(u64_keys), &rows, 0);

    size_t hits = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (i < rows.size() && rows[i].Data() != nullptr && rows[i].Size() > 0) {
        const size_t bytes =
            std::min(value_stride, rows[i].Size() * sizeof(float));
        std::memcpy(values + i * value_stride, rows[i].Data(), bytes);
        if (bytes < value_stride) {
          std::memset(
              values + i * value_stride + bytes, 0, value_stride - bytes);
        }
        ++hits;
      } else {
        on_miss(i);
      }
    }
    return hits;
  }

  size_t
  fetch(const std::string& table_name,
        size_t num_indices,
        const size_t* indices,
        const Key* keys,
        char* values,
        size_t value_stride,
        const HugeCTR::DatabaseMissCallback& on_miss,
        const std::chrono::nanoseconds& time_budget) override {
    (void)time_budget;
    size_t hits = 0;
    for (size_t i = 0; i < num_indices; ++i) {
      const size_t index = indices[i];
      const Key key      = keys[index];
      hits += fetch(
          table_name,
          1,
          &key,
          values + index * value_stride,
          value_stride,
          [&](size_t) { on_miss(index); },
          std::chrono::nanoseconds::zero());
    }
    return hits;
  }

  size_t evict(const std::string& table_name) override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    table_sizes_.erase(table_name);
    return 0;
  }

  size_t evict(const std::string& table_name,
               size_t num_keys,
               const Key* keys) override {
    (void)table_name;
    (void)num_keys;
    (void)keys;
    return 0;
  }

  std::vector<std::string> find_tables(const std::string& model_name) override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    std::vector<std::string> tables;
    for (const auto& [name, _] : table_sizes_) {
      if (name.find(model_name) != std::string::npos) {
        tables.push_back(name);
      }
    }
    return tables;
  }

  size_t dump_bin(const std::string& table_name, std::ofstream& file) override {
    (void)table_name;
    (void)file;
    return 0;
  }

#ifdef HCTR_USE_ROCKS_DB
  size_t dump_sst(const std::string& table_name,
                  rocksdb::SstFileWriter& file) override {
    (void)table_name;
    (void)file;
    return 0;
  }
#endif

private:
  std::unique_ptr<BaseKV> kv_;
  mutable std::mutex table_sizes_mu_;
  std::unordered_map<std::string, size_t> table_sizes_;
};

} // namespace storage
} // namespace recstore
