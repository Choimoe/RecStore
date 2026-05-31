#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "base/factory.h"
#include "storage/index/dram/extendible_hash_index.h"
#include "storage/index/dram/pet_hash_index.h"
#include "storage/index/dram/unordered_map_index.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/hybrid_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class KVEngineComposite : public BaseKV {
public:
  KVEngineComposite(std::unique_ptr<Index> index,
                    std::unique_ptr<ValueStore> value_store,
                    int num_threads = 0)
      : BaseKV(BaseKVConfig{}),
        index_(std::move(index)),
        value_store_(std::move(value_store)),
        num_threads_(num_threads) {}

  explicit KVEngineComposite(const BaseKVConfig& config) : BaseKV(config) {
    config_                      = config;
    const auto& j                = config.json_config_;
    const std::string index_type = j.at("index").at("type").get<std::string>();
    const std::string value_type = j.at("value").at("type").get<std::string>();
    using IF                     = base::Factory<Index, const BaseKVConfig&>;
    using VF = base::Factory<ValueStore, const BaseKVConfig&>;
    index_.reset(IF::NewInstance(index_type, config));
    value_store_.reset(VF::NewInstance(value_type, config));
    num_threads_ = config.num_threads_;
    if (!index_ || !value_store_) {
      throw std::runtime_error("failed to create KVEngine components");
    }
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    (void)tid;
    GetOptimistic(key, value);
  }

  void GetUnlocked(const uint64_t key, std::string& value) {
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    if (handle == kValueHandleNone) {
      value.clear();
      return;
    }
    if (const char* ptr = value_store_->DirectPtr(handle)) {
      value.resize(value_store_->SlotCapacity(handle));
      std::memcpy(value.data(), ptr, value.size());
      return;
    }
    value.resize(value_store_->SlotCapacity(handle));
    const size_t actual =
        value_store_->Read(handle, value.data(), value.size());
    value.resize(actual);
  }

  bool Exists(const uint64_t key, unsigned tid) override {
    (void)tid;
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    return handle != kValueHandleNone;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    const size_t stripe = StripeFor(key);
    std::lock_guard<std::mutex> write_lock(stripe_write_locks_[stripe]);
    if (TryOverwriteExistingUnlocked(key, value.data(), value.size(), tid)) {
      return;
    }
    Value_t new_handle =
        value_store_->AllocAndWrite(value.data(), value.size());
    if (new_handle == kValueHandleNone) {
      LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                 << " size=" << value.size();
      return;
    }
    PublishHandleUnlocked(key, new_handle, tid);
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    if (values == nullptr || keys.Size() != static_cast<int>(values->size())) {
      LOG(FATAL) << "KVEngine::BatchPut size mismatch";
    }
    (void)tid;
    if (keys.Size() == 0) {
      return;
    }

    std::unordered_set<uint64_t> seen_keys;
    seen_keys.reserve(static_cast<size_t>(keys.Size()));
    bool has_duplicate_key = false;
    for (int i = 0; i < keys.Size(); ++i) {
      if (!seen_keys.insert(keys[i]).second) {
        has_duplicate_key = true;
        break;
      }
    }
    if (has_duplicate_key) {
      for (int i = 0; i < keys.Size(); ++i) {
        const auto& item = (*values)[i];
        Put(keys[i],
            std::string_view(reinterpret_cast<const char*>(item.Data()),
                             static_cast<size_t>(item.Size()) * sizeof(float)),
            tid);
      }
      return;
    }

    struct PutItem {
      uint64_t key = 0;
      ValueStore::WriteSpec spec{};
    };
    std::vector<PutItem> items;
    items.reserve(static_cast<size_t>(keys.Size()));

    for (int i = 0; i < keys.Size(); ++i) {
      const auto& item  = (*values)[i];
      const void* data  = item.Data();
      const size_t size = static_cast<size_t>(item.Size()) * sizeof(float);
      items.push_back(PutItem{keys[i], ValueStore::WriteSpec{data, size}});
    }

    for (size_t i = 0; i < items.size(); ++i) {
      const size_t stripe = StripeFor(items[i].key);
      std::lock_guard<std::mutex> write_lock(stripe_write_locks_[stripe]);
      if (TryOverwriteExistingUnlocked(
              items[i].key, items[i].spec.data, items[i].spec.size, tid)) {
        continue;
      }
      Value_t new_handle =
          value_store_->AllocAndWrite(items[i].spec.data, items[i].spec.size);
      if (new_handle == kValueHandleNone) {
        LOG(FATAL) << "KVEngine value allocation failed, key=" << items[i].key
                   << " size=" << items[i].spec.size;
      }
      PublishHandleUnlocked(items[i].key, new_handle, tid);
    }
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    values->resize(keys.Size());
    thread_local std::vector<std::vector<float>> buffers;
    buffers.clear();
    buffers.resize(keys.Size());

    for (int i = 0; i < keys.Size(); ++i) {
      std::string row;
      GetOptimistic(keys[i], row);
      if (row.empty()) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      auto& buffer = buffers[static_cast<size_t>(i)];
      buffer.resize(row.size() / sizeof(float));
      if (!row.empty()) {
        std::memcpy(buffer.data(), row.data(), row.size());
      }
      (*values)[i] = base::ConstArray<float>(buffer.data(), buffer.size());
    }
  }

  bool BatchGetFlat(base::ConstArray<uint64_t> keys,
                    float* values,
                    int64_t num_rows,
                    int64_t embedding_dim,
                    unsigned tid,
                    int64_t* missing_rows) override {
    if (values == nullptr || embedding_dim <= 0 ||
        keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }
    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    std::vector<size_t> stripes;
    std::vector<uint64_t> versions;
    for (;;) {
      BeginBatchRead(keys, &stripes, &versions);
      if (keys.Size() > 0) {
        index_->BatchGet(keys, handles.data(), tid);
      }
      int64_t local_missing_rows = 0;
      const size_t expected_bytes =
          static_cast<size_t>(embedding_dim) * sizeof(float);
      bool ok = true;
      for (int64_t row = 0; row < num_rows; ++row) {
        const Value_t handle = handles[static_cast<size_t>(row)];
        float* dst           = values + row * embedding_dim;
        if (handle == kValueHandleNone) {
          std::fill_n(dst, static_cast<size_t>(embedding_dim), 0.0f);
          ++local_missing_rows;
          continue;
        }
        const size_t bytes = value_store_->SlotCapacity(handle);
        if (bytes != expected_bytes) {
          LOG(ERROR) << "KVEngine::BatchGetFlat embedding_dim mismatch at row="
                     << row << " key=" << keys[static_cast<size_t>(row)]
                     << " expected_bytes=" << expected_bytes
                     << " actual_bytes=" << bytes;
          ok = false;
          break;
        }
        if (const char* ptr = value_store_->DirectPtr(handle)) {
          std::memcpy(dst, ptr, bytes);
        } else {
          const size_t actual = value_store_->Read(handle, dst, bytes);
          if (actual != bytes) {
            LOG(ERROR) << "KVEngine::BatchGetFlat read size mismatch at row="
                       << row << " expected_bytes=" << bytes
                       << " actual_bytes=" << actual;
            ok = false;
            break;
          }
        }
      }
      const bool stable = EndBatchReadAndValidate(stripes, versions);
      if (!ok) {
        return false;
      }
      if (stable) {
        if (missing_rows != nullptr) {
          *missing_rows = local_missing_rows;
        }
        return true;
      }
    }
  }

  bool ApplySgdUpdateFlat(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      uint8_t tag,
      unsigned tid) override {
    if (grads == nullptr || keys.Size() != static_cast<size_t>(num_rows) ||
        embedding_dim <= 0) {
      return false;
    }
    const int tag_bits      = static_cast<int>(sizeof(tag) * 8);
    const int shift         = static_cast<int>(sizeof(uint64_t) * 8) - tag_bits;
    const uint64_t key_mask = ~0ULL >> tag_bits;
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
    std::vector<float> row(embedding_dim);
    for (int64_t r = 0; r < num_rows; ++r) {
      const uint64_t key = (static_cast<uint64_t>(tag) << shift) |
                           (keys[static_cast<size_t>(r)] & key_mask);
      const size_t stripe = StripeFor(key);
      std::lock_guard<std::mutex> update_lock(stripe_write_locks_[stripe]);
      Value_t handle = kValueHandleNone;
      index_->Get(key, handle);
      const float* grad = grads + r * embedding_dim;
      if (handle != kValueHandleNone &&
          value_store_->SlotCapacity(handle) == row_bytes) {
        if (const char* ptr = value_store_->DirectPtr(handle)) {
          BeginStripeWrite(stripe);
          WaitForStripeReaders(stripe);
          float* direct = reinterpret_cast<float*>(const_cast<char*>(ptr));
          for (int64_t c = 0; c < embedding_dim; ++c) {
            direct[c] -= learning_rate * grad[c];
          }
          EndStripeWrite(stripe);
          continue;
        }
        const size_t actual = value_store_->Read(handle, row.data(), row_bytes);
        if (actual == row_bytes) {
          for (int64_t c = 0; c < embedding_dim; ++c) {
            row[c] -= learning_rate * grad[c];
          }
        } else {
          std::fill(row.begin(), row.end(), 0.0f);
          for (int64_t c = 0; c < embedding_dim; ++c) {
            row[c] -= learning_rate * grad[c];
          }
        }
      } else {
        std::fill(row.begin(), row.end(), 0.0f);
        for (int64_t c = 0; c < embedding_dim; ++c) {
          row[c] -= learning_rate * grad[c];
        }
      }
      Value_t new_handle = value_store_->AllocAndWrite(row.data(), row_bytes);
      if (new_handle == kValueHandleNone) {
        LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                   << " size=" << row_bytes;
        return false;
      }
      PublishHandleUnlocked(key, new_handle, tid);
    }
    return true;
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    const auto& j           = config_.json_config_;
    const size_t value_size = j.at("value").value("default_value_size_hint", 0);
    if (value_size == 0) {
      LOG(FATAL) << "KVEngine::BulkLoad requires value_size hint";
    }
    if (keys.Size() == 0) {
      return;
    }
    const char* data = reinterpret_cast<const char*>(value);
    std::vector<ValueStore::WriteSpec> specs;
    specs.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      specs.push_back(ValueStore::WriteSpec{data + i * value_size, value_size});
    }
    std::vector<uint64_t> handles = value_store_->BatchAllocAndWrite(specs);
    if (handles.size() != static_cast<size_t>(keys.Size())) {
      LOG(FATAL) << "KVEngine::BulkLoad allocation result size mismatch";
    }
    for (int i = 0; i < keys.Size(); ++i) {
      if (handles[static_cast<size_t>(i)] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine bulk value allocation failed, key=" << keys[i]
                   << " size=" << value_size;
      }
    }
    index_->BatchPut(keys, handles.data(), 0);
  }

  void Util() override {
    LOG(INFO) << "KVEngine index utilization=" << index_->Utilization()
              << " value=" << value_store_->GetInfo();
  }

  void DebugInfo() const override {
    index_->DebugInfo();
    LOG(INFO) << value_store_->GetInfo();
  }

  std::string ExtraResultFields() const override {
    return value_store_ ? value_store_->ExtraResultFields() : "";
  }

private:
  static size_t StripeFor(uint64_t key) { return key % kStripeCount; }

  uint64_t BeginStripeRead(size_t stripe) {
    for (;;) {
      const uint64_t version =
          stripe_versions_[stripe].load(std::memory_order_acquire);
      if ((version & 1U) != 0) {
        std::this_thread::yield();
        continue;
      }
      stripe_readers_[stripe].fetch_add(1, std::memory_order_acq_rel);
      const uint64_t confirmed =
          stripe_versions_[stripe].load(std::memory_order_acquire);
      if (version == confirmed && (confirmed & 1U) == 0) {
        return version;
      }
      stripe_readers_[stripe].fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  bool EndStripeReadAndValidate(size_t stripe, uint64_t version) {
    stripe_readers_[stripe].fetch_sub(1, std::memory_order_acq_rel);
    return stripe_versions_[stripe].load(std::memory_order_acquire) == version;
  }

  void BeginBatchRead(base::ConstArray<uint64_t> keys,
                      std::vector<size_t>* stripes,
                      std::vector<uint64_t>* versions) {
    for (;;) {
      stripes->clear();
      stripes->reserve(static_cast<size_t>(keys.Size()));
      std::array<bool, kStripeCount> seen{};
      for (int i = 0; i < keys.Size(); ++i) {
        const size_t stripe = StripeFor(keys[i]);
        if (!seen[stripe]) {
          seen[stripe] = true;
          stripes->push_back(stripe);
        }
      }
      versions->clear();
      versions->reserve(stripes->size());
      bool stable = true;
      for (size_t stripe : *stripes) {
        const uint64_t version =
            stripe_versions_[stripe].load(std::memory_order_acquire);
        if ((version & 1U) != 0) {
          stable = false;
          break;
        }
        stripe_readers_[stripe].fetch_add(1, std::memory_order_acq_rel);
        versions->push_back(version);
        const uint64_t confirmed =
            stripe_versions_[stripe].load(std::memory_order_acquire);
        if (confirmed != version || (confirmed & 1U) != 0) {
          stable = false;
          break;
        }
      }
      if (stable && versions->size() == stripes->size()) {
        return;
      }
      for (size_t i = 0; i < versions->size(); ++i) {
        stripe_readers_[(*stripes)[i]].fetch_sub(1, std::memory_order_acq_rel);
      }
      std::this_thread::yield();
    }
  }

  bool EndBatchReadAndValidate(const std::vector<size_t>& stripes,
                               const std::vector<uint64_t>& versions) {
    bool stable = true;
    for (size_t i = 0; i < stripes.size(); ++i) {
      const size_t stripe = stripes[i];
      stripe_readers_[stripe].fetch_sub(1, std::memory_order_acq_rel);
      if (stripe_versions_[stripe].load(std::memory_order_acquire) !=
          versions[i]) {
        stable = false;
      }
    }
    return stable;
  }

  void BeginStripeWrite(size_t stripe) {
    const uint64_t previous =
        stripe_versions_[stripe].fetch_add(1, std::memory_order_acq_rel);
    if ((previous & 1U) != 0) {
      LOG(FATAL) << "KVEngine stripe write version already odd";
    }
  }

  void WaitForStripeReaders(size_t stripe) {
    while (stripe_readers_[stripe].load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }
  }

  void EndStripeWrite(size_t stripe) {
    const uint64_t previous =
        stripe_versions_[stripe].fetch_add(1, std::memory_order_release);
    if ((previous & 1U) == 0) {
      LOG(FATAL) << "KVEngine stripe write version already even";
    }
  }

  void PublishHandleUnlocked(uint64_t key, Value_t new_handle, unsigned tid) {
    const size_t stripe = StripeFor(key);
    BeginStripeWrite(stripe);
    Value_t old_handle = index_->Put(key, new_handle, tid);
    WaitForStripeReaders(stripe);
    if (old_handle != kValueHandleNone) {
      value_store_->Retire(old_handle);
    }
    EndStripeWrite(stripe);
  }

  bool TryOverwriteExistingUnlocked(
      uint64_t key, const void* data, size_t size, unsigned tid) {
    (void)tid;
    const size_t stripe = StripeFor(key);
    Value_t handle      = kValueHandleNone;
    index_->Get(key, handle);
    if (handle == kValueHandleNone ||
        value_store_->SlotCapacity(handle) != size) {
      return false;
    }
    const char* ptr = value_store_->DirectPtr(handle);
    if (ptr == nullptr) {
      return false;
    }
    BeginStripeWrite(stripe);
    WaitForStripeReaders(stripe);
    std::memcpy(const_cast<char*>(ptr), data, size);
    EndStripeWrite(stripe);
    return true;
  }

  void GetOptimistic(uint64_t key, std::string& value) {
    const size_t stripe = StripeFor(key);
    for (;;) {
      const uint64_t version = BeginStripeRead(stripe);
      GetUnlocked(key, value);
      if (EndStripeReadAndValidate(stripe, version)) {
        return;
      }
    }
  }

  bool ReadFlatRowOptimistic(
      uint64_t key,
      float* dst,
      size_t expected_bytes,
      int64_t embedding_dim,
      bool* missing) {
    const size_t stripe = StripeFor(key);
    for (;;) {
      const uint64_t version = BeginStripeRead(stripe);
      Value_t handle         = kValueHandleNone;
      index_->Get(key, handle);
      bool ok  = true;
      *missing = false;
      if (handle == kValueHandleNone) {
        std::fill_n(dst, static_cast<size_t>(embedding_dim), 0.0f);
        *missing = true;
      } else {
        const size_t bytes = value_store_->SlotCapacity(handle);
        if (bytes != expected_bytes) {
          LOG(ERROR) << "KVEngine::BatchGetFlat embedding_dim mismatch key="
                     << key << " expected_bytes=" << expected_bytes
                     << " actual_bytes=" << bytes;
          ok = false;
        } else if (const char* ptr = value_store_->DirectPtr(handle)) {
          std::memcpy(dst, ptr, bytes);
        } else {
          const size_t actual = value_store_->Read(handle, dst, bytes);
          if (actual != bytes) {
            LOG(ERROR) << "KVEngine::BatchGetFlat read size mismatch key="
                       << key << " expected_bytes=" << bytes
                       << " actual_bytes=" << actual;
            ok = false;
          }
        }
      }
      const bool stable = EndStripeReadAndValidate(stripe, version);
      if (!ok) {
        return false;
      }
      if (stable) {
        return true;
      }
    }
  }

  BaseKVConfig config_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<ValueStore> value_store_;
  int num_threads_                     = 0;
  static constexpr size_t kStripeCount = 4096;
  std::array<std::mutex, kStripeCount> stripe_write_locks_;
  std::array<std::atomic<uint64_t>, kStripeCount> stripe_versions_{};
  std::array<std::atomic<uint32_t>, kStripeCount> stripe_readers_{};
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
