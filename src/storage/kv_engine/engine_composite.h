#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "base/factory.h"
#include "storage/index/dram/extendible_hash_index.h"
#include "storage/index/dram/pet_hash_index.h"
#include "storage/index/dram/unordered_map_index.h"
#include "storage/index/ssd/ssd_extendible_hash_index.h"
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
    std::shared_lock<std::shared_mutex> lock(KeyMutex(key));
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    if (handle == kValueHandleNone) {
      value.clear();
      return;
    }
    value.resize(value_store_->SlotCapacity(handle));
    const size_t actual =
        value_store_->Read(handle, value.data(), value.size());
    value.resize(actual);
  }

  bool Exists(const uint64_t key, unsigned tid) override {
    (void)tid;
    std::shared_lock<std::shared_mutex> lock(KeyMutex(key));
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    return handle != kValueHandleNone;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    PutInternal(key, value.data(), value.size(), tid, true);
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
        PutInternal(keys[i],
                    item.Data(),
                    static_cast<size_t>(item.Size()) * sizeof(float),
                    tid,
                    false);
      }
      return;
    }

    std::vector<size_t> lock_ids;
    lock_ids.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      lock_ids.push_back(LockIndex(keys[i]));
    }
    std::sort(lock_ids.begin(), lock_ids.end());
    lock_ids.erase(
        std::unique(lock_ids.begin(), lock_ids.end()), lock_ids.end());

    std::vector<std::unique_lock<std::shared_mutex>> locks;
    locks.reserve(lock_ids.size());
    for (size_t id : lock_ids) {
      locks.emplace_back(key_mutexes_[id]);
    }

    struct OverwriteItem {
      Value_t handle = kValueHandleNone;
      ValueStore::WriteSpec spec{};
    };
    struct ReallocItem {
      uint64_t key       = 0;
      Value_t old_handle = kValueHandleNone;
      ValueStore::WriteSpec spec{};
    };
    std::vector<OverwriteItem> overwrites;
    std::vector<ReallocItem> reallocs;
    overwrites.reserve(static_cast<size_t>(keys.Size()));
    reallocs.reserve(static_cast<size_t>(keys.Size()));

    for (int i = 0; i < keys.Size(); ++i) {
      const auto& item   = (*values)[i];
      const void* data   = item.Data();
      const size_t size  = static_cast<size_t>(item.Size()) * sizeof(float);
      Value_t old_handle = kValueHandleNone;
      index_->Get(keys[i], old_handle);
      if (old_handle != kValueHandleNone &&
          value_store_->SlotCapacity(old_handle) >= size) {
        overwrites.push_back(
            OverwriteItem{old_handle, ValueStore::WriteSpec{data, size}});
        continue;
      }
      reallocs.push_back(
          ReallocItem{keys[i], old_handle, ValueStore::WriteSpec{data, size}});
    }

    if (!overwrites.empty()) {
      std::vector<uint64_t> handles;
      std::vector<ValueStore::WriteSpec> specs;
      handles.reserve(overwrites.size());
      specs.reserve(overwrites.size());
      for (const auto& item : overwrites) {
        handles.push_back(item.handle);
        specs.push_back(item.spec);
      }
      value_store_->BatchWrite(handles, specs);
    }

    if (!reallocs.empty()) {
      std::vector<ValueStore::WriteSpec> specs;
      specs.reserve(reallocs.size());
      for (const auto& item : reallocs) {
        specs.push_back(item.spec);
      }
      const auto new_handles = value_store_->BatchAllocAndWrite(specs);
      if (new_handles.size() != reallocs.size()) {
        LOG(FATAL) << "KVEngine::BatchPut allocation result size mismatch";
      }
      for (size_t i = 0; i < reallocs.size(); ++i) {
        if (new_handles[i] == kValueHandleNone) {
          LOG(FATAL) << "KVEngine batch value allocation failed, key="
                     << reallocs[i].key << " size=" << reallocs[i].spec.size;
        }
        index_->Put(reallocs[i].key, new_handles[i]);
        if (reallocs[i].old_handle != kValueHandleNone) {
          value_store_->Free(reallocs[i].old_handle);
        }
      }
    }
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    values->resize(keys.Size());
    thread_local std::vector<Value_t> handles;
    thread_local std::vector<std::vector<float>> buffers;
    handles.assign(keys.Size(), kValueHandleNone);
    buffers.clear();
    buffers.resize(keys.Size());

    std::vector<size_t> lock_ids;
    lock_ids.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      lock_ids.push_back(LockIndex(keys[i]));
    }
    std::sort(lock_ids.begin(), lock_ids.end());
    lock_ids.erase(
        std::unique(lock_ids.begin(), lock_ids.end()), lock_ids.end());

    std::vector<std::shared_lock<std::shared_mutex>> locks;
    locks.reserve(lock_ids.size());
    for (size_t id : lock_ids) {
      locks.emplace_back(key_mutexes_[id]);
    }

    std::vector<uint64_t> batch_handles;
    std::vector<size_t> batch_indices;
    batch_handles.reserve(static_cast<size_t>(keys.Size()));
    batch_indices.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      index_->Get(keys[i], handles[i]);
      if (handles[i] == kValueHandleNone) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      if (const char* ptr = value_store_->DirectPtr(handles[i])) {
        const size_t bytes = value_store_->SlotCapacity(handles[i]);
        (*values)[i]       = base::ConstArray<float>(
            reinterpret_cast<float*>(const_cast<char*>(ptr)),
            bytes / sizeof(float));
        continue;
      }
      batch_handles.push_back(handles[i]);
      batch_indices.push_back(static_cast<size_t>(i));
    }

    std::vector<ValueStore::ReadResult> batch_results;
    if (!batch_handles.empty()) {
      value_store_->BatchRead(batch_handles, batch_results);
      if (batch_results.size() != batch_indices.size()) {
        LOG(FATAL) << "KVEngine::BatchGet read result size mismatch";
      }
      for (size_t i = 0; i < batch_indices.size(); ++i) {
        const size_t idx   = batch_indices[i];
        const auto& result = batch_results[i];
        buffers[idx].resize(result.data.size() / sizeof(float));
        if (!result.data.empty()) {
          std::memcpy(
              buffers[idx].data(), result.data.data(), result.data.size());
        }
        (*values)[idx] =
            base::ConstArray<float>(buffers[idx].data(), buffers[idx].size());
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
      std::string current;
      Get(key, current, tid);
      if (current.size() == row_bytes) {
        std::memcpy(row.data(), current.data(), row_bytes);
      } else {
        std::fill(row.begin(), row.end(), 0.0f);
      }
      const float* grad = grads + r * embedding_dim;
      for (int64_t c = 0; c < embedding_dim; ++c) {
        row[c] -= learning_rate * grad[c];
      }
      PutInternal(key, row.data(), row_bytes, tid, false);
    }
    return true;
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    const auto& j           = config_.json_config_;
    const size_t value_size = j.at("value").value("default_value_size_hint", 0);
    if (value_size == 0) {
      LOG(FATAL) << "KVEngine::BulkLoad requires value_size hint";
    }
    const char* data = reinterpret_cast<const char*>(value);
    for (int i = 0; i < keys.Size(); ++i) {
      PutInternal(keys[i], data + i * value_size, value_size, 0, false);
    }
  }

  void Util() override {
    LOG(INFO) << "KVEngine index utilization=" << index_->Utilization()
              << " value=" << value_store_->GetInfo();
  }

  void DebugInfo() const override {
    index_->DebugInfo();
    LOG(INFO) << value_store_->GetInfo();
  }

private:
  void PutInternal(uint64_t key,
                   const void* data,
                   size_t size,
                   unsigned tid,
                   bool emit_fence) {
    (void)tid;
    (void)emit_fence;
    std::unique_lock<std::shared_mutex> lock(KeyMutex(key));
    Value_t old_handle = kValueHandleNone;
    index_->Get(key, old_handle);
    if (old_handle != kValueHandleNone &&
        value_store_->SlotCapacity(old_handle) >= size) {
      value_store_->Write(old_handle, data, size);
      return;
    }
    Value_t new_handle = value_store_->AllocAndWrite(data, size);
    if (new_handle == kValueHandleNone) {
      LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                 << " size=" << size;
      return;
    }
    index_->Put(key, new_handle);
    if (old_handle != kValueHandleNone) {
      value_store_->Free(old_handle);
    }
  }

  std::shared_mutex& KeyMutex(uint64_t key) {
    return key_mutexes_[LockIndex(key)];
  }

  size_t LockIndex(uint64_t key) const {
    return static_cast<size_t>(key & (kLockStripeNum - 1));
  }

  BaseKVConfig config_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<ValueStore> value_store_;
  int num_threads_                       = 0;
  static constexpr size_t kLockStripeNum = 4096;
  std::array<std::shared_mutex, kLockStripeNum> key_mutexes_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
