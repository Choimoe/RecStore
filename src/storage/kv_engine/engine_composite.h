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
    std::shared_lock<std::shared_mutex> read_lock(
        sgd_update_locks_[key % kSgdUpdateLockCount]);
    GetUnlocked(key, value);
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
    std::shared_lock<std::shared_mutex> read_lock(
        sgd_update_locks_[key % kSgdUpdateLockCount]);
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    return handle != kValueHandleNone;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    std::unique_lock<std::shared_mutex> write_lock(
        sgd_update_locks_[key % kSgdUpdateLockCount]);
    PutInternalUnlocked(key, value.data(), value.size(), tid, true);
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
        std::unique_lock<std::shared_mutex> write_lock(
            sgd_update_locks_[keys[i] % kSgdUpdateLockCount]);
        PutInternalUnlocked(
            keys[i],
            item.Data(),
            static_cast<size_t>(item.Size()) * sizeof(float),
            tid,
            false);
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

    std::vector<ValueStore::WriteSpec> specs;
    specs.reserve(items.size());
    for (const auto& item : items) {
      specs.push_back(item.spec);
    }
    const auto new_handles = value_store_->BatchAllocAndWrite(specs);
    if (new_handles.size() != items.size()) {
      LOG(FATAL) << "KVEngine::BatchPut allocation result size mismatch";
    }
    for (size_t i = 0; i < items.size(); ++i) {
      if (new_handles[i] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine batch value allocation failed, key="
                   << items[i].key << " size=" << items[i].spec.size;
      }
    }

    for (size_t i = 0; i < items.size(); ++i) {
      std::unique_lock<std::shared_mutex> write_lock(
          sgd_update_locks_[items[i].key % kSgdUpdateLockCount]);
      Value_t old_handle = index_->Put(items[i].key, new_handles[i], tid);
      if (old_handle != kValueHandleNone) {
        value_store_->Retire(old_handle);
      }
    }
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    std::vector<std::shared_lock<std::shared_mutex>> read_locks;
    LockReadStripes(keys, &read_locks);
    values->resize(keys.Size());
    thread_local std::vector<Value_t> handles;
    thread_local std::vector<std::vector<float>> buffers;
    handles.assign(keys.Size(), kValueHandleNone);
    buffers.clear();
    buffers.resize(keys.Size());

    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    std::vector<uint64_t> batch_handles;
    std::vector<size_t> batch_indices;
    batch_handles.reserve(static_cast<size_t>(keys.Size()));
    batch_indices.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
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
    std::vector<std::shared_lock<std::shared_mutex>> read_locks;
    LockReadStripes(keys, &read_locks);

    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }

    int64_t local_missing_rows = 0;
    const size_t expected_bytes =
        static_cast<size_t>(embedding_dim) * sizeof(float);
    std::vector<uint64_t> batch_handles;
    std::vector<size_t> batch_indices;
    batch_handles.reserve(static_cast<size_t>(keys.Size()));
    batch_indices.reserve(static_cast<size_t>(keys.Size()));

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
        return false;
      }
      if (const char* ptr = value_store_->DirectPtr(handle)) {
        std::memcpy(dst, ptr, bytes);
        continue;
      }
      batch_handles.push_back(handle);
      batch_indices.push_back(static_cast<size_t>(row));
    }

    if (!batch_handles.empty()) {
      std::vector<ValueStore::ReadResult> batch_results;
      value_store_->BatchRead(batch_handles, batch_results);
      if (batch_results.size() != batch_indices.size()) {
        LOG(FATAL) << "KVEngine::BatchGetFlat read result size mismatch";
      }
      for (size_t i = 0; i < batch_indices.size(); ++i) {
        const auto& data = batch_results[i].data;
        if (data.size() != expected_bytes) {
          LOG(ERROR) << "KVEngine::BatchGetFlat read size mismatch at row="
                     << batch_indices[i] << " expected_bytes=" << expected_bytes
                     << " actual_bytes=" << data.size();
          return false;
        }
        std::memcpy(values + batch_indices[i] * embedding_dim,
                    data.data(),
                    data.size());
      }
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing_rows;
    }
    return true;
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
      std::unique_lock<std::shared_mutex> update_lock(
          sgd_update_locks_[key % kSgdUpdateLockCount]);
      std::string current;
      GetUnlocked(key, current);
      if (current.size() == row_bytes) {
        std::memcpy(row.data(), current.data(), row_bytes);
      } else {
        std::fill(row.begin(), row.end(), 0.0f);
      }
      const float* grad = grads + r * embedding_dim;
      for (int64_t c = 0; c < embedding_dim; ++c) {
        row[c] -= learning_rate * grad[c];
      }
      PutInternalUnlocked(key, row.data(), row_bytes, tid, false);
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
  void
  LockReadStripes(base::ConstArray<uint64_t> keys,
                  std::vector<std::shared_lock<std::shared_mutex>>* locks) {
    locks->clear();
    locks->reserve(static_cast<size_t>(keys.Size()));
    std::array<bool, kSgdUpdateLockCount> locked_stripes{};
    for (int i = 0; i < keys.Size(); ++i) {
      const size_t stripe = keys[i] % kSgdUpdateLockCount;
      if (!locked_stripes[stripe]) {
        locks->emplace_back(sgd_update_locks_[stripe]);
        locked_stripes[stripe] = true;
      }
    }
  }

  void PutInternalUnlocked(
      uint64_t key,
      const void* data,
      size_t size,
      unsigned tid,
      bool emit_fence) {
    (void)tid;
    (void)emit_fence;
    Value_t new_handle = value_store_->AllocAndWrite(data, size);
    if (new_handle == kValueHandleNone) {
      LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                 << " size=" << size;
      return;
    }
    Value_t old_handle = index_->Put(key, new_handle, tid);
    if (old_handle != kValueHandleNone) {
      value_store_->Retire(old_handle);
    }
  }

  BaseKVConfig config_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<ValueStore> value_store_;
  int num_threads_                            = 0;
  static constexpr size_t kSgdUpdateLockCount = 4096;
  std::array<std::shared_mutex, kSgdUpdateLockCount> sgd_update_locks_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
