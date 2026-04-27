#pragma once
#include <folly/ProducerConsumerQueue.h>

#include <algorithm>
#include <atomic>
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <experimental/filesystem>
#include <random>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h" // NOLINT
#include "base/timer.h"
#include "parameters.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "optimizer/optimizer.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include "base/report/report_client.h"
#endif

using boost::coroutines2::coroutine;

static const int KEY_CNT = 12543670;

template <typename key_t>
struct TaskElement {
  TaskElement(const base::ConstArray<key_t>& keys,
              const base::MutableArray<ParameterPack>& packs,
              std::atomic_bool* promise)
      : keys(keys), packs(packs), promise(promise) {}

  TaskElement() {}

  base::ConstArray<key_t> keys;
  base::MutableArray<ParameterPack> packs;
  std::atomic_bool* promise;
};

class CachePS {
public:
  using key_t = uint64_t;

  CachePS(json config) {
    LOG(INFO) << "cache ps config: " << config.dump(2);
    BaseKVConfig kv_config;
    kv_config.num_threads_ = config["num_threads"].get<int>();
    kv_config.json_config_ = config["base_kv_config"];
    auto r                 = base::ResolveEngine(kv_config);
    base_kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        r.engine, r.cfg));
  }

  ~CachePS() {}

  bool Initialize(const std::vector<std::string>& model_config_path,
                  const std::vector<std::string>& emb_file_path) {
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() { base_kv_->clear(); }

  void LoadFakeData(int64_t key_capacity, int value_size) {
    base_kv_->LoadFakeData(key_capacity, value_size);
  }

  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path) {
    // base_kv_->loadCkpt();
    // LoadFakeData(KEY_CNT);
    return true;
  }

  void PutSingleParameter(
      const uint64_t key, const void* data, const int dim, const int tid) {
    base_kv_->Put(key, std::string_view((char*)data, dim * sizeof(float)), tid);
  }

  void PutDenseParameterBatch(
      const uint64_t* keys,
      const float* values,
      int key_count,
      int embedding_dim,
      int tid) {
    if (key_count <= 0 || embedding_dim <= 0) {
      return;
    }
    base::ConstArray<uint64_t> key_array(keys, key_count);
    std::vector<base::ConstArray<float>> value_slices;
    value_slices.reserve(static_cast<std::size_t>(key_count));
    for (int i = 0; i < key_count; ++i) {
      value_slices.emplace_back(values + i * embedding_dim, embedding_dim);
    }
    base_kv_->BatchPut(key_array, &value_slices, tid);
  }

  void PutSingleParameter(const ParameterCompressItem* item, int tid) {
    auto key = item->key;
    auto dim = item->dim;
    base_kv_->Put(
        key, std::string_view((char*)item->data(), dim * sizeof(float)), tid);
  }

  void PutParameter(coroutine<void>::push_type& sink,
                    const ParameterCompressReader* reader,
                    int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back(
          (float*)reader->item(i)->data(), reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(sink, keys, &values, tid);
  }

  void PutParameter(const ParameterCompressReader* reader, int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back(
          (float*)reader->item(i)->data(), reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(keys, &values, tid);
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack& pack, int tid) {
    std::vector<uint64_t> keys = {key};
    base::ConstArray<uint64_t> keys_array(keys);
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(keys_array, &values, tid);
    base::ConstArray<float> value = values[0];

    if (value.Size() == 0) {
      pack.key      = key;
      pack.dim      = 0;
      pack.emb_data = nullptr;
      FB_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key      = key;
    pack.dim      = value.Size();
    pack.emb_data = value.Data();
    // LOG(ERROR) << "Get key " << key << " dim " << pack.dim;
    return true;
  }

  bool GetParameterRun2Completion(base::ConstArray<uint64_t> keys,
                                  std::vector<ParameterPack>& packs,
                                  int tid) {
#ifdef ENABLE_PERF_REPORT
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    std::vector<base::ConstArray<float>> values;
    base_kv_->BatchGet(keys, &values, tid);

    for (int i = 0; i < keys.Size(); i++) {
      packs.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }

#ifdef ENABLE_PERF_REPORT
    auto end_time = std::chrono::high_resolution_clock::now();
    double start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();

    std::string report_id = "cache_ps::GetParameterRun2Completion|" +
                            std::to_string(static_cast<uint64_t>(start_us));

    report("embread_stages",
           report_id.c_str(),
           "duration_us",
           static_cast<double>(duration));

    report("embread_stages",
           report_id.c_str(),
           "request_size",
           static_cast<double>(keys.Size()));

    std::string unique_id =
        "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
    FlameGraphData fg_data = {
        "cache_ps::GetParameterRun2Completion",
        start_us,
        3, // level
        static_cast<double>(duration),
        static_cast<double>(duration)};
    report_flame_graph("emb_read_flame_map", unique_id.c_str(), fg_data);
#endif
    return true;
  }

  bool GetParameterFlat(base::ConstArray<uint64_t> keys,
                        float* values,
                        int64_t num_rows,
                        int64_t embedding_dim,
                        int tid) {
    if (values == nullptr) {
      LOG(ERROR) << "GetParameterFlat values pointer is null";
      return false;
    }
    if (num_rows < 0 || embedding_dim <= 0) {
      LOG(ERROR) << "GetParameterFlat invalid shape rows=" << num_rows
                 << " dim=" << embedding_dim;
      return false;
    }
    if (keys.Size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "GetParameterFlat keys size mismatch " << keys.Size()
                 << " vs " << num_rows;
      return false;
    }

    std::vector<base::ConstArray<float>> value_slices;
    base_kv_->BatchGet(keys, &value_slices, tid);
    if (value_slices.size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "GetParameterFlat BatchGet returned " << value_slices.size()
                 << " rows, expected " << num_rows;
      return false;
    }

    for (int64_t row = 0; row < num_rows; ++row) {
      const auto& slice = value_slices[static_cast<size_t>(row)];
      if (slice.Size() != 0 &&
          static_cast<int64_t>(slice.Size()) != embedding_dim) {
        LOG(ERROR) << "GetParameterFlat embedding_dim mismatch at row=" << row
                   << " key=" << keys[static_cast<size_t>(row)]
                   << " expected=" << embedding_dim
                   << " actual=" << slice.Size();
        return false;
      }
    }

    std::fill_n(values,
                static_cast<size_t>(num_rows) *
                    static_cast<size_t>(embedding_dim),
                0.0f);
    for (int64_t row = 0; row < num_rows; ++row) {
      const auto& slice = value_slices[static_cast<size_t>(row)];
      if (slice.Size() > 0) {
        std::copy_n(slice.Data(),
                    static_cast<int64_t>(slice.Size()),
                    values + row * embedding_dim);
      }
    }
    return true;
  }

  bool GetParameterRun2Completion(
      coroutine<void>::push_type& sink,
      base::ConstArray<uint64_t> keys,
      std::vector<ParameterPack>& pack,
      int tid) {
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(sink, keys, &values, tid);

    for (int i = 0; i < keys.Size(); i++) {
      pack.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }
    return true;
  }

  /// optimizer interface

  bool InitTable(const std::string& table_name,
                 uint64_t num_embeddings,
                 uint64_t embedding_dim) {
    if (!optimizer_) {
      // TODO: optimizer type from config
      optimizer_ = std::make_unique<SGD>(0.01);
    }

    EmbeddingTableConfig config{num_embeddings, embedding_dim};
    optimizer_->Init({table_name}, config, base_kv_.get());
    return true;
  }

  bool UpdateParameter(const std::string& table_name,
                       const ParameterCompressReader* reader,
                       unsigned tid) {
    if (!optimizer_) {
      LOG(ERROR) << "Optimizer not initialized. Please call InitTable first.";
      return false;
    }

    optimizer_->Update(table_name, reader, tid);
    return true;
  }

  bool UpdateParameterFlat(
      const std::string& table_name,
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid) {
    if (grads == nullptr) {
      LOG(ERROR) << "UpdateParameterFlat grads pointer is null";
      return false;
    }
    if (num_rows < 0 || embedding_dim <= 0) {
      LOG(ERROR) << "UpdateParameterFlat invalid shape rows=" << num_rows
                 << " dim=" << embedding_dim;
      return false;
    }
    if (keys.Size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "UpdateParameterFlat keys size mismatch " << keys.Size()
                 << " vs " << num_rows;
      return false;
    }
    if (!optimizer_) {
      LOG(ERROR) << "Optimizer not initialized. Please call InitTable first.";
      return false;
    }
    optimizer_->UpdateFlat(
        table_name, keys, grads, num_rows, embedding_dim, tid);
    return true;
  }

private:
  std::unique_ptr<BaseKV> base_kv_;
  std::unique_ptr<Optimizer> optimizer_;
  std::atomic<bool> stopFlag_{false};
};
