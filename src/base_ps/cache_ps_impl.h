#pragma once
#include <folly/ProducerConsumerQueue.h>

#include <algorithm>
#include <atomic>
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <experimental/filesystem>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"  // NOLINT
#include "base/timer.h"
#include "parameters.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
using boost::coroutines2::coroutine;

static const int KEY_CNT = 12543670;

template <typename key_t>
struct TaskElement {
  TaskElement(const base::ConstArray<key_t> &keys,
              const base::MutableArray<ParameterPack> &packs,
              std::atomic_bool *promise)
      : keys(keys), packs(packs), promise(promise) {}

  TaskElement() {}

  base::ConstArray<key_t> keys;
  base::MutableArray<ParameterPack> packs;
  std::atomic_bool *promise;
};

class CachePS {
 public:
  using key_t = uint64_t;

  CachePS(json config) {
    LOG(INFO) << "cache ps config: " << config.dump(2);
    BaseKVConfig kv_config;
    kv_config.num_threads_ = config["num_threads"].get<int>();
    kv_config.json_config_ = config["base_kv_config"];
    auto r = base::ResolveEngine(kv_config);
    base_kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(r.engine, r.cfg));
  }

  ~CachePS() {}

  bool Initialize(const std::vector<std::string> &model_config_path,
                  const std::vector<std::string> &emb_file_path) {
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() { base_kv_->clear(); }

  void LoadFakeData(int64_t key_capacity, int value_size) {
    base_kv_->LoadFakeData(key_capacity, value_size);
  }

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path) {
    // base_kv_->loadCkpt();
    // LoadFakeData(KEY_CNT);
    return true;
  }

  void PutSingleParameter(const uint64_t key, const void *data, const int dim,
                          const int tid) {
    base_kv_->Put(key, std::string_view((char *)data, dim * sizeof(float)),
                  tid);
  }

  void PutSingleParameter(const ParameterCompressItem *item, int tid) {
    auto key = item->key;
    auto dim = item->dim;
    base_kv_->Put(
        key, std::string_view((char *)item->data(), dim * sizeof(float)), tid);
  }

  void PutParameter(coroutine<void>::push_type &sink,
                    const ParameterCompressReader *reader, int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back((float *)reader->item(i)->data(),
                          reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(sink, keys, &values, tid);
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack &pack, int tid) {
    std::vector<uint64_t> keys = {key};
    base::ConstArray<uint64_t> keys_array(keys);
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(keys_array, &values, tid);
    base::ConstArray<float> value = values[0];

    if (value.Size() == 0) {
      pack.key = key;
      pack.dim = 0;
      pack.emb_data = nullptr;
      FB_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key = key;
    pack.dim = value.Size();
    pack.emb_data = value.Data();
    // LOG(ERROR) << "Get key " << key << " dim " << pack.dim;
    return true;
  }

  bool GetParameterRun2Completion(coroutine<void>::push_type &sink,
                                  base::ConstArray<uint64_t> keys,
                                  std::vector<ParameterPack> &pack, int tid) {
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(sink, keys, &values, tid);

    for (int i = 0; i < keys.Size(); i++) {
      pack.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }

    return true;
  }

 private:
  std::unique_ptr<BaseKV> base_kv_;
  std::atomic<bool> stopFlag_{false};
};