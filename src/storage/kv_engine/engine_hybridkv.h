#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../dram/extendible_hash.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"

class KVEngineHybrid : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineHybrid(const BaseKVConfig &config)
      : BaseKV(config),
      valm(config.json_config_.at("path").get<std::string>()+"/value",
                    config.json_config_.at("capacity").get<size_t>())
  {
    // value_size_ = config.json_config_.at("value_size").get<int>();

    // 初始化extendible hash表
    IndexConfig index_config;
    index_config.json_config_ = config.json_config_;
    index = new ExtendibleHash(index_config);
  }

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    // TODO:
    // 1. get <key, pointer> from index
    // 2. get value length from value header
    // 2. get value from pointer and length
    std::shared_lock<std::shared_mutex> lock(lock_);
    uint64_t pointer;
    index->Get(key, pointer, tid);
    UnifiedPointer p = UnifiedPointer::FromRaw(pointer);
    value = valm.RetrieveValue(p);

  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    UnifiedPointer p = valm.WriteValue(value);
    uint64_t value_put = p.RawValue();
    index->Put(key, value_put, tid);

  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>> *values,
                unsigned tid) {
     values->clear();
    std::shared_lock<std::shared_mutex> _(lock_);
      // 新增：持久化存储float数据
    storage.reserve(keys.Size());             // 预分配空间
    values->reserve(keys.Size());             // 预分配空间避免多次扩容

    for (int k = 0; k < keys.Size(); k++) {
        uint64_t pointer;
        index->Get(keys[k], pointer, tid);
        std::string temp_values = valm.RetrieveValue(UnifiedPointer::FromRaw(pointer));
        // 处理空字符串情况
        if (temp_values.empty()) {
            // 创建空向量并添加到持久化存储
            storage.push_back(std::vector<float>());
            values->push_back(base::ConstArray<float>(
                nullptr,  // 空指针
                0         // 大小为0
            ));
            continue;
        }
        // 将字符串数据转换为float并持久保存
        else{
          std::vector<float> floatData;
          floatData.reserve(temp_values.size());
          for (char c : temp_values) {
              floatData.push_back(static_cast<float>(static_cast<unsigned char>(c)));
              // std::cout<<c<<' ';
          }
          // for(int i=0;i<floatData.size();i++){
          //   std::cout<<floatData[i]<<' ';
          // }
          // std::cout<<std::endl;
          // 将float数组存入持久化存储
          storage.push_back(std::move(floatData));
          // 从storage中引用数据构造ConstArray
          values->push_back(base::ConstArray<float>(storage.back()));
        }
        // for(int i=0;i<(*values)[k].Size();i++){
        //   std::cout<<(*values)[k][i]<<' ';
        // }
        // 调试输出（直接使用storage中的数据）
        // std::cout << storage.back().data() << "===" << storage.back().size() << std::endl;
    }
    // std::cout<<"batchgetsuccess"<<std::endl;
  }

  ~KVEngineHybrid() {
    std::cout << "exit KVEngineHybrid" << std::endl;
    for(int i=0;i<storage.size();i++){
      storage[i].clear();
    }  
  }

private:
  Index *index;
  ValueManager valm;
  mutable std::shared_mutex lock_;
  std::vector<std::vector<float>> storage;
  
  uint64_t counter = 0;
  std::string dict_pool_name_;
  size_t dict_pool_size_;
  // int value_size_;
  // base::PersistLoopShmMalloc shm_malloc_;
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid,
                 const BaseKVConfig &);