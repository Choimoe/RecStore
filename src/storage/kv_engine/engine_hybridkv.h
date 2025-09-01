#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../dram/extendible_hash.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"
#include "src/storage/hybrid/value.h"

class KVEngineHybrid : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineHybrid(const BaseKVConfig &config)
      : BaseKV(config),valm( 
          config.json_config_.at("path").get<std::string>() + "/shmvalue",
          config.json_config_.at("shmcapacity").get<size_t>(),
          config.json_config_.at("path").get<std::string>() + "/ssdvalue",
          config.json_config_.at("ssdcapacity").get<size_t>(),
          [&] {
              IndexConfig ic;
              ic.json_config_ = config.json_config_;
              return ic;
          }()
      )
  {}

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    value = valm.RetrieveValue(key,tid);

  }

  void GetIndex(uint64_t key, uint64_t &pointer, 
            unsigned int tid){
    valm.index->Get(key,pointer,tid);
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    valm.WriteValue(key,value,0);
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
        std::string temp_values;
        temp_values = valm.RetrieveValue(keys[k],0);
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
        else{
          std::vector<float> floatData;
          floatData.reserve(temp_values.size());
          for (char c : temp_values) {
              floatData.push_back(static_cast<float>(static_cast<unsigned char>(c)));
              // std::cout<<c<<' ';
          }
          // 将float数组存入持久化存储
          storage.push_back(std::move(floatData));
          // 从storage中引用数据构造ConstArray
          values->push_back(base::ConstArray<float>(storage.back()));
        }
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
  
  ValueManager valm;
  mutable std::shared_mutex lock_;
  std::vector<std::vector<float>> storage;
  
  uint64_t counter = 0;
  std::string dict_pool_name_;
  size_t dict_pool_size_;
  
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid,
                 const BaseKVConfig &);