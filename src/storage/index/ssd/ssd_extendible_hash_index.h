#pragma once

#include <memory>

#include "base/factory.h"
#include "storage/index/ssd/CCEH.h"

class SsdExtendibleHashIndex : public Index {
public:
  explicit SsdExtendibleHashIndex(const BaseKVConfig& config)
      : Index(config), impl_(std::make_unique<CCEH>(BuildConfig(config))) {}

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    impl_->Get(key, pointer, tid);
  }

  void Get(coroutine<void>::push_type& sink,
           int index,
           Key_t key,
           Value_t& pointer,
           unsigned tid) override {
    impl_->Get(sink, index, key, pointer, tid);
  }

  void Put(Key_t key, Value_t pointer, unsigned tid) override {
    impl_->Put(key, pointer, tid);
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    impl_->BatchGet(keys, pointers, tid);
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    impl_->BatchPut(keys, pointers, tid);
  }

  bool Delete(Key_t& key) override { return impl_->Delete(key); }

private:
  static BaseKVConfig BuildConfig(const BaseKVConfig& config) {
    BaseKVConfig out = config;
    auto& j          = out.json_config_;
    if (j.contains("index") && j.at("index").contains("io")) {
      const auto& io       = j.at("index").at("io");
      j["io_backend_type"] = io.value("type", "IOURING");
      j["queue_cnt"]       = io.value("queue_depth", 512);
      j["page_id_offset"] =
          io.value("base_offset_bytes", static_cast<uint64_t>(0)) / PAGE_SIZE;
      j["file_path"] = j.at("index").at("path").get<std::string>();
    }
    return out;
  }

  std::unique_ptr<CCEH> impl_;
};

FACTORY_REGISTER(Index, SSD, SsdExtendibleHashIndex, const BaseKVConfig&);
FACTORY_REGISTER(
    Index, SSD_EXTENDIBLE_HASH, SsdExtendibleHashIndex, const BaseKVConfig&);
