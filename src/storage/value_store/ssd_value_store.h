#pragma once

#include <memory>
#include <stdexcept>

#include "base/factory.h"
#include "storage/allocator/ssd/ssd_buddy_allocator.h"
#include "storage/allocator/ssd/ssd_slab_allocator.h"
#include "storage/io_backend/io_backend.h"
#include "storage/value_store/value_store.h"

class SsdValueStore : public ValueStore {
public:
  explicit SsdValueStore(const BaseKVConfig& config) {
    const auto& v = config.json_config_.at("value");
    if (!v.contains("path") || v.at("path").get<std::string>().empty()) {
      throw std::invalid_argument(
          "SsdValueStore requires non-empty value.path");
    }
    const auto& ssd = v.at("ssd_allocator");
    const auto& io  = ssd.at("io");

    BaseKVConfig io_cfg  = config;
    auto& j              = io_cfg.json_config_;
    j["io_backend_type"] = io.value("type", "IOURING");
    j["file_path"]       = v.at("path").get<std::string>();
    j["queue_cnt"]       = io.value("queue_depth", 512);
    j["page_id_offset"] =
        io.value("base_offset_bytes", static_cast<uint64_t>(0)) / PAGE_SIZE;

    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    io_backend_.reset(IOF::NewInstance(j.at("io_backend_type"), io_cfg));
    io_backend_->init();

    const std::string allocator_type = ssd.value("type", "SSD_SLAB");
    const uint64_t capacity_bytes    = ssd.at("capacity_bytes").get<uint64_t>();
    const uint64_t base_offset =
        ssd.value("base_offset_bytes", static_cast<uint64_t>(0));
    if (allocator_type == "SSD_SLAB") {
      std::vector<int> classes{128, 256, 512, 1024, 4096};
      if (ssd.contains("size_classes")) {
        classes = ssd.at("size_classes").get<std::vector<int>>();
      }
      allocator_.reset(new SsdSlabAllocator(
          io_backend_.get(), classes, capacity_bytes, base_offset));
    } else if (allocator_type == "SSD_BUDDY") {
      const int min_block = ssd.value("min_block_size", 128);
      const int max_block = ssd.value("max_block_size", 4096);
      int levels          = 1;
      for (int size = min_block; size < max_block; size <<= 1) {
        ++levels;
      }
      allocator_.reset(new SsdBuddyAllocator(
          io_backend_.get(), min_block, levels, capacity_bytes, base_offset));
    } else {
      throw std::invalid_argument("unknown SSD allocator: " + allocator_type);
    }
  }

  uint64_t Alloc(size_t size) override {
    const uint64_t handle = allocator_->Alloc(size);
    return handle == SsdBlockAllocator::kInvalidHandle
             ? kValueHandleNone
             : handle;
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    if (handle != kValueHandleNone) {
      allocator_->Write(handle, data, size);
    }
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = allocator_->AllocAndWrite(data, size);
    return handle == SsdBlockAllocator::kInvalidHandle
             ? kValueHandleNone
             : handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    return allocator_->Read(handle, out_buf, buf_size);
  }

  void Free(uint64_t handle) override {
    if (handle != kValueHandleNone) {
      allocator_->Free(handle);
    }
  }

  size_t SlotCapacity(uint64_t handle) const override {
    return allocator_->SlotCapacity(handle);
  }

  void BatchWrite(const std::vector<uint64_t>& handles,
                  const std::vector<WriteSpec>& specs) override {
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("SsdValueStore::BatchWrite size mismatch");
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        continue;
      }
      allocator_->Write(handles[i], specs[i].data, specs[i].size);
    }
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& out_results) override {
    out_results.resize(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
        continue;
      }
      out_results[i].data.resize(SlotCapacity(handles[i]));
      const size_t actual = Read(
          handles[i], out_results[i].data.data(), out_results[i].data.size());
      out_results[i].data.resize(actual);
    }
  }

private:
  std::unique_ptr<IOBackend> io_backend_;
  std::unique_ptr<SsdBlockAllocator> allocator_;
};

FACTORY_REGISTER(
    ValueStore, SSD_VALUE_STORE, SsdValueStore, const BaseKVConfig&);
