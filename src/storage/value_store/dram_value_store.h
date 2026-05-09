#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "base/factory.h"
#include "memory/allocators/allocator_factory.h"
#include "storage/value_store/value_store.h"

class DramValueStore : public ValueStore {
public:
  explicit DramValueStore(const BaseKVConfig& config) {
    const auto& j          = config.json_config_;
    const std::string path = j.at("path").get<std::string>() + "/value";
    if (!j.contains("value") || !j.at("value").contains("dram_allocator")) {
      throw std::invalid_argument(
          "DramValueStore requires value.dram_allocator");
    }
    const auto& dram                 = j.at("value").at("dram_allocator");
    const uint64_t capacity_bytes = dram.at("capacity_bytes").get<uint64_t>();
    allocator_ = base::allocators::CreateAllocator(
        dram, path, static_cast<int64>(capacity_bytes), "DRAM", "impl", "type");
    if (!allocator_) {
      throw std::runtime_error("failed to create DramValueStore allocator");
    }
    allocator_->Initialize();
  }

  uint64_t Alloc(size_t size) override {
    char* data = allocator_->New(static_cast<int>(size));
    if (data == nullptr) {
      return kValueHandleNone;
    }
    return EncodeOffset(allocator_->GetMallocOffset(data));
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    char* dst = Ptr(handle);
    if (dst == nullptr) {
      return;
    }
    std::memcpy(dst, data, size);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    char* src = Ptr(handle);
    if (src == nullptr || out_buf == nullptr) {
      return 0;
    }
    const size_t n = std::min(
        buf_size,
        static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle))));
    std::memcpy(out_buf, src, n);
    return n;
  }

  void Free(uint64_t handle) override {
    char* data = Ptr(handle);
    if (data != nullptr) {
      allocator_->Free(data);
    }
  }

  const char* DirectPtr(uint64_t handle) const override {
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    return static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle)));
  }

  std::string GetInfo() const override { return allocator_->GetInfo(); }
  uint64_t TotalAllocCount() const { return allocator_->total_malloc(); }

private:
  static uint64_t EncodeOffset(int64 offset) {
    return static_cast<uint64_t>(offset) + 1;
  }

  static int64 DecodeOffset(uint64_t handle) {
    return static_cast<int64>(handle - 1);
  }

  char* Ptr(uint64_t handle) const {
    if (handle == kValueHandleNone) {
      return nullptr;
    }
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  std::unique_ptr<base::MallocApi> allocator_;
};

FACTORY_REGISTER(
    ValueStore, DRAM_VALUE_STORE, DramValueStore, const BaseKVConfig&);
