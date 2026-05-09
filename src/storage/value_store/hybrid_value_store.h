#pragma once

#include <atomic>
#include <memory>
#include <sstream>

#include "base/factory.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class HybridValueStore : public ValueStore {
public:
  explicit HybridValueStore(const BaseKVConfig& config)
      : dram_store_(BuildDramConfig(config)), ssd_store_(config) {
    const auto& v        = config.json_config_.at("value");
    const auto& dram     = v.at("dram_allocator");
    dram_capacity_bytes_ = dram.at("capacity_bytes").get<uint64_t>();
    const auto& tiering =
        v.contains("tiering") ? v.at("tiering") : json::object();
    high_watermark_ratio_ = tiering.value("high_watermark_ratio", 0.85);
  }

  uint64_t Alloc(size_t size) override {
    if (ShouldUseDram(size)) {
      const uint64_t raw = dram_store_.Alloc(size);
      if (raw != kValueHandleNone) {
        dram_bytes_reserved_.fetch_add(
            dram_store_.SlotCapacity(raw), std::memory_order_relaxed);
        return MakeDramHandle(raw);
      }
    }
    const uint64_t raw = ssd_store_.Alloc(size);
    return raw == kValueHandleNone ? kValueHandleNone : MakeSsdHandle(raw);
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      ssd_store_.Write(SsdRawHandle(handle), data, size);
      return;
    }
    dram_store_.Write(DramRawHandle(handle), data, size);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      return ssd_store_.Read(SsdRawHandle(handle), out_buf, buf_size);
    }
    return dram_store_.Read(DramRawHandle(handle), out_buf, buf_size);
  }

  void Free(uint64_t handle) override {
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      ssd_store_.Free(SsdRawHandle(handle));
      return;
    }
    const uint64_t raw = DramRawHandle(handle);
    const size_t cap   = dram_store_.SlotCapacity(raw);
    dram_store_.Free(raw);
    uint64_t cur = dram_bytes_reserved_.load(std::memory_order_relaxed);
    while (cur != 0 &&
           !dram_bytes_reserved_.compare_exchange_weak(
               cur, cur > cap ? cur - cap : 0, std::memory_order_relaxed)) {
    }
  }

  const char* DirectPtr(uint64_t handle) const override {
    if (handle == kValueHandleNone || IsOnSSD(handle)) {
      return nullptr;
    }
    return dram_store_.DirectPtr(DramRawHandle(handle));
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      return ssd_store_.SlotCapacity(SsdRawHandle(handle));
    }
    return dram_store_.SlotCapacity(DramRawHandle(handle));
  }

  std::string GetInfo() const override {
    std::ostringstream os;
    os << "HybridValueStore(dram_reserved="
       << dram_bytes_reserved_.load(std::memory_order_relaxed)
       << ", dram_capacity=" << dram_capacity_bytes_ << ", dram="
       << dram_store_.GetInfo() << ", ssd=" << ssd_store_.GetInfo() << ")";
    return os.str();
  }

private:
  static constexpr uint64_t kSsdFlag = 1ULL << 63;

  static bool IsOnSSD(uint64_t handle) { return (handle & kSsdFlag) != 0; }

  static uint64_t MakeDramHandle(uint64_t raw) { return raw; }

  static uint64_t MakeSsdHandle(uint64_t raw) { return raw | kSsdFlag; }

  static uint64_t DramRawHandle(uint64_t handle) { return handle; }

  static uint64_t SsdRawHandle(uint64_t handle) { return handle & ~kSsdFlag; }

  static BaseKVConfig BuildDramConfig(const BaseKVConfig& config) {
    BaseKVConfig out = config;
    auto& j          = out.json_config_;
    if (!j.contains("value") || !j.at("value").contains("dram_allocator")) {
      return out;
    }
    json value    = j.at("value");
    value["type"] = "DRAM_VALUE_STORE";
    value.erase("ssd_allocator");
    value.erase("tiering");
    j["value"] = value;
    return out;
  }

  bool ShouldUseDram(size_t size) const {
    if (dram_capacity_bytes_ == 0) {
      return false;
    }
    const double next =
        static_cast<double>(
            dram_bytes_reserved_.load(std::memory_order_relaxed) + size) /
        static_cast<double>(dram_capacity_bytes_);
    return next <= high_watermark_ratio_;
  }

  DramValueStore dram_store_;
  SsdValueStore ssd_store_;
  uint64_t dram_capacity_bytes_ = 0;
  double high_watermark_ratio_  = 0.85;
  std::atomic<uint64_t> dram_bytes_reserved_{0};
};

FACTORY_REGISTER(
    ValueStore, TIERED_VALUE_STORE, HybridValueStore, const BaseKVConfig&);
