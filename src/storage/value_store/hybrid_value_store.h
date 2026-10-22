#pragma once

#include <atomic>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "base/factory.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/ssd_static_layout_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class HybridValueStore : public ValueStore {
public:
  explicit HybridValueStore(const BaseKVConfig& config)
      : dram_store_(BuildDramConfig(config)) {
    const auto& value = config.json_config_.at("value");
    ssd_store_.reset(base::Factory<ValueStore, const BaseKVConfig&>::NewInstance(
        SsdStoreType(value), BuildSsdConfig(config)));
    ssd_static_ = ssd_store_ && !ssd_store_->GetLayoutId().plain();
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
        dram_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        dram_total_allocs_.fetch_add(1, std::memory_order_relaxed);
        return MakeDramHandle(raw);
      }
    }
    const uint64_t raw = ssd_store_->Alloc(size);
    if (raw == kValueHandleNone) {
      return kValueHandleNone;
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(raw);
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      ssd_store_->Write(SsdRawHandle(handle), data, size);
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

  uint64_t AllocAndWrite(uint64_t key, const void* data, size_t size) override {
    if (!ssd_static_) {
      return AllocAndWrite(data, size);
    }
    const uint64_t handle = PlaceStatic(key, size);
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
      return ssd_store_->Read(SsdRawHandle(handle), out_buf, buf_size);
    }
    return dram_store_.Read(DramRawHandle(handle), out_buf, buf_size);
  }

  void Free(uint64_t handle) override {
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      ssd_store_->Free(SsdRawHandle(handle));
      ssd_live_allocs_.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
    const uint64_t raw = DramRawHandle(handle);
    const size_t cap   = dram_store_.SlotCapacity(raw);
    dram_store_.Free(raw);
    dram_live_allocs_.fetch_sub(1, std::memory_order_relaxed);
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

  char* RDMABackingData() const override {
    return dram_store_.RDMABackingData();
  }

  size_t RDMABackingSize() const override {
    return dram_store_.RDMABackingSize();
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      return ssd_store_->SlotCapacity(SsdRawHandle(handle));
    }
    return dram_store_.SlotCapacity(DramRawHandle(handle));
  }

  void BatchWrite(const std::vector<uint64_t>& handles,
                  const std::vector<WriteSpec>& specs) override {
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("HybridValueStore::BatchWrite size mismatch");
    }
    std::vector<uint64_t> dram_handles;
    std::vector<WriteSpec> dram_specs;
    std::vector<uint64_t> ssd_handles;
    std::vector<WriteSpec> ssd_specs;
    dram_handles.reserve(handles.size());
    dram_specs.reserve(handles.size());
    ssd_handles.reserve(handles.size());
    ssd_specs.reserve(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        continue;
      }
      if (IsOnSSD(handles[i])) {
        ssd_handles.push_back(SsdRawHandle(handles[i]));
        ssd_specs.push_back(specs[i]);
      } else {
        dram_handles.push_back(DramRawHandle(handles[i]));
        dram_specs.push_back(specs[i]);
      }
    }
    if (!dram_handles.empty()) {
      dram_store_.BatchWrite(dram_handles, dram_specs);
    }
    if (!ssd_handles.empty()) {
      ssd_store_->BatchWrite(ssd_handles, ssd_specs);
    }
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& out_results) override {
    out_results.clear();
    out_results.resize(handles.size());

    std::vector<uint64_t> ssd_handles;
    std::vector<size_t> ssd_indices;
    ssd_handles.reserve(handles.size());
    ssd_indices.reserve(handles.size());

    for (size_t i = 0; i < handles.size(); ++i) {
      const uint64_t handle = handles[i];
      if (handle == kValueHandleNone) {
        continue;
      }
      if (IsOnSSD(handle)) {
        ssd_handles.push_back(SsdRawHandle(handle));
        ssd_indices.push_back(i);
        continue;
      }

      const uint64_t raw = DramRawHandle(handle);
      out_results[i].data.resize(dram_store_.SlotCapacity(raw));
      const size_t actual = dram_store_.Read(
          raw, out_results[i].data.data(), out_results[i].data.size());
      out_results[i].data.resize(actual);
    }

    if (ssd_handles.empty()) {
      return;
    }

    std::vector<ReadResult> ssd_results;
    ssd_store_->BatchRead(ssd_handles, ssd_results);
    if (ssd_results.size() != ssd_indices.size()) {
      throw std::runtime_error(
          "HybridValueStore::BatchRead result size mismatch");
    }
    for (size_t i = 0; i < ssd_indices.size(); ++i) {
      out_results[ssd_indices[i]] = std::move(ssd_results[i]);
    }
  }

  std::string GetInfo() const override {
    std::ostringstream os;
    os << "HybridValueStore(dram_reserved="
       << dram_bytes_reserved_.load(std::memory_order_relaxed)
       << ", dram_capacity=" << dram_capacity_bytes_ << ", dram="
       << dram_store_.GetInfo() << ", ssd=" << ssd_store_->GetInfo() << ")";
    return os.str();
  }

  std::string ExtraResultFields() const override {
    const uint64_t dram_live =
        dram_live_allocs_.load(std::memory_order_relaxed);
    const uint64_t ssd_live = ssd_live_allocs_.load(std::memory_order_relaxed);
    const uint64_t live     = dram_live + ssd_live;
    const uint64_t dram_total =
        dram_total_allocs_.load(std::memory_order_relaxed);
    const uint64_t ssd_total =
        ssd_total_allocs_.load(std::memory_order_relaxed);
    const uint64_t total = dram_total + ssd_total;
    const double live_ratio =
        live == 0 ? 0.0 : static_cast<double>(ssd_live) / live;
    const double total_ratio =
        total == 0 ? 0.0 : static_cast<double>(ssd_total) / total;
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << " tiered_dram_capacity_bytes=" << dram_capacity_bytes_
       << " tiered_high_watermark_ratio=" << high_watermark_ratio_
       << " tiered_dram_live_allocs=" << dram_live << " tiered_ssd_live_allocs="
       << ssd_live << " tiered_live_allocs=" << live
       << " tiered_ssd_live_ratio=" << live_ratio
       << " tiered_dram_total_allocs=" << dram_total
       << " tiered_ssd_total_allocs=" << ssd_total << " tiered_total_allocs="
       << total << " tiered_ssd_total_ratio=" << total_ratio;
    return os.str() + ssd_store_->ExtraResultFields();
  }

  ValueStore::LayoutId GetLayoutId() const override {
    return ssd_store_ ? ssd_store_->GetLayoutId() : ValueStore::LayoutId{};
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
    if (value.at("dram_allocator").contains("path")) {
      value["path"] = value.at("dram_allocator").at("path");
    }
    value.erase("ssd_allocator");
    value.erase("ssd_layout");
    value.erase("tiering");
    j["value"] = value;
    return out;
  }

  static std::string SsdStoreType(const json& value) {
    if (!value.contains("ssd_layout")) {
      return "SSD_VALUE_STORE";
    }
    const std::string strategy = value.at("ssd_layout").value("strategy", "plain");
    if (strategy.empty() || strategy == "plain") {
      return "SSD_VALUE_STORE";
    }
    if (strategy == "bandana" || strategy == "maxembed") {
      return "SSD_STATIC_LAYOUT_VALUE_STORE";
    }
    throw std::invalid_argument("unknown SSD layout strategy: " + strategy);
  }

  static BaseKVConfig BuildSsdConfig(const BaseKVConfig& config) {
    BaseKVConfig out = config;
    auto& j          = out.json_config_;
    if (!j.contains("value") || !j.at("value").contains("ssd_allocator")) {
      return out;
    }
    json value    = j.at("value");
    value["type"] = SsdStoreType(value);
    if (value.at("ssd_allocator").contains("path")) {
      value["path"] = value.at("ssd_allocator").at("path");
    }
    value.erase("dram_allocator");
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

  uint64_t PlaceStatic(uint64_t key, size_t size) {
    if (key > ssd_layout::kMaxKey) {
      throw std::invalid_argument("static SSD key is out of range");
    }
    if (ShouldUseDram(size)) {
      const uint64_t raw = dram_store_.Alloc(size);
      if (raw != kValueHandleNone) {
        dram_bytes_reserved_.fetch_add(
            dram_store_.SlotCapacity(raw), std::memory_order_relaxed);
        dram_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        dram_total_allocs_.fetch_add(1, std::memory_order_relaxed);
        return MakeDramHandle(raw);
      }
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(key + 1);
  }

  DramValueStore dram_store_;
  std::unique_ptr<ValueStore> ssd_store_;
  bool ssd_static_           = false;
  uint64_t dram_capacity_bytes_ = 0;
  double high_watermark_ratio_  = 0.85;
  std::atomic<uint64_t> dram_bytes_reserved_{0};
  std::atomic<uint64_t> dram_live_allocs_{0};
  std::atomic<uint64_t> ssd_live_allocs_{0};
  std::atomic<uint64_t> dram_total_allocs_{0};
  std::atomic<uint64_t> ssd_total_allocs_{0};
};

FACTORY_REGISTER(
    ValueStore, TIERED_VALUE_STORE, HybridValueStore, const BaseKVConfig&);
