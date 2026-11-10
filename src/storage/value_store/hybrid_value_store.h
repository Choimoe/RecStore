#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "base/factory.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/ssd_static_layout_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class HybridValueStore : public ValueStore {
public:
  explicit HybridValueStore(const BaseKVConfig& config)
      : dram_store_(BuildDramConfig(config)) {
    const auto& value = config.json_config_.at("value");
    ssd_store_.reset(
        base::Factory<ValueStore, const BaseKVConfig&>::NewInstance(
            SsdStoreType(value), BuildSsdConfig(config)));
    ssd_static_          = ssd_store_ && !ssd_store_->GetLayoutId().plain();
    const auto& v        = config.json_config_.at("value");
    const auto& dram     = v.at("dram_allocator");
    dram_capacity_bytes_ = dram.at("capacity_bytes").get<uint64_t>();
    const auto& tiering =
        v.contains("tiering") ? v.at("tiering") : json::object();
    high_watermark_ratio_ = tiering.value("high_watermark_ratio", 0.85);
    if (!std::isfinite(high_watermark_ratio_) || high_watermark_ratio_ < 0 ||
        high_watermark_ratio_ > 1) {
      throw std::invalid_argument("tiered high_watermark_ratio must be in [0, 1]");
    }
    const std::string profile_path = tiering.value("dram_keys_path", "");
    if (!profile_path.empty()) {
      std::ifstream input(profile_path, std::ios::binary | std::ios::ate);
      if (!input) {
        throw std::invalid_argument("cannot open tiered DRAM key profile");
      }
      const std::streamoff bytes = input.tellg();
      if (bytes < 0 || bytes % sizeof(uint64_t) != 0) {
        throw std::invalid_argument("invalid tiered DRAM key profile size");
      }
      dram_profile_.resize(static_cast<size_t>(bytes) / sizeof(uint64_t));
      input.seekg(0);
      if (!input.read(reinterpret_cast<char*>(dram_profile_.data()), bytes) ||
          !std::is_sorted(dram_profile_.begin(), dram_profile_.end()) ||
          std::adjacent_find(dram_profile_.begin(), dram_profile_.end()) !=
              dram_profile_.end()) {
        throw std::invalid_argument("tiered DRAM key profile must be sorted and unique");
      }
      profiled_placement_ = true;
    }
    online_placement_ = tiering.value("online_admission", false);
    if (online_placement_) {
      if (!profiled_placement_ || ssd_static_ || dram_profile_.empty()) {
        throw std::invalid_argument(
            "online admission requires a nonempty DRAM profile and plain SSD layout");
      }
      online_dram_keys_.reserve(dram_profile_.size());
      online_dram_keys_.insert(dram_profile_.begin(), dram_profile_.end());
    }
  }

  bool OnlinePlacementEnabled() const { return online_placement_; }
  bool DramKeyEligible(uint64_t key) const { return KeyUsesDram(key); }
  const std::vector<uint64_t>& DramProfileKeys() const { return dram_profile_; }

  // Private destinations are not visible through the index or eligibility set.
  // The caller publishes them only after validating its source versions.
  std::vector<uint64_t> PrepareExchange(
      const std::vector<WriteSpec>& victims,
      const std::vector<WriteSpec>& hot) {
    if (!online_placement_ || victims.size() != hot.size()) {
      throw std::invalid_argument("invalid prepared tier exchange");
    }
    const size_t count = hot.size();
    std::vector<uint64_t> handles(count * 2, kValueHandleNone);
    try {
      for (size_t i = 0; i < count; ++i) {
        const uint64_t raw = TryAllocDram(hot[i].size);
        if (raw == kValueHandleNone) {
          for (uint64_t handle : handles) Free(handle);
          return {};
        }
        handles[count + i] = MakeDramHandle(raw);
        dram_store_.Write(raw, hot[i].data, hot[i].size);
      }
      const auto cold = ssd_store_->BatchAllocAndWrite(victims);
      if (cold.size() != count) throw std::runtime_error("prepared SSD count mismatch");
      bool valid = true;
      for (size_t i = 0; i < count; ++i) {
        if (cold[i] == kValueHandleNone) { valid = false; continue; }
        handles[i] = MakeSsdHandle(cold[i]);
        ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
      }
      if (!valid) {
        for (uint64_t handle : handles) Free(handle);
        return {};
      }
    } catch (...) {
      for (uint64_t handle : handles) Free(handle);
      throw;
    }
    return handles;
  }

  // The engine must exclude readers and writers while changing eligibility.
  // Reuse nodes so a validated, fixed-size exchange needs no new allocation.
  void ExchangeDramKeys(const std::vector<uint64_t>& from,
                        const std::vector<uint64_t>& to) {
    if (!online_placement_ || from.size() != to.size()) {
      throw std::invalid_argument("invalid online DRAM key exchange");
    }
    std::unordered_set<uint64_t> seen_from, seen_to;
    for (size_t i = 0; i < from.size(); ++i) {
      if (!seen_from.insert(from[i]).second ||
          !seen_to.insert(to[i]).second ||
          online_dram_keys_.count(from[i]) == 0 ||
          online_dram_keys_.count(to[i]) != 0) {
        throw std::invalid_argument("online DRAM key exchange membership mismatch");
      }
    }
    for (size_t i = 0; i < from.size(); ++i) {
      auto node = online_dram_keys_.extract(from[i]);
      node.value() = to[i];
      online_dram_keys_.insert(std::move(node));
    }
  }

  uint64_t Alloc(size_t size) override {
    return AllocPlain(size, !profiled_placement_);
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
      if (profiled_placement_) {
        const uint64_t handle = AllocPlain(size, KeyUsesDram(key));
        if (handle != kValueHandleNone) {
          Write(handle, data, size);
        }
        return handle;
      }
      return AllocAndWrite(data, size);
    }
    const uint64_t handle = PlaceStatic(key, size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteSpec>& specs) override {
    return BatchAllocAndWritePlain(nullptr, specs);
  }

  std::vector<uint64_t> BatchAllocAndWrite(
      const std::vector<uint64_t>& keys,
      const std::vector<WriteSpec>& specs) override {
    if (keys.size() != specs.size()) {
      throw std::invalid_argument("tiered batch key/spec size mismatch");
    }
    if (!ssd_static_) {
      return BatchAllocAndWritePlain(keys.data(), specs);
    }
    std::vector<uint64_t> handles;
    handles.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      handles.push_back(PlaceStatic(keys[i], specs[i].size));
    }
    BatchWrite(handles, specs);
    return handles;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      ssd_read_rows_.fetch_add(1, std::memory_order_relaxed);
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
    return online_placement_ ? nullptr : dram_store_.RDMABackingData();
  }

  size_t RDMABackingSize() const override {
    return online_placement_ ? 0 : dram_store_.RDMABackingSize();
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
    ssd_read_rows_.fetch_add(ssd_handles.size(), std::memory_order_relaxed);
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

  void ResetStats() override {
    ssd_read_rows_.store(0, std::memory_order_relaxed);
    if (ssd_store_) {
      ssd_store_->ResetStats();
    }
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
       << total << " tiered_ssd_total_ratio=" << total_ratio
       << " tiered_profile_enabled=" << profiled_placement_
       << " tiered_dram_profile_keys=" << dram_profile_.size()
       << " tiered_dram_profile_bytes=" << dram_profile_.size() * sizeof(uint64_t)
       << " tiered_online_enabled=" << online_placement_
       << " tiered_online_dram_keys=" << online_dram_keys_.size()
       << " tiered_ssd_read_rows="
       << ssd_read_rows_.load(std::memory_order_relaxed);
    return os.str() + ssd_store_->ExtraResultFields();
  }

  ValueStore::LayoutId GetLayoutId() const override {
    return ssd_store_ ? ssd_store_->GetLayoutId() : ValueStore::LayoutId{};
  }

private:
  bool KeyUsesDram(uint64_t key) const {
    if (online_placement_) {
      return online_dram_keys_.count(key) != 0;
    }
    return !profiled_placement_ ||
           std::binary_search(dram_profile_.begin(), dram_profile_.end(), key);
  }

  uint64_t AllocPlain(size_t size, bool use_dram) {
    const uint64_t dram = use_dram ? TryAllocDram(size) : kValueHandleNone;
    if (dram != kValueHandleNone) {
      return MakeDramHandle(dram);
    }
    const uint64_t raw = ssd_store_->Alloc(size);
    if (raw == kValueHandleNone) {
      return kValueHandleNone;
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(raw);
  }
  std::vector<uint64_t> BatchAllocAndWritePlain(
      const uint64_t* keys, const std::vector<WriteSpec>& specs) {
    std::vector<uint64_t> handles(specs.size(), kValueHandleNone);
    std::vector<WriteSpec> ssd_specs;
    std::vector<size_t> ssd_indices;
    try {
      for (size_t i = 0; i < specs.size(); ++i) {
        const bool use_dram =
            !profiled_placement_ || (keys != nullptr && KeyUsesDram(keys[i]));
        const uint64_t raw =
            use_dram ? TryAllocDram(specs[i].size) : kValueHandleNone;
        if (raw != kValueHandleNone) {
          handles[i] = MakeDramHandle(raw);
          dram_store_.Write(raw, specs[i].data, specs[i].size);
        } else {
          ssd_specs.push_back(specs[i]);
          ssd_indices.push_back(i);
        }
      }
      if (!ssd_specs.empty()) {
        const auto ssd_handles = ssd_store_->BatchAllocAndWrite(ssd_specs);
        if (ssd_handles.size() != ssd_indices.size()) {
          throw std::runtime_error("tiered batch allocation result size mismatch");
        }
        for (size_t i = 0; i < ssd_indices.size(); ++i) {
          if (ssd_handles[i] != kValueHandleNone) {
            handles[ssd_indices[i]] = MakeSsdHandle(ssd_handles[i]);
            ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
            ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    } catch (...) {
      for (uint64_t handle : handles) Free(handle);
      throw;
    }
    return handles;
  }

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
    const std::string strategy =
        value.at("ssd_layout").value("strategy", "plain");
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

  uint64_t TryAllocDram(size_t size) {
    const uint64_t limit = static_cast<uint64_t>(
        static_cast<long double>(dram_capacity_bytes_) * high_watermark_ratio_);
    uint64_t current = dram_bytes_reserved_.load(std::memory_order_relaxed);
    if (current >= limit || size > limit - current) {
      return kValueHandleNone;
    }
    const uint64_t raw = dram_store_.Alloc(size);
    if (raw == kValueHandleNone) {
      return raw;
    }
    const size_t capacity = dram_store_.SlotCapacity(raw);
    // Reserve the actual slot size atomically: concurrent load workers must
    // not all pass the watermark check and overfill the DRAM tier.
    for (;;) {
      if (current > limit || capacity > limit - current) {
        dram_store_.Free(raw);
        return kValueHandleNone;
      }
      if (dram_bytes_reserved_.compare_exchange_weak(
              current, current + capacity, std::memory_order_relaxed)) {
        dram_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        dram_total_allocs_.fetch_add(1, std::memory_order_relaxed);
        return raw;
      }
    }
  }

  uint64_t PlaceStatic(uint64_t key, size_t size) {
    if (key > ssd_layout::kMaxKey) {
      throw std::invalid_argument("static SSD key is out of range");
    }
    const size_t row_size = ssd_store_->SlotCapacity(key + 1);
    if (size != row_size) {
      throw std::invalid_argument("static SSD layout value size mismatch");
    }
    const uint64_t raw = KeyUsesDram(key) ? TryAllocDram(size) : kValueHandleNone;
    if (raw != kValueHandleNone) {
      return MakeDramHandle(raw);
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(key + 1);
  }

  DramValueStore dram_store_;
  std::unique_ptr<ValueStore> ssd_store_;
  bool ssd_static_              = false;
  bool profiled_placement_      = false;
  bool online_placement_        = false;
  std::vector<uint64_t> dram_profile_;
  std::unordered_set<uint64_t> online_dram_keys_;
  uint64_t dram_capacity_bytes_ = 0;
  double high_watermark_ratio_  = 0.85;
  std::atomic<uint64_t> dram_bytes_reserved_{0};
  std::atomic<uint64_t> dram_live_allocs_{0};
  std::atomic<uint64_t> ssd_live_allocs_{0};
  std::atomic<uint64_t> dram_total_allocs_{0};
  std::atomic<uint64_t> ssd_total_allocs_{0};
  std::atomic<uint64_t> ssd_read_rows_{0};
};

FACTORY_REGISTER(
    ValueStore, TIERED_VALUE_STORE, HybridValueStore, const BaseKVConfig&);
