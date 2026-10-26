#pragma once

#include <torch/custom_class.h>
#include <torch/extension.h>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace recstore::framework {

// CPU metadata only. The controller owns the bounded GPU values and orders
// these calls with PS commits; no embeddings or transport state live here.
class BagPipeCacheMetadata : public torch::CustomClassHolder {
 public:
  explicit BagPipeCacheMetadata(int64_t capacity) : capacity_(capacity) {
    TORCH_CHECK(capacity > 0, "BagPipe cache capacity must be positive");
  }

  int64_t size() const { return entries_.size(); }

  torch::Tensor previous(int64_t step, const torch::Tensor& local,
                         const torch::Tensor& active) {
    const auto* keys = Keys(local);
    const auto* updates = Keys(active);
    auto result = torch::empty_like(local);
    auto* out = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < local.numel(); ++i) {
      auto it = last_update_.find(keys[i]);
      out[i] = it == last_update_.end() ? -1 : it->second;
    }
    for (int64_t i = 0; i < active.numel(); ++i) last_update_[updates[i]] = step;
    return result;
  }

  void clear_plan() { last_update_.clear(); }

  torch::Tensor updated_ids() const {
    std::vector<int64_t> ids;
    ids.reserve(last_update_.size());
    for (const auto& entry : last_update_) ids.push_back(entry.first);
    return Tensor(ids);
  }

  torch::Tensor prefetch(const torch::Tensor& ids,
                         const torch::Tensor& previous, int64_t committed) const {
    const auto* keys = Keys(ids);
    const auto* versions = Keys(previous);
    TORCH_CHECK(ids.numel() == previous.numel(), "BagPipe version count mismatch");
    std::vector<int64_t> candidates;
    for (int64_t i = 0; i < ids.numel(); ++i) {
      if (versions[i] <= committed && !entries_.count(keys[i])) {
        candidates.push_back(keys[i]);
      }
    }
    return Tensor(candidates);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lookup(
      const torch::Tensor& ids) {
    const auto* keys = Keys(ids);
    std::vector<int64_t> positions, slots, missing;
    for (int64_t i = 0; i < ids.numel(); ++i) {
      auto it = entries_.find(keys[i]);
      if (it == entries_.end()) {
        missing.push_back(i);
      } else {
        positions.push_back(i);
        slots.push_back(it->second.slot);
        lru_.splice(lru_.end(), lru_, it->second.order);
      }
    }
    return {Tensor(positions), Tensor(slots), Tensor(missing)};
  }

  std::tuple<torch::Tensor, torch::Tensor, int64_t> commit(
      const torch::Tensor& active, const torch::Tensor& local,
      const torch::Tensor& exclusive, bool ps) {
    const auto* active_keys = Keys(active);
    const auto* local_keys = Keys(local);
    TORCH_CHECK(exclusive.device().is_cpu() && exclusive.dim() == 1 &&
                    exclusive.scalar_type() == torch::kBool &&
                    exclusive.is_contiguous() && exclusive.numel() == local.numel(),
                "BagPipe exclusive mask must match local IDs");
    int64_t invalidated = 0;
    if (ps) {
      std::unordered_set<int64_t> local_only;
      const auto* mask = exclusive.data_ptr<bool>();
      for (int64_t i = 0; i < local.numel(); ++i) {
        if (mask[i]) local_only.insert(local_keys[i]);
      }
      for (int64_t i = 0; i < active.numel(); ++i) {
        if (local_only.count(active_keys[i])) continue;
        auto it = entries_.find(active_keys[i]);
        if (it != entries_.end()) {
          free_.push_back(it->second.slot);
          lru_.erase(it->second.order);
          entries_.erase(it);
          ++invalidated;
        }
      }
    }
    const auto& ids = ps ? local : active;
    const auto* keys = ps ? local_keys : active_keys;
    std::vector<int64_t> slots, rows;
    for (int64_t i = 0; i < ids.numel(); ++i) {
      auto it = entries_.find(keys[i]);
      if (it != entries_.end()) {
        slots.push_back(it->second.slot);
        rows.push_back(i);
      }
    }
    return {Tensor(slots), Tensor(rows), invalidated};
  }

  std::tuple<torch::Tensor, torch::Tensor, int64_t> admit(
      const torch::Tensor& ids) {
    const auto* keys = Keys(ids);
    std::unordered_map<int64_t, int64_t> destinations;
    int64_t evicted = 0;
    for (int64_t i = 0; i < ids.numel(); ++i) {
      int64_t slot;
      auto it = entries_.find(keys[i]);
      if (it != entries_.end()) {
        slot = it->second.slot;
        lru_.splice(lru_.end(), lru_, it->second.order);
      } else {
        if (!free_.empty()) {
          slot = free_.back();
          free_.pop_back();
        } else if (next_slot_ < capacity_) {
          slot = next_slot_++;
        } else {
          auto victim = entries_.find(lru_.front());
          slot = victim->second.slot;
          entries_.erase(victim);
          lru_.pop_front();
          ++evicted;
        }
        lru_.push_back(keys[i]);
        entries_.emplace(keys[i], Entry{slot, std::prev(lru_.end())});
      }
      // Oversized batches may reuse a slot. Copy only its final row, avoiding
      // duplicate GPU index_copy destinations and preserving admission order.
      destinations[slot] = i;
    }
    std::vector<int64_t> slots, rows;
    for (const auto& [slot, row] : destinations) {
      slots.push_back(slot);
      rows.push_back(row);
    }
    return {Tensor(slots), Tensor(rows), evicted};
  }

 private:
  static const int64_t* Keys(const torch::Tensor& ids) {
    TORCH_CHECK(ids.device().is_cpu() && ids.dim() == 1 &&
                    ids.scalar_type() == torch::kInt64 && ids.is_contiguous(),
                "BagPipe metadata requires contiguous CPU int64 IDs");
    return ids.data_ptr<int64_t>();
  }

  static torch::Tensor Tensor(const std::vector<int64_t>& values) {
    auto result = torch::empty({static_cast<int64_t>(values.size())}, torch::kInt64);
    std::copy(values.begin(), values.end(), result.data_ptr<int64_t>());
    return result;
  }

  struct Entry {
    int64_t slot;
    std::list<int64_t>::iterator order;
  };
  int64_t capacity_, next_slot_ = 0;
  std::list<int64_t> lru_;
  std::unordered_map<int64_t, Entry> entries_;
  std::vector<int64_t> free_;
  std::unordered_map<int64_t, int64_t> last_update_;
};

}  // namespace recstore::framework
