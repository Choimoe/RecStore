#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace recstore::optimization::bagpipe {

struct CacheLookup {
  std::vector<int64_t> hit_positions, hit_slots, missing_positions;
};

struct CacheCommit {
  std::vector<int64_t> slots, rows;
  int64_t invalidated = 0;
};

struct CacheAdmission {
  std::vector<int64_t> slots, rows;
  int64_t evicted = 0;
};

// Framework-independent cache policy and update dependencies. Values and
// transport state belong to the caller, which serializes calls with commits.
// Input arrays are borrowed for the duration of a call; empty arrays may be
// null.
class CacheMetadata {
public:
  explicit CacheMetadata(int64_t capacity) : capacity_(capacity) {
    if (capacity <= 0)
      throw std::invalid_argument("BagPipe cache capacity must be positive");
  }
  CacheMetadata(const CacheMetadata&)            = delete;
  CacheMetadata& operator=(const CacheMetadata&) = delete;

  int64_t Size() const { return entries_.size(); }

  std::vector<int64_t>
  Previous(int64_t step,
           const int64_t* local,
           size_t local_count,
           const int64_t* active,
           size_t active_count) {
    std::vector<int64_t> previous(local_count);
    for (size_t i = 0; i < local_count; ++i) {
      auto it     = last_update_.find(local[i]);
      previous[i] = it == last_update_.end() ? -1 : it->second;
    }
    for (size_t i = 0; i < active_count; ++i)
      last_update_[active[i]] = step;
    return previous;
  }

  void ClearPlan() { last_update_.clear(); }

  std::vector<int64_t> UpdatedIds() const {
    std::vector<int64_t> ids;
    ids.reserve(last_update_.size());
    for (const auto& entry : last_update_)
      ids.push_back(entry.first);
    return ids;
  }

  std::vector<int64_t>
  Prefetch(const int64_t* ids,
           const int64_t* previous,
           size_t count,
           int64_t committed) const {
    std::vector<int64_t> candidates;
    for (size_t i = 0; i < count; ++i) {
      if (previous[i] <= committed && !entries_.count(ids[i])) {
        candidates.push_back(ids[i]);
      }
    }
    return candidates;
  }

  CacheLookup Lookup(const int64_t* ids, size_t count) {
    CacheLookup result;
    for (size_t i = 0; i < count; ++i) {
      auto it = entries_.find(ids[i]);
      if (it == entries_.end()) {
        result.missing_positions.push_back(i);
      } else {
        result.hit_positions.push_back(i);
        result.hit_slots.push_back(it->second.slot);
        lru_.splice(lru_.end(), lru_, it->second.order);
      }
    }
    return result;
  }

  CacheCommit
  Commit(const int64_t* active,
         size_t active_count,
         const int64_t* local,
         size_t local_count,
         const bool* exclusive,
         bool ps_aggregation) {
    CacheCommit result;
    if (ps_aggregation) {
      std::unordered_set<int64_t> local_only;
      for (size_t i = 0; i < local_count; ++i) {
        if (exclusive[i])
          local_only.insert(local[i]);
      }
      for (size_t i = 0; i < active_count; ++i) {
        if (local_only.count(active[i]))
          continue;
        auto it = entries_.find(active[i]);
        if (it != entries_.end()) {
          free_.push_back(it->second.slot);
          lru_.erase(it->second.order);
          entries_.erase(it);
          ++result.invalidated;
        }
      }
    }
    const auto* ids  = ps_aggregation ? local : active;
    const auto count = ps_aggregation ? local_count : active_count;
    for (size_t i = 0; i < count; ++i) {
      auto it = entries_.find(ids[i]);
      if (it != entries_.end()) {
        result.slots.push_back(it->second.slot);
        result.rows.push_back(i);
      }
    }
    return result;
  }

  CacheAdmission Admit(const int64_t* ids, size_t count) {
    CacheAdmission result;
    std::unordered_map<int64_t, int64_t> destinations;
    for (size_t i = 0; i < count; ++i) {
      int64_t slot;
      auto it = entries_.find(ids[i]);
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
          slot        = victim->second.slot;
          entries_.erase(victim);
          lru_.pop_front();
          ++result.evicted;
        }
        lru_.push_back(ids[i]);
        entries_.emplace(ids[i], Entry{slot, std::prev(lru_.end())});
      }
      // An oversized batch can reuse slots. Return only the final source row
      // per slot so the caller never writes duplicate scatter destinations.
      destinations[slot] = i;
    }
    for (const auto& [slot, row] : destinations) {
      result.slots.push_back(slot);
      result.rows.push_back(row);
    }
    return result;
  }

private:
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

} // namespace recstore::optimization::bagpipe
