#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <glog/logging.h>

namespace recstore::framework {

// Callbacks adapt buffers/devices; ordering, bypass and fallback belong here.
struct CacheLookupActions {
  bool usable = false;
  std::function<int64_t()> query;
  std::function<void()> read_missing, fill, scatter, read_all, clear;
};

struct CacheUpdateActions {
  std::function<bool()> enabled, can_invalidate;
  std::function<void()> invalidate, clear;
};

class CacheAwareOp {
public:
  bool BypassEnabled() const { return bypass_enabled_; }
  bool Bypassed() const { return bypassed_; }
  void Reset() {
    low_hit_streak_ = 0;
    bypassed_       = false;
  }
  void SetBypassEnabled(bool enabled) {
    bypass_enabled_ = enabled;
    if (!enabled)
      Reset();
  }

  static void ClearNoThrow(const std::function<void()>& clear) {
    try {
      if (clear)
        clear();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to clear cache: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to clear cache";
    }
  }

  void
  Lookup(int64_t rows, const CacheLookupActions& io, bool no_evict = false) {
    if (no_evict && !io.usable)
      throw std::runtime_error("No-eviction lookup requires an enabled GPU "
                               "cache on the keys' device");
    bool use_cache = io.usable;
    if (!no_evict && use_cache) {
      if (bypass_enabled_ && rows >= kMinRows && low_hit_streak_ >= 1) {
        if (!bypassed_)
          ClearNoThrow(io.clear);
        bypassed_ = true;
        use_cache = false;
      } else if (bypassed_) {
        ClearNoThrow(io.clear);
        Reset();
      }
    }
    if (use_cache) {
      try {
        const auto missing = io.query();
        if (missing < 0 || missing > rows)
          throw std::runtime_error("Invalid cache miss count");
        if (rows >= kMinRows) {
          if (double(rows - missing) / rows < 0.05)
            ++low_hit_streak_;
          else
            Reset();
        }
        if (missing) {
          io.read_missing();
          io.fill();
          io.scatter();
        }
        return;
      } catch (...) {
        if (no_evict)
          throw;
        LOG(WARNING)
            << "Cache lookup failed; clearing cache and reading from backend";
        ClearNoThrow(io.clear);
      }
    }
    io.read_all();
  }

  void AfterUpdate(int64_t rows,
                   const CacheUpdateActions& io,
                   bool force_invalidation = false) {
    if (!io.enabled || !io.enabled())
      return;
    // A controller with bypass disabled owns its cache coherence protocol.
    if (!force_invalidation &&
        (!bypass_enabled_ || (rows >= kMinRows && bypassed_)))
      return;
    if (io.can_invalidate && io.can_invalidate()) {
      try {
        io.invalidate();
        return;
      } catch (...) {
        LOG(WARNING) << "Cache invalidation failed after backend update";
      }
    }
    ClearNoThrow(io.clear);
  }

  void Update(int64_t rows,
              const CacheUpdateActions& io,
              const std::function<void()>& update) {
    try {
      update();
    } catch (...) {
      ClearNoThrow(io.clear);
      throw;
    }
    AfterUpdate(rows, io);
  }

private:
  static constexpr int64_t kMinRows = 1024;
  bool bypass_enabled_ = true, bypassed_ = false;
  int low_hit_streak_ = 0;
};

inline thread_local CacheAwareOp g_cache_aware_op;

// Opaque adapter callbacks retain the input buffers needed for invalidation.
// Transport completion must precede cache maintenance; failures clear the
// cache.
class PendingCacheUpdates {
public:
  void Add(uint64_t id, int64_t rows, CacheUpdateActions actions) {
    std::lock_guard<std::mutex> guard(mu_);
    if (!pending_.emplace(id, Pending{rows, std::move(actions)}).second)
      throw std::runtime_error("Duplicate asynchronous update handle");
  }
  void Wait(uint64_t id, const std::function<void()>& wait) {
    Pending pending;
    {
      std::lock_guard<std::mutex> guard(mu_);
      auto it = pending_.find(id);
      if (it == pending_.end())
        throw std::runtime_error("Unknown asynchronous update handle");
      pending = std::move(it->second);
      pending_.erase(it);
    }
    try {
      wait();
    } catch (...) {
      CacheAwareOp::ClearNoThrow(pending.actions.clear);
      throw;
    }
    g_cache_aware_op.AfterUpdate(pending.rows, pending.actions);
  }

private:
  struct Pending {
    int64_t rows = 0;
    CacheUpdateActions actions;
  };
  std::mutex mu_;
  std::unordered_map<uint64_t, Pending> pending_;
};

} // namespace recstore::framework
