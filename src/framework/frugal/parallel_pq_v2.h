#pragma once
#include <array>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "base/lock.h"
#include "memory/epoch_manager.h"
#include "memory/malloc.h"

#define PPQ_ALL_SCAN
// #define PPQ_SELECTIVE_SCAN

namespace recstore {

template <typename T>
class HashMapPool {
  using index_type = std::unordered_map<int64_t, T*>;

public:
  HashMapPool(int pool_size, int64_t capacity) {
    for (int i = 0; i < pool_size; i++) {
      pool_.push_back(new index_type());
      pool_.back()->reserve(capacity);
    }
  }

  index_type* Get() {
    base::LockGuard _(lock_);
    auto back = pool_.back();
    pool_.pop_back();
    return back;
  }

  void Put(index_type* map) {
    base::LockGuard _(lock_);
    pool_.push_back(map);
  }

private:
  std::vector<index_type*> pool_;
  mutable base::SpinLock lock_;
};

template <typename T>
class PriorityHashTable {
  using index_type = std::unordered_map<int64_t, T*>;

private:
  index_type* index_;
  std::atomic_bool isCleaning_{false};

  const int queue_priority_;
  mutable base::NamedSpinLock lock_;

  base::epoch::EpochManager* epoch_manager_ =
      base::epoch::EpochManager::GetInstance();

  const int kUseParallelClean_ = false;

public:
  PriorityHashTable(
      int queue_priority, index_type* init_index, int kUseParallelClean)
      : queue_priority_(queue_priority), kUseParallelClean_(kUseParallelClean) {
    index_ = init_index;
  }

  static constexpr int kBumpInterval = 1;

  // maybe BUG
  void insert(T* newNode) {
    // base::NamedLockGuard _(lock_, "insert");
    base::NamedLockGuard lock_guard(lock_, "insert");
    epoch_manager_->Protect();
    auto readed_index                 = base::Atomic::load(&index_);
    (*readed_index)[newNode->GetID()] = newNode;
    epoch_manager_->UnProtect();
    epoch_manager_->BumpCurrentEpoch();
  }

  void remove(T* node) {
    // base::NamedLockGuard _(lock_, "remove");
    base::NamedLockGuard lock_guard(lock_, "remove");
    epoch_manager_->Protect();
    auto readed_index = base::Atomic::load(&index_);
    readed_index->erase(node->GetID());
    epoch_manager_->UnProtect();
    epoch_manager_->BumpCurrentEpoch();
  }

  index_type* Clean(index_type* new_index, std::function<void(T*)> drain_fn) {
    // base::NamedLockGuard _(lock_, "clean");
    index_type* old_index = nullptr;
    bool success          = false;
    {
      base::NamedLockGuard lock_guard(lock_, "clean");
      isCleaning_ = true;
      old_index   = base::Atomic::load(&index_);
      success     = base::Atomic::CAS((void**)&index_, old_index, new_index);
    }
    auto epoch = epoch_manager_->GetCurrentEpoch();

    while (!epoch_manager_->IsSafeToReclaim(epoch)) {
      epoch_manager_->BumpCurrentEpoch();
      //   RECSTORE_LOG_EVERY_MS(ERROR, 1000)
      //       << "stall in IsSafeToReclaim. "
      //       << "epoch=" << epoch_manager_->GetCurrentEpoch()
      //       << ". safe_epoch=" << epoch_manager_->GetSafeEpoch4Debug();
    }
    CHECK(success);

    if (!success)
      return nullptr;

    // LOG(INFO) << "cleaning priorityhashtable, priority=" << queue_priority_
    //           << ", size=" << old_index->size();

    if (kUseParallelClean_) {
      RECSTORE_LOG_EVERY_MS(WARNING, 1000)
          << "Parallel clean is disabled for standard unordered_map backend.";
    }

    for (auto [key, value] : *old_index) {
      drain_fn(value);
    }

    old_index->clear();
    {
      base::NamedLockGuard lock_guard(lock_, "clean_done");
      isCleaning_ = false;
    }
    return old_index;
  }

  bool empty() const {
    base::NamedLockGuard lock_guard(lock_, "empty");
    return (!isCleaning_) && index_->size() == 0;
  }

  std::string ToString() const {
    base::NamedLockGuard lock_guard(lock_, "ToString");
    std::stringstream ss;
    for (auto [key, value] : *index_) {
      ss << value->ToString() << " \n";
    }
    return ss.str();
  }
};

template <typename T>
class ParallelPqIndexV2 {
private:
  std::vector<int> index_;

public:
  ParallelPqIndexV2(int64_t capacity) { index_.resize(capacity, -1); }

  void assign(T* value, int priority) { index_[value->GetID()] = priority; }

  void erase(T* value) { index_[value->GetID()] = -1; }

  bool exists(T* value) const { return index_[value->GetID()] != -1; }

  bool try_insert(T* value, int priority) {
    return base::Atomic::CAS((int*)&index_[value->GetID()], -1, priority);
  }

  int operator[](T* key) const { return index_[key->GetID()]; }
};

template <typename T>
class ParallelPqV2 : public AbstractParallelPq<T> {
  using BaseClass                   = AbstractParallelPq<T>;
  constexpr static int kMaxPriority = BaseClass::kMaxPriority;
  static constexpr int kInf         = BaseClass::kInf;

public:
  ParallelPqV2(int64_t total_count, int kUseParallelClean)
      : index_(total_count), hash_map_pool_(kMaxPriority + 100, 100000) {
    for (int i = 0; i < kMaxPriority; i++) {
      if (i == kMaxPriority - 1)
        qs_[i] = new PriorityHashTable<T>(
            kInf, hash_map_pool_.Get(), kUseParallelClean);
      else
        qs_[i] = new PriorityHashTable<T>(
            i, hash_map_pool_.Get(), kUseParallelClean);
    }
  }

  void PushOrUpdate(T* value) override { Upsert(value); }

  T* top() const override {
    LOG(FATAL) << "dont call";
    return nullptr;
  }

  void UpdatePossibleMIN(int min) const override { LOG(FATAL) << "dont call"; }

  std::string ToString() const {
    std::stringstream ss;
    ss << "ParallelPqV2:\n";
    if (empty()) {
      ss << "\t\t" << "empty\n";
      return ss.str();
    }

    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        ss << "\t\t" << "Q" << i << " :" << qs_[i]->ToString() << "\n";
      }
    }
    return ss.str();
  }

  void ForDebug(const std::string& head) {}

  void CheckConsistency(const std::string& hint = "") {
    std::unordered_set<int64_t> id_set;
    for (int i = 0; i < kMaxPriority; i++) {
      auto id_set_per_q = qs_[i]->CheckConsistency();
    }
  }

  bool empty() const override {
#if defined(PPQ_ALL_SCAN)
    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        return false;
      }
    }
    return true;

#elif defined(PPQ_SELECTIVE_SCAN)
    for (int i = priority_possible_min_; i < priority_possible_max_; i++) {
      if (!qs_[i]->empty()) {
        return false;
      }
    }

    int i = kMaxPriority;
    if (!qs_[i]->empty()) {
      return false;
    }
    return true;
#else
#  error "defined a macro"
#endif
  }

  void ChunkClean(std::function<void(T*)> drain_fn) override {
    xmh::RAIITimer clean_timer("ChunkClean");
    std::lock_guard<std::recursive_mutex> pq_guard(lock_);
#if defined(PPQ_ALL_SCAN)
    for (int i = 0; i < kMaxPriority; i++) {
      if (qs_[i]->empty())
        continue;
      auto* ret = qs_[i]->Clean(hash_map_pool_.Get(), drain_fn);
      if (ret)
        hash_map_pool_.Put(ret);
      break;
    }

#elif defined(PPQ_SELECTIVE_SCAN)
    for (int i = priority_possible_min_; i < priority_possible_max_; i++) {
      auto* p = qs_[i]->top();
      if (qs_[i]->empty())
        continue;

      if (p) {
        // update MIN
        if (i != (kMaxPriority - 1))
          priority_possible_min_ = std::max(i - 1, 0);
        return p->Data();
      }
    }
    int i   = kMaxPriority;
    auto* p = qs_[i]->top();
    if (p) {
      return p->Data();
    }
    return nullptr;

    auto* ret = qs_[i]->Clean(hash_map_pool_.Get(), drain_fn);
    if (ret)
      hash_map_pool_.Put(ret);
#else
#  error "defined a macro"
#endif
  }

  int MinPriority() const override {
#if defined(PPQ_ALL_SCAN)
    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty())
        return BaseClass::CastQueueNoToPriority(i);
    }
    return kInf;

#elif defined(PPQ_SELECTIVE_SCAN)
    for (int i = priority_possible_min_; i < priority_possible_max_; i++) {
      if (!qs_[i]->empty()) {
        // update MIN
        if (i != (kMaxPriority - 1))
          priority_possible_min_ = std::max(i - 1, 0);
        return BaseClass::CastQueueNoToPriority(i);
      }
    }
    int i = kMaxPriority;
    if (!qs_[i]->empty())
      return BaseClass::CastQueueNoToPriority(i);
    return kInf;
#else
#  error "defined a macro"
#endif
  }

private:
  void Upsert(T* value) {
    std::lock_guard<std::recursive_mutex> pq_guard(lock_);
    int new_priority = value->Priority();

#if defined(PPQ_SELECTIVE_SCAN)
    // update MAX
    if (new_priority != kInf && (new_priority + 1) > priority_possible_max_) {
      // priority_possible_max_ = std::min(new_priority + 1, kMaxPriority);
      priority_possible_max_ = new_priority + 1;
    }
#endif
    if (index_.exists(value)) {
      int old_priority = index_[value];
      if (old_priority == new_priority) {
        return;
      }
      // ↓ atomically
      qs_[BaseClass::CastPriorityToQueueNo(new_priority)]->insert(value);
      qs_[BaseClass::CastPriorityToQueueNo(old_priority)]->remove(value);
      index_.assign(value, new_priority);
    } else {
      // atomically
      auto success = index_.try_insert(value, new_priority);
      if (success) {
        qs_[BaseClass::CastPriorityToQueueNo(new_priority)]->insert(value);
      } else {
        ;
      }
    }
  }

  std::array<PriorityHashTable<T>*, kMaxPriority> qs_;
  ParallelPqIndexV2<T> index_;
  mutable std::recursive_mutex lock_;

  HashMapPool<T> hash_map_pool_;

  mutable std::atomic_int priority_possible_min_{0};
  mutable std::atomic_int priority_possible_max_{kMaxPriority};

  thread_local static base::StdDelayedRecycle recycle_;
};

template <typename T>
thread_local base::StdDelayedRecycle ParallelPqV2<T>::recycle_;

} // namespace recstore
