#pragma once
#include <array>
#include <iostream>
#include <unordered_map>

#include "base/lock.h"
#include "src/memory/malloc.h"

// #define PPQ_ALL_SCAN
#define PPQ_SELECTIVE_SCAN

namespace recstore {

template <typename T>
class DoublyLinkedList;

template <typename T>
struct Node {
private:
  T data;
  int queue_priority;
  Node* prev;
  Node* next;

  friend class recstore::DoublyLinkedList<T>;

public:
  Node(const T& value, int queue_priority)
      : data(value),
        queue_priority(queue_priority),
        prev(nullptr),
        next(nullptr) {}

  T Data() const { return data; }

  void ResetPointer() {
    prev = nullptr;
    next = nullptr;
  }

  int QueuePriority() const { return queue_priority; }

  void SetQueuePriority(int queue_priority) {
    this->queue_priority = queue_priority;
  }
};

template <typename T>
class DoublyLinkedList {
private:
  std::atomic<Node<T>*> head_;
  Node<T>* tail_;
  std::atomic_long size_;
  const int queue_priority_;
  mutable base::SpinLock lock_;

public:
  DoublyLinkedList(int queue_priority)
      : head_(nullptr),
        tail_(nullptr),
        size_(0),
        queue_priority_(queue_priority) {}

  void insert(Node<T>* newNode) {
    base::LockGuard _(lock_);
    if (!head_) {
      head_ = tail_ = newNode;
    } else {
      tail_->next   = newNode;
      newNode->prev = tail_;
      tail_         = newNode;
    }
    size_++;
  }

  void remove(Node<T>* nodeToRemove) {
    base::LockGuard _(lock_);
    CHECK_EQ(nodeToRemove->queue_priority, queue_priority_);
    if (nodeToRemove->prev) {
      nodeToRemove->prev->next = nodeToRemove->next;
    } else {
      // if head_ node
      head_ = nodeToRemove->next;
    }
    if (nodeToRemove->next) {
      nodeToRemove->next->prev = nodeToRemove->prev;
    } else {
      // if tail_ node
      tail_ = nodeToRemove->prev;
    }
    size_--;
  }

  Node<T>* pop() {
    base::LockGuard _(lock_);

    Node<T>* nodeToRemove = head_;
    head_                 = head_->next;
    head_->prev           = nullptr;
    size_--;
    return nodeToRemove;
  }

  Node<T>* top() { return head_; }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  std::unordered_set<int64_t> CheckConsistency() {
    base::LockGuard _(lock_);
    std::unordered_set<int64_t> id_set;

    Node<T>* current = head_;
    while (current) {
      if (current->prev)
        CHECK_EQ(current->prev->next, current);
      CHECK_EQ(current->data->Priority(), queue_priority_);
      id_set.insert(current->data->GetID());
      current = current->next;
    }
    return id_set;
  }

  std::string ToString() const {
    base::LockGuard _(lock_);

    std::stringstream ss;
    Node<T>* current = head_;
    int temp         = 0;
    while (current) {
      ss << current->data->ToString() << " \n";
      current = current->next;
      temp++;
      if (temp > size_)
        LOG(FATAL) << "linklist may not be linked properly";
    }
    return ss.str();
  }
};

template <typename T>
class ParallelPqIndex {
private:
  std::vector<Node<T>*> index_;
  mutable base::SpinLock lock_;

public:
  ParallelPqIndex(int64_t capacity) { index_.resize(capacity, nullptr); }

  void assign(const T& key, Node<T>* value) {
    base::LockGuard _(lock_);
    index_[key->GetID()] = value;
  }

  void erase(const T& key) {
    base::LockGuard _(lock_);
    index_[key->GetID()] = nullptr;
  }

  bool find(const T& key) const {
    base::LockGuard _(lock_);
    return index_[key->GetID()] != nullptr;
  }

  Node<T>* operator[](const T& key) const {
    base::LockGuard _(lock_);
    return index_[key->GetID()];
  }

  bool try_insert(const T& key, Node<T>* value) {
    base::LockGuard _(lock_);
    auto& slot = index_[key->GetID()];
    if (slot != nullptr) {
      return false;
    }
    slot = value;
    return true;
  }
};

template <typename T>
class AbstractParallelPq {
protected:
  // priority ranges from 0~<kMaxPriority-1>
  constexpr static int kMaxPriority = 1000;
  static constexpr int kInf         = std::numeric_limits<int>::max();

  static inline int CastPriorityToQueueNo(int queue_priority) {
    if (queue_priority == kInf)
      return kMaxPriority - 1;
    CHECK_LT(queue_priority, kMaxPriority - 1)
        << "Please increase kMaxPriority";
    CHECK_GE(queue_priority, 0);
    return queue_priority;
  }

  static inline int CastQueueNoToPriority(int queue_no) {
    if (queue_no == kMaxPriority - 1)
      return kInf;
    CHECK_LT(queue_no, kMaxPriority - 1) << "Please increase kMaxPriority";
    CHECK_GE(queue_no, 0);
    return queue_no;
  }

public:
  virtual void PushOrUpdate(T* value) = 0;

  virtual bool empty() const = 0;

  virtual int MinPriority() const = 0;

  virtual T* top() const = 0;

  virtual void ChunkClean(std::function<void(T*)> drain_fn) = 0;

  virtual void UpdatePossibleMIN(int min) const = 0;

  virtual ~AbstractParallelPq() = default;

  virtual std::string ToString() const = 0;
};

template <typename T>
class ParallelPq : public AbstractParallelPq<T> {
  using BaseClass                   = AbstractParallelPq<T>;
  constexpr static int kMaxPriority = BaseClass::kMaxPriority;
  static constexpr int kInf         = BaseClass::kInf;

public:
  ParallelPq(int64_t total_count) : index_(total_count) {
    for (int i = 0; i < kMaxPriority; i++) {
      if (i == kMaxPriority - 1)
        qs_[i] = new DoublyLinkedList<T*>(kInf);
      else
        qs_[i] = new DoublyLinkedList<T*>(i);
    }
  }

  void PushOrUpdate(T* value) override {
    Upsert(value);

    // // LOG(INFO) << "PushOrUpdate " << value->GetID();
    // if (!index_.find(value)) {
    //   push_inner(value);
    // } else {
    //   adjustPriority(value);
    // }
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << "CustomParallelPriorityQueue:\n";
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

  T* top() const override {
#if defined(PPQ_ALL_SCAN)
    for (int i = 0; i < kMaxPriority; i++) {
      auto* p = qs_[i]->top();
      if (p)
        return p->Data();
    }
    return nullptr;

#elif defined(PPQ_SELECTIVE_SCAN)
    for (int i = priority_possible_min_; i < priority_possible_max_; i++) {
      auto* p = qs_[i]->top();
      if (p) {
        // update MIN
        // if (i != (kMaxPriority - 1))
        //   priority_possible_min_ = std::max(i - 1, 0);
        return p->Data();
      }
    }
    int i   = kMaxPriority;
    auto* p = qs_[i]->top();
    if (p) {
      return p->Data();
    }
    return nullptr;
#else
#  error "defined a macro"
#endif
  }

  int MinPriority() const override {
#if defined(PPQ_ALL_SCAN)
    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty())
        return CastQueueNoToPriority(i);
    }
    return kInf;

#elif defined(PPQ_SELECTIVE_SCAN)
    for (int i = priority_possible_min_; i < priority_possible_max_; i++) {
      if (!qs_[i]->empty()) {
        // update MIN
        // if (i != (kMaxPriority - 1))
        //   priority_possible_min_ = std::max(i - 1, 0);
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

  void UpdatePossibleMIN(int min) const override {
    priority_possible_min_ = min;
  }

  void ChunkClean(std::function<void(T*)> drain_fn) override {
    LOG(FATAL) << "should not call this";
  }

  void pop_x(const T& value) {
    base::LockGuard pq_guard(lock_);
    LOG(FATAL) << "not USED now";
    // base::LockGuard guard(lock_);
    CHECK(index_.find(value));
    Node<T>* node = index_[value];
    // LOG(ERROR) << "Node<T> * node" << node;
    // LOG(ERROR) << "CastPriorityToQueueNo(node->queue_priority)="
    //            << CastPriorityToQueueNo(node->queue_priority);

    // LOG(ERROR) << "qs_[CastPriorityToQueueNo(node->queue_priority)] = "
    //            << qs_[CastPriorityToQueueNo(node->queue_priority)];
    qs_[CastPriorityToQueueNo(node->QueuePriority())]->remove(node);
    index_.erase(value);
    // delete node;
    recycle_.Recycle(node);
  }

private:
  void Upsert(T* value) {
    base::LockGuard pq_guard(lock_);
    int new_priority = value->Priority();

#if defined(PPQ_SELECTIVE_SCAN)
    // update MAX
    if (new_priority != kInf && (new_priority + 1) > priority_possible_max_) {
      // priority_possible_max_ = std::min(new_priority + 1, kMaxPriority);
      priority_possible_max_ = new_priority + 1;
    }
#endif
    if (index_.find(value)) {
      Node<T*>* node = index_[value];
      CHECK(node);
      int old_priority = node->QueuePriority();
      // ↓ atomically
      Node<T*>* newnode = new Node<T*>(value, new_priority);
      qs_[BaseClass::CastPriorityToQueueNo(new_priority)]->insert(newnode);
      qs_[BaseClass::CastPriorityToQueueNo(old_priority)]->remove(node);
      index_.assign(value, newnode);

      // delete node;
      recycle_.Recycle(node);

    } else {
      Node<T*>* newNode = new Node<T*>(value, new_priority);
      // atomically
      auto success = index_.try_insert(value, newNode);
      if (success) {
        qs_[BaseClass::CastPriorityToQueueNo(new_priority)]->insert(newNode);
      } else {
        delete newNode;
      }
    }
  }

  // void adjustPriority(const T &value) {
  //   // static base::SpinLock adjust_lock;
  //   // base::LockGuard adjust_guard(adjust_lock);
  //   LOG(FATAL) << "not USED now";

  //   Node<T> *node;
  //   do {
  //     node = index_[value];
  //     RECSTORE_LOG_EVERY_MS(ERROR, 1000) << "node is nullptr";
  //   } while (!node);

  //   CHECK(node);
  //   int new_priority = value->Priority();
  //   int old_priority = node->QueuePriority();

  //   // ↓ atomically
  //   Node<T> *newnode = new Node<T>(value, new_priority);
  //   qs_[CastPriorityToQueueNo(new_priority)]->insert(newnode);
  //   qs_[CastPriorityToQueueNo(old_priority)]->remove(node);

  //   index_.assign(value, newnode);
  //   recycle_.Recycle(node);
  // }

  // void push_inner(const T &value) {
  //   // static base::SpinLock push_lock;
  //   // base::LockGuard guard(push_lock);
  //   LOG(FATAL) << "not USED now";

  //   int priority = value->Priority();
  //   Node<T> *newNode = new Node<T>(value, priority);
  //   // atomically
  //   auto success = index_.try_insert(value, newNode);
  //   if (success) {
  //     // LOG(ERROR) << base::SFormat("index_[{}] = {})", value->GetID(),
  //     //                              newNode);
  //     qs_[CastPriorityToQueueNo(priority)]->insert(newNode);
  //   } else {
  //     delete newNode;
  //   }
  // }

  std::array<DoublyLinkedList<T*>*, kMaxPriority> qs_;
  ParallelPqIndex<T*> index_;
  mutable base::SpinLock lock_;

  mutable std::atomic_int priority_possible_min_{0};
  mutable std::atomic_int priority_possible_max_{kMaxPriority};

  thread_local static base::StdDelayedRecycle recycle_;
};

template <typename T>
thread_local base::StdDelayedRecycle ParallelPq<T>::recycle_;

} // namespace recstore

#undef PPQ_ALL_SCAN
#undef PPQ_SELECTIVE_SCAN
