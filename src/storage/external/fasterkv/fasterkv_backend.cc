#include "storage/external/fasterkv/fasterkv_backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/faster.h"
#include "core/key_hash.h"
#include "core/status.h"
#include "device/null_disk.h"

namespace recstore::storage::fasterkv {
namespace {

using FASTER::core::CallbackContext;
using FASTER::core::FasterHashHelper;
using FASTER::core::FasterKv;
using FASTER::core::IAsyncContext;
using FASTER::core::KeyHash;
using FASTER::core::Status;
using FASTER::core::StatusStr;

class GenLock {
public:
  GenLock() : control_(0) {}
  explicit GenLock(uint64_t control) : control_(control) {}

  GenLock& operator=(const GenLock& other) {
    control_ = other.control_;
    return *this;
  }

  union {
    struct {
      uint64_t gen_number : 62;
      uint64_t locked     : 1;
      uint64_t replaced   : 1;
    };
    uint64_t control_;
  };
};

static_assert(sizeof(GenLock) == 8, "GenLock must be 8 bytes");

class AtomicGenLock {
public:
  AtomicGenLock() : control_(0) {}
  explicit AtomicGenLock(uint64_t control) : control_(control) {}

  GenLock load() const { return GenLock{control_.load()}; }

  void store(GenLock desired) { control_.store(desired.control_); }

  bool try_lock(bool& replaced) {
    replaced = false;
    GenLock expected{control_.load()};
    expected.locked   = 0;
    expected.replaced = 0;
    GenLock desired{expected.control_};
    desired.locked = 1;

    if (control_.compare_exchange_strong(expected.control_, desired.control_)) {
      return true;
    }
    if (expected.replaced) {
      replaced = true;
    }
    return false;
  }

  void unlock(bool replaced) {
    if (!replaced) {
      const uint64_t sub_delta = (uint64_t{1} << 62) - 1;
      control_.fetch_sub(sub_delta);
    } else {
      const uint64_t add_delta = (uint64_t{1} << 63) - (uint64_t{1} << 62) + 1;
      control_.fetch_add(add_delta);
    }
  }

private:
  std::atomic<uint64_t> control_;
};

static_assert(sizeof(AtomicGenLock) == 8, "AtomicGenLock must be 8 bytes");

class UInt64Key {
public:
  explicit UInt64Key(uint64_t key) : key(key) {}

  UInt64Key(const UInt64Key&) = default;

  inline static constexpr uint32_t size() {
    return static_cast<uint32_t>(sizeof(UInt64Key));
  }

  inline KeyHash GetHash() const {
    return KeyHash{FasterHashHelper<uint64_t>::compute(key)};
  }

  inline bool operator==(const UInt64Key& other) const {
    return key == other.key;
  }

  inline bool operator!=(const UInt64Key& other) const {
    return key != other.key;
  }

  uint64_t key;
};

class VariableValue {
public:
  VariableValue() : gen_lock_{0}, size_{0}, length_{0} {}

  inline uint32_t size() const { return size_; }

private:
  friend class UpsertContext;
  friend class ReadContext;

  AtomicGenLock gen_lock_;
  uint32_t size_;
  uint32_t length_;

  inline const uint8_t* buffer() const {
    return reinterpret_cast<const uint8_t*>(this + 1);
  }

  inline uint8_t* buffer() { return reinterpret_cast<uint8_t*>(this + 1); }
};

class UpsertContext : public IAsyncContext {
public:
  using key_t   = UInt64Key;
  using value_t = VariableValue;

  UpsertContext(uint64_t key, const char* value, uint32_t value_size)
      : key_(key),
        value_(reinterpret_cast<const uint8_t*>(value)),
        value_size_(value_size) {}

  UpsertContext(const UpsertContext& other)
      : key_(other.key_),
        owned_value_(other.value_, other.value_ + other.value_size_),
        value_(owned_value_.data()),
        value_size_(other.value_size_) {}

  inline const key_t& key() const { return key_; }

  inline uint32_t value_size() const {
    return static_cast<uint32_t>(sizeof(value_t) + value_size_);
  }

  inline void Put(value_t& value) {
    value.gen_lock_.store(GenLock{0});
    value.size_   = value_size();
    value.length_ = value_size_;
    std::memcpy(value.buffer(), value_, value_size_);
  }

  inline bool PutAtomic(value_t& value) {
    bool replaced = false;
    while (!value.gen_lock_.try_lock(replaced) && !replaced) {
      std::this_thread::yield();
    }
    if (replaced) {
      return false;
    }
    if (value.size_ < value_size()) {
      value.gen_lock_.unlock(true);
      return false;
    }
    value.length_ = value_size_;
    std::memcpy(value.buffer(), value_, value_size_);
    value.gen_lock_.unlock(false);
    return true;
  }

protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) override {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

private:
  key_t key_;
  std::vector<uint8_t> owned_value_;
  const uint8_t* value_;
  uint32_t value_size_;
};

class ReadContext : public IAsyncContext {
public:
  using key_t   = UInt64Key;
  using value_t = VariableValue;

  ReadContext(uint64_t key,
              char* output,
              uint32_t expected_size,
              size_t index,
              const FasterKVBackend::MissCallback* on_miss)
      : key_(key),
        output_(reinterpret_cast<uint8_t*>(output)),
        expected_size_(expected_size),
        index_(index),
        on_miss_(on_miss),
        found_(false) {}

  ReadContext(const ReadContext& other)
      : key_(other.key_),
        output_(other.output_),
        expected_size_(other.expected_size_),
        index_(other.index_),
        on_miss_(other.on_miss_),
        found_(other.found_) {}

  inline const key_t& key() const { return key_; }

  inline void Get(const value_t& value) { CopyValue(value); }

  inline void GetAtomic(const value_t& value) {
    GenLock before;
    GenLock after;
    do {
      before = value.gen_lock_.load();
      CopyValue(value);
      after = value.gen_lock_.load();
    } while (before.gen_number != after.gen_number);
  }

  bool found() const { return found_; }

  void NotifyMiss() const {
    if (on_miss_ != nullptr && *on_miss_) {
      (*on_miss_)(index_);
    }
  }

protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) override {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

private:
  void CopyValue(const value_t& value) {
    if (value.length_ != expected_size_) {
      throw std::runtime_error("FasterKV value size mismatch");
    }
    std::memcpy(output_, value.buffer(), expected_size_);
    found_ = true;
  }

  key_t key_;
  uint8_t* output_;
  uint32_t expected_size_;
  size_t index_;
  const FasterKVBackend::MissCallback* on_miss_;
  bool found_;
};

using FasterStore =
    FasterKv<UInt64Key, VariableValue, FASTER::device::NullDisk>;

uint64_t NextPowerOfTwo(uint64_t value) {
  uint64_t out = 1;
  while (out < value && out <= (uint64_t{1} << 63)) {
    out <<= 1;
  }
  return out;
}

uint64_t ComputeHashTableSize(uint64_t capacity) {
  const uint64_t min_size = 1024;
  const uint64_t wanted   = std::max(min_size, capacity * 2);
  return NextPowerOfTwo(wanted);
}

uint64_t ComputeLogSize(uint64_t capacity, size_t value_size) {
  const uint64_t min_size        = uint64_t{256} << 20;
  const uint64_t record_overhead = 128;
  const uint64_t per_record =
      record_overhead + static_cast<uint64_t>(value_size);
  return std::max(min_size, NextPowerOfTwo(capacity * per_record * 2));
}

void CheckStatus(Status status, const char* operation) {
  if (status != Status::Ok) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + StatusStr(status));
  }
}

void CompletePendingOrThrow(FasterStore& store, const char* operation) {
  if (!store.CompletePending(true)) {
    throw std::runtime_error(std::string(operation) +
                             " failed: pending operations did not complete");
  }
}

} // namespace

class FasterKVBackend::Impl {
public:
  Impl(uint64_t capacity, size_t value_size)
      : value_size_(value_size),
        store_(FasterStore::IndexConfig{ComputeHashTableSize(capacity)},
               ComputeLogSize(capacity, value_size),
               "",
               1.0) {
    if (capacity == 0) {
      throw std::invalid_argument("FasterKVBackend capacity must be > 0");
    }
    if (value_size_ == 0 ||
        value_size_ >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::invalid_argument("FasterKVBackend value_size is invalid");
    }
  }

  ~Impl() { StopThreadSessionIfNeeded(); }

  void Insert(size_t num_keys, const long long* keys, const char* values) {
    EnsureThreadSession();
    auto callback = [](IAsyncContext*, Status status) {
      CheckStatus(status, "FasterKV::Upsert async");
    };

    for (size_t i = 0; i < num_keys; ++i) {
      UpsertContext context{static_cast<uint64_t>(keys[i]),
                            values + i * value_size_,
                            static_cast<uint32_t>(value_size_)};
      const Status status =
          store_.Upsert(context, callback, NextSerialNumber());
      if (status == Status::Pending) {
        continue;
      }
      CheckStatus(status, "FasterKV::Upsert");
    }
    CompletePendingOrThrow(store_, "FasterKV::Upsert");
  }

  void Fetch(size_t num_keys,
             const long long* keys,
             char* values,
             const MissCallback& on_miss) {
    EnsureThreadSession();
    for (size_t i = 0; i < num_keys; ++i) {
      auto callback = [](IAsyncContext* context, Status status) {
        CallbackContext<ReadContext> read_context{context};
        if (status == Status::NotFound) {
          read_context->NotifyMiss();
          return;
        }
        CheckStatus(status, "FasterKV::Read async");
        if (!read_context->found()) {
          throw std::runtime_error("FasterKV::Read async returned no value");
        }
      };

      ReadContext context{
          static_cast<uint64_t>(keys[i]),
          values + i * value_size_,
          static_cast<uint32_t>(value_size_),
          i,
          &on_miss};
      const Status status = store_.Read(context, callback, NextSerialNumber());
      if (status == Status::NotFound) {
        context.NotifyMiss();
        continue;
      }
      if (status == Status::Pending) {
        CompletePendingOrThrow(store_, "FasterKV::Read");
        continue;
      }
      CheckStatus(status, "FasterKV::Read");
      if (!context.found()) {
        throw std::runtime_error("FasterKV::Read returned Ok without value");
      }
    }
  }

private:
  void EnsureThreadSession() {
    if (session_.owner == this) {
      return;
    }
    if (session_.owner != nullptr) {
      throw std::runtime_error("Thread already owns a FasterKVBackend session");
    }
    store_.StartSession();
    session_.owner = this;
  }

  void StopThreadSessionIfNeeded() {
    if (session_.owner == this) {
      store_.StopSession();
      session_.owner = nullptr;
    }
  }

  void StopSessionFromThreadExit() { store_.StopSession(); }

  uint64_t NextSerialNumber() { return serial_number_.fetch_add(1) + 1; }

  struct ThreadSession {
    Impl* owner = nullptr;

    ~ThreadSession() {
      if (owner != nullptr) {
        owner->StopSessionFromThreadExit();
      }
    }
  };

  size_t value_size_;
  FasterStore store_;
  std::atomic<uint64_t> serial_number_{0};

  static thread_local ThreadSession session_;
};

thread_local FasterKVBackend::Impl::ThreadSession
    FasterKVBackend::Impl::session_;

FasterKVBackend::FasterKVBackend(uint64_t capacity, size_t value_size)
    : impl_(std::make_unique<Impl>(capacity, value_size)) {}

FasterKVBackend::~FasterKVBackend() = default;

void FasterKVBackend::Insert(
    size_t num_keys, const long long* keys, const char* values) {
  impl_->Insert(num_keys, keys, values);
}

void FasterKVBackend::Fetch(size_t num_keys,
                            const long long* keys,
                            char* values,
                            const MissCallback& on_miss) {
  impl_->Fetch(num_keys, keys, values, on_miss);
}

} // namespace recstore::storage::fasterkv
