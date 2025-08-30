#pragma once

#include "hash_interface.h"
#include "pair.h"
#include <cstring>
#include <vector>
#include "../hybrid/index.h"
// #include "/home/nammh/quartz/src/lib/pmalloc.h"

#define LSB
const size_t kMask = 256 - 1;
const size_t kShift = 8;

struct Block {
  static const size_t kBlockSize = 256; // 4 - 1
  // static const size_t kBlockSize = 1024; // 16 - 1
  // static const size_t kBlockSize = 4*1024; // 64 - 1
  // static const size_t kBlockSize = 16*1024; // 256 - 1
  // static const size_t kBlockSize = 64*1024; // 1024 - 1
  // static const size_t kBlockSize = 256 * 1024; // 4096 - 1
  static const size_t kNumSlot = kBlockSize / sizeof(Pair);

  Block(void) : local_depth{0} {}

  Block(size_t depth) : local_depth{depth} {}

  ~Block(void) {}

  void *operator new(size_t size) {
    void *ret;
    posix_memalign(&ret, 64, size);
    // ret = pmalloc(size);
    return ret;
  }

  void *operator new[](size_t size) {
    void *ret;
    posix_memalign(&ret, 64, size);
    // ret = pmalloc(size);
    return ret;
  }

  int Insert(Key_t &, Value_t, size_t);
  void Insert4split(Key_t &, Value_t);
  bool Put(Key_t &, Value_t, size_t);
  Block **Split(void);
  bool Delete(Key_t &);

  void operator delete[](void* p) noexcept { std::free(p); }
  void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
  void operator delete(void* p) noexcept { std::free(p); }
  void operator delete(void* p, std::size_t) noexcept { std::free(p); }

  Pair _[kNumSlot];
  size_t local_depth;
  int64_t sema = 0;
  size_t pattern = 0;
  size_t numElem(void);
};

struct Directory {
  static const size_t kDefaultDirectorySize = 1024;
  Block **_;
  size_t capacity;
  bool lock;
  int sema = 0;

  Directory(void) {
    capacity = kDefaultDirectorySize;
    _ = new Block *[capacity];
    lock = false;
    sema = 0;
  }

  Directory(size_t size) {
    capacity = size;
    _ = new Block *[capacity];
    lock = false;
    sema = 0;
  }

  ~Directory(void) { delete[] _; }

  bool Acquire(void) {
    bool unlocked = false;
    return CAS(&lock, &unlocked, true);
  }

  bool Release(void) {
    bool locked = true;
    return CAS(&lock, &locked, false);
  }

  void SanityCheck(void *);
  void LSBUpdate(int, int, int, int, Block **);
};

class ExtendibleHash : public Index, public Hash {
public:
  ExtendibleHash(const IndexConfig &config);
  ExtendibleHash(void);
  ExtendibleHash(size_t);
  ~ExtendibleHash(void);
  void Insert(Key_t &, Value_t);
  bool InsertOnly(Key_t &, Value_t);
  bool Delete(Key_t &);
  Value_t Get(Key_t &);
  Value_t FindAnyway(Key_t &);
  void Insert(const Key_t &key, Value_t value);
  bool InsertOnly(const Key_t &key, Value_t value);
  Value_t Get(const Key_t &key);
  double Utilization(void);
  size_t Capacity(void);

  void *operator new(size_t size) {
    void *ret;
    posix_memalign(&ret, 64, size);
    // ret = pmalloc(size);
    return ret;
  }
// From Index
  void Util() override;

  void Get(const uint64_t key, uint64_t &value, unsigned tid) override;
  void Put(const uint64_t key, uint64_t value, unsigned tid) override;

  void BatchPut(coroutine<void>::push_type &sink,
                base::ConstArray<uint64_t> keys,
                uint64_t* pointers,
                unsigned tid) override;

  void BatchGet(base::ConstArray<uint64_t> keys,
                uint64_t* pointers,
                unsigned tid) override;

  void BatchGet(coroutine<void>::push_type &sink,
                base::ConstArray<uint64_t> keys,
                uint64_t* pointers,
                unsigned tid) override;

  void DebugInfo() const override;

  void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) override;

  void LoadFakeData(int64_t key_capacity, int value_size) override;

  void clear() override;

  void operator delete(void* p) noexcept { std::free(p); }
  void operator delete(void* p, std::size_t) noexcept { std::free(p); }
  // std::string RetrieveValue(uint64_t raw_value) override;
private:
  Directory dir;
  size_t global_depth;
};