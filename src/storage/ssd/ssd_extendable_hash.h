#pragma once

#include <string>
#include <vector>

#include "../ssd/CCEH.h"   // underlying SSD hash implementation
#include "../hybrid/index.h"         // generic index interface definition
#include "../../base/base.h"
#include "../../base/factory.h"
class ExtendibleHashSSD : public Index {
 public:
  explicit ExtendibleHashSSD(const IndexConfig &config);
  ~ExtendibleHashSSD() override;

  void Util() override;

  void Put(const uint64_t key, uint64_t pointer, unsigned tid = 0 ) override;
  void Get(const uint64_t key, uint64_t &pointer, unsigned tid = 0) override;
  void BatchGet(base::ConstArray<uint64_t> keys, uint64_t *pointers,
                unsigned tid) override;
  void BatchPut(coroutine<void>::push_type &sink, base::ConstArray<uint64_t> keys,
                uint64_t *pointers, unsigned tid) override;
  void BatchGet(coroutine<void>::push_type &sink,
              base::ConstArray<uint64_t> keys,
              uint64_t* pointers,
              unsigned tid) override;

  bool Delete(uint64_t &key) override;
  void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) override;
  void LoadFakeData(int64_t key_capacity, int value_size) override;
  void clear() override;
  void DebugInfo() const override;

 private:
  std::string filename_;
  CCEH *hash_table_;
};

FACTORY_REGISTER(Index, SSD, ExtendibleHashSSD, const IndexConfig &);
