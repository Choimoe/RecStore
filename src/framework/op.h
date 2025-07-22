#pragma once

#include "base/tensor.h"

using base::RecTensor;

namespace recstore {
enum class InitStrategyType { Normal, Uniform, Xavier, Zero };

struct InitStrategy {
  InitStrategy() = delete;
  InitStrategyType type;

  // Optional fields depending on type
  float mean  = 0.0f;
  float std   = 1.0f;
  float lower = -1.0f;
  float upper = 1.0f;

  InitStrategy(InitStrategyType t) : type(t) {}

  static InitStrategy Normal(float mean, float std) {
    InitStrategy s(InitStrategyType::Normal);
    s.mean = mean;
    s.std  = std;
    return s;
  }

  static InitStrategy Uniform(float lower, float upper) {
    InitStrategy s(InitStrategyType::Uniform);
    s.lower = lower;
    s.upper = upper;
    return s;
  }

  static InitStrategy Xavier() {
    return InitStrategy(InitStrategyType::Xavier);
  }
  static InitStrategy Zero() { return InitStrategy(InitStrategyType::Zero); }
};
class CommonOp {
public:
  // keys: uint64_t tensor with shape [N]
  // values: emb.dtype tensor with shape [N, D]

  virtual void EmbInit(const RecTensor& keys, const RecTensor& init_values) = 0;
  virtual void EmbInit(const RecTensor& keys, const InitStrategy& strategy) = 0;

  // Core KV APIs (sync)
  virtual void
  EmbRead(const RecTensor& keys, RecTensor& values) = 0; // sync read
  virtual void
  EmbWrite(const RecTensor& keys, const RecTensor& values) = 0; // sync write

  virtual bool
  EmbExists(const RecTensor& keys) = 0; // not urgent, optional existence check
  virtual void
  EmbDelete(const RecTensor& keys) = 0; // not urgent, optional deletion

  // Optional Gradient Hook (can be omitted if optimizer is outside)
  virtual void
  EmbUpdate(const RecTensor& keys, const RecTensor& grads) = 0; // not urgent

  // Prefetch & write (async)
  virtual uint64_t
  EmbPrefetch(const RecTensor& keys,
              const RecTensor& values) = 0; // async prefetch, returns a unique
                                            // ID to track the prefetch status.
  virtual bool IsPrefetchDone(
      uint64_t prefetch_id) = 0; // returns true if the prefetch identified by
                                 // prefetch_id is complete.
  virtual void WaitForPrefetch(
      uint64_t prefetch_id) = 0; // blocks until the prefetch identified by
                                 // prefetch_id is complete.

  virtual uint64_t
  EmbWriteAsync(const RecTensor& keys,
                const RecTensor& values) = 0; // async write, returns a unique
                                              // ID to track the write status.
  virtual bool
  IsWriteDone(uint64_t write_id) = 0; // returns true if the asynchronous write
                                      // identified by write_id is complete.
  virtual void
  WaitForWrite(uint64_t write_id) = 0; // blocks until the asynchronous write
                                       // identified by write_id is complete.

  // Persistence
  virtual void SaveToFile(const std::string& path)   = 0; // not urgent
  virtual void LoadFromFile(const std::string& path) = 0; // not urgent

  virtual ~CommonOp() = default;
};

namespace framework {
std::shared_ptr<CommonOp> GetKVClientOp();
} // namespace framework

} // namespace recstore
