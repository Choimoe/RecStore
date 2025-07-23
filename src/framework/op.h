#pragma once

#include "base/tensor.h"
#include "base_ps/base_client.h"

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

  CommonOp() {
    if (!ps_client_) {
      ps_client_ = new BasePSClient(js);
    }
  }

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
  void GetPretchResult(uint64_t prefetch_id,
                       std::vector<std::vector<float>>* values) = 0;

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

class KVClientOp : public CommonOp {
public:
  KVClientOp() : learning_rate_(0.01f), embedding_dim_(-1) {
    if (!ps_client_) {
      ps_client_ = new BasePSClient(js);
    }
    // std::cout << "KVClientOp initialized with full interface." << std::endl;
  }

  void EmbInit(const base::RecTensor& keys,
               const base::RecTensor& init_values) override {
    EmbWrite(keys, init_values);
  }

  // void EmbInit(const base::RecTensor& keys, const InitStrategy& strategy)
  // override;

  void EmbRead(const base::RecTensor& keys, base::RecTensor& values) override;

  void EmbWrite(const base::RecTensor& keys,
                const base::RecTensor& values) override;

  void EmbUpdate(const base::RecTensor& keys,
                 const base::RecTensor& grads) override;
  // {
  //     std::lock_guard<std::mutex> lock(mtx_);
  //     const int64_t emb_dim = grads.shape(1);
  //     if (embedding_dim_ == -1) {
  //         embedding_dim_ = emb_dim;
  //     } else if (embedding_dim_ != emb_dim) {
  //         throw std::runtime_error("KVClientOp Error: Inconsistent embedding
  //         dimension for update.");
  //     }

  //     const uint64_t* key_data = keys.data_as<uint64_t>();
  //     const float* grad_data = grads.data_as<float>();
  //     const int64_t num_keys = keys.shape(0);

  //     for (int64_t i = 0; i < num_keys; ++i) {
  //         uint64_t key = key_data[i];
  //         auto it = store_.find(key);
  //         if (it != store_.end()) {
  //             for (int64_t j = 0; j < emb_dim; ++j) {
  //                 it->second[j] -= learning_rate_ * grad_data[i * emb_dim +
  //                 j];
  //             }
  //         }
  //     }
  // }

  // Stubs for other optional APIs remain unchanged...
  bool EmbExists(const base::RecTensor& keys) override {
    throw std::runtime_error("Not impl");
  }
  void EmbDelete(const base::RecTensor& keys) override {
    throw std::runtime_error("Not impl");
  }

  uint64_t EmbPrefetch(const base::RecTensor& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  void GetPretchResult(uint64_t prefetch_id,
                       std::vector<std::vector<float>>* values) override;

  // 还需GetResult，clear某个prefetch_id
  uint64_t EmbWriteAsync(const base::RecTensor& keys,
                         const base::RecTensor& values) override {
    throw std::runtime_error("Not impl");
  }
  bool IsWriteDone(uint64_t write_id) override {
    throw std::runtime_error("Not impl");
  }
  void WaitForWrite(uint64_t write_id) override {
    throw std::runtime_error("Not impl");
  }

  void SaveToFile(const std::string& path) override {
    throw std::runtime_error("Not impl");
  }
  void LoadFromFile(const std::string& path) override {
    throw std::runtime_error("Not impl");
  }

private:
  std::unordered_map<uint64_t, std::vector<float>> store_;
  std::mutex mtx_;
  float learning_rate_;
  int64_t embedding_dim_;
  static BasePSClient* ps_client_ = nullptr;
};

std::shared_ptr<CommonOp> GetKVClientOp();

} // namespace recstore