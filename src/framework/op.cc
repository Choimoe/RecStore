#include "framework/op.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <numeric>
#include <thread>

// Assuming InitStrategyType is defined in base/tensor.h
#include "base/tensor.h"
#include "base/tensor.h" 
#include "op.h"

namespace recstore {
namespace framework {

class KVClientOp : public CommonOp {
public:
  KVClientOp() : learning_rate_(0.01f), embedding_dim_(-1) {
    std::cout << "KVClientOp initialized with full interface." << std::endl;
  }

  void EmbInit(const base::RecTensor& keys,
               const base::RecTensor& init_values) override {
    EmbWrite(keys, init_values);
  }

  void EmbInit(const base::RecTensor& keys,
               const InitStrategy& strategy) override {
    std::lock_guard<std::mutex> lock(mtx_);
    if (embedding_dim_ == -1) {
      throw std::runtime_error(
          "KVClientOp Error: Embedding dimension has not been set.");
    }

    const uint64_t* key_data = keys.data_as<uint64_t>();
    const int64_t num_keys   = keys.shape(0);
    const int64_t emb_dim    = this->embedding_dim_;

    for (int64_t i = 0; i < num_keys; ++i) {
      uint64_t key = key_data[i];
      store_[key]  = std::vector<float>(emb_dim, 0.0f);
    }
  }

  void EmbRead(const base::RecTensor& keys, base::RecTensor& values) override {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = values.shape(1);
    if (embedding_dim_ != -1 && embedding_dim_ != emb_dim) {
      throw std::runtime_error(
          "KVClientOp Error: Inconsistent embedding dimension for read.");
    }

    const uint64_t* key_data = keys.data_as<uint64_t>();
    float* value_data        = values.data_as<float>();
    const int64_t num_keys   = keys.shape(0);

    for (int64_t i = 0; i < num_keys; ++i) {
      uint64_t key = key_data[i];
      auto it      = store_.find(key);
      if (it == store_.end()) {
        std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
      } else {
        std::copy(
            it->second.begin(), it->second.end(), value_data + i * emb_dim);
      }
    }
  }

  void EmbWrite(const base::RecTensor& keys,
                const base::RecTensor& values) override {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = values.shape(1);
    if (embedding_dim_ == -1) {
      embedding_dim_ = emb_dim;
      std::cout << "KVClientOp: Inferred and set embedding dimension to "
                << embedding_dim_ << std::endl;
    } else if (embedding_dim_ != emb_dim) {
      throw std::runtime_error(
          "KVClientOp Error: Inconsistent embedding dimension for write.");
    }

    const uint64_t* key_data = keys.data_as<uint64_t>();
    const float* value_data  = values.data_as<float>();
    const int64_t num_keys   = keys.shape(0);

    for (int64_t i = 0; i < num_keys; ++i) {
      uint64_t key       = key_data[i];
      const float* start = value_data + i * emb_dim;
      const float* end   = start + emb_dim;
      store_[key].assign(start, end);
    }
  }

  void EmbUpdate(const base::RecTensor& keys,
                 const base::RecTensor& grads) override {
    std::lock_guard<std::mutex> lock(mtx_);
    const int64_t emb_dim = grads.shape(1);
    if (embedding_dim_ == -1) {
      embedding_dim_ = emb_dim;
    } else if (embedding_dim_ != emb_dim) {
      throw std::runtime_error(
          "KVClientOp Error: Inconsistent embedding dimension for update.");
    }

    const uint64_t* key_data = keys.data_as<uint64_t>();
    const float* grad_data   = grads.data_as<float>();
    const int64_t num_keys   = keys.shape(0);

    for (int64_t i = 0; i < num_keys; ++i) {
      uint64_t key = key_data[i];
      auto it      = store_.find(key);
      if (it != store_.end()) {
        for (int64_t j = 0; j < emb_dim; ++j) {
          it->second[j] -= learning_rate_ * grad_data[i * emb_dim + j];
        }
      }
    }
  }

  // Stubs for other optional APIs remain unchanged...
  bool EmbExists(const base::RecTensor& keys) override {
    throw std::runtime_error("Not impl");
  }
  void EmbDelete(const base::RecTensor& keys) override {
    throw std::runtime_error("Not impl");
  }
  uint64_t EmbPrefetch(const base::RecTensor& keys) override {
    throw std::runtime_error("Not impl");
  }
  bool IsPrefetchDone(uint64_t prefetch_id) override {
    throw std::runtime_error("Not impl");
  }
  void WaitForPrefetch(uint64_t prefetch_id) override {
    throw std::runtime_error("Not impl");
  }
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
};

void KVClientOp::EmbRead (const RecTensor& keys, RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but values has length " +
                                std::to_string(values.shape(0)));
  }
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  ConstArray<uint64_t> keys_array(keys_data, L);
  float* values_data = values.data_as<float>();
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  bool success = ps_client_.GetParameter( keys_array, values_data);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

void KVClientOp::EmbWrite (const RecTensor& keys, const RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but values has length " +
                                std::to_string(values.shape(0)));
  }
  const int64_t D = values.shape(1);

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  ConstArray<uint64_t> keys_array(keys_data, L);
  const float* values_data = values.data_as<float>();
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;
  
  const std::vector<std::vector<float>> values_vector(L, std::vector<float>(D));
  for (int64_t i = 0; i < L; ++i) {
    for (int64_t j = 0; j < D; ++j) {
        values_vector[i][j] = values_data[i * D + j];
    }
  }
  bool success = ps_client_.PutParameter( keys_array, values_vector);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

uint64_t KVClientOp::EmbPrefetch(const base::RecTensor& keys) {
    auto prefetch_id = ps_client_->PrefetchParameter(keys);

    return prefetch_id;
}

bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
    return ps_client_->IsPrefetchDone(prefetch_id);
}

void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
    while (!IsPrefetchDone(prefetch_id)) {
      // do nothing
    }
}

void KVClientOp::GetPretchResult(uint64_t prefetch_id, std::vector<std::vector<float>>* values) override {
    ps_client_->GetPrefetchResult(prefetch_id, values);
}

// **FIX**: Replaced the previous singleton implementation with the robust
// std::call_once pattern. This guarantees that the KVClientOp instance is
// created exactly once, regardless of build environment complexities.
std::shared_ptr<CommonOp> GetKVClientOp() {
  static std::shared_ptr<CommonOp> instance;
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    instance = std::make_shared<KVClientOp>();
  });
  return instance;
}


} // namespace recstore
