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
#include "op.h"

namespace recstore {

void KVClientOp::EmbRead(const RecTensor& keys, RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  ConstArray<uint64_t> keys_array(keys_data, L);
  float* values_data = values.data_as<float>();
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  bool success = ps_client_.GetParameter(keys_array, values_data);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

void KVClientOp::EmbWrite(const RecTensor& keys,
                          const RecTensor& values) override {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
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
  bool success = ps_client_.PutParameter(keys_array, values_vector);
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

void KVClientOp::GetPretchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) override {
  ps_client_->GetPrefetchResult(prefetch_id, values);
}

std::shared_ptr<CommonOp> GetKVClientOp() {
  static std::shared_ptr<CommonOp> instance;
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    instance = std::make_shared<KVClientOp>();
  });
  return instance;
}

} // namespace recstore