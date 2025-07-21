#pragma once

#include "framework/op.h"
#include <torch/extension.h>
#include <optional>

namespace recstore {
namespace framework {

inline torch::Tensor ToTorchTensor(const base::RecTensor& tensor, torch::Device device);

/**
 * @brief A CommonOp implementation that stores the embedding table directly on
 * a single GPU's memory.
 *
 * This class manages a large torch::Tensor on a specified GPU and uses
 * PyTorch's ATen library to perform read, write, and update operations
 * directly on the GPU, minimizing CPU-GPU data transfers.
 */
class GPUKVStore : public CommonOp {
public:
  /**
   * @brief Constructs a GPUKVStore.
   * @param device_id The index of the GPU device to use (e.g., 0 for cuda:0).
   */
  explicit GPUKVStore(int device_id);
  ~GPUKVStore() override = default;

  // ===================================================================
  // CommonOp core interface implementation
  // ===================================================================
  void EmbInit(const RecTensor& keys, const RecTensor& init_values) override;
  void EmbInit(const RecTensor& keys, const InitStrategy& strategy) override;
  void EmbRead(const RecTensor& keys, RecTensor& values) override;
  void EmbWrite(const RecTensor& keys, const RecTensor& values) override;
  void EmbUpdate(const RecTensor& keys, const RecTensor& grads) override;
  void barrier() override;

  /**
   * @brief Gets the torch::Device this store is located on.
   * @return The torch::Device object.
   */
  torch::Device device() const { return device_; }

  // ===================================================================
  // Not implemented interface stubs
  // ===================================================================
  bool EmbExists(const RecTensor& keys) override;
  void EmbDelete(const RecTensor& keys) override;
  uint64_t EmbPrefetch(const RecTensor& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  uint64_t
  EmbWriteAsync(const RecTensor& keys, const RecTensor& values) override;
  bool IsWriteDone(uint64_t write_id) override;
  void WaitForWrite(uint64_t write_id) override;
  void SaveToFile(const std::string& path) override;
  void LoadFromFile(const std::string& path) override;

private:
  /**
   * @brief Initializes the main storage tensor on the first write operation.
   * The size is inferred from the keys and values provided.
   */
  void
  initialize_storage_if_needed(const RecTensor& keys, const RecTensor& values);

  torch::Device device_;
  std::optional<torch::Tensor>
      storage_; // The embedding table tensor on the GPU
  float learning_rate_;
  int64_t num_embeddings_;
  int64_t embedding_dim_;
};

} // namespace framework
} // namespace recstore
