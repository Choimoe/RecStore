#include "gpu_kv_store.h"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <stdexcept>

namespace recstore {
namespace framework {

static std::optional<torch::Tensor> global_storage_;

inline torch::Tensor
ToTorchTensor(const base::RecTensor& tensor, torch::Device device) {
  auto options = torch::TensorOptions()
                     .dtype(base::ToTorchDType(tensor.dtype()))
                     .device(torch::kCPU);
  auto t = torch::from_blob(tensor.data(), tensor.shape_as_vector(), options);
  if (device.type() == torch::kCUDA) {
    if (t.numel() == 0) {
      return torch::empty(tensor.shape_as_vector(), options.device(device));
    }
    return t.to(device);
  }
  return t;
}

GPUKVStore::GPUKVStore(int device_id)
    : device_(torch::kCUDA, device_id),
      learning_rate_(0.01f),
      num_embeddings_(-1),
      embedding_dim_(-1) {
  if (!torch::cuda::is_available()) {
    throw std::runtime_error(
        "CUDA is not available, cannot create GPUKVStore.");
  }
  if (device_id >= torch::cuda::device_count()) {
    throw std::runtime_error("Invalid device_id: " + std::to_string(device_id));
  }
  std::cout << "GPUKVStore: Initialized on device cuda:" << device_id
            << std::endl;
}

void GPUKVStore::initialize_storage_if_needed(const RecTensor& keys,
                                              const RecTensor& values) {
  if (global_storage_.has_value()) {
    return;
  }

  // Infer storage size from the first write operation.
  // We assume the largest key seen defines the number of embeddings.
  auto keys_tensor = ToTorchTensor(keys, device_).to(torch::kInt64);
  auto max_key_val =
      keys_tensor.numel() > 0 ? keys_tensor.max().item<int64_t>() : 0;

  num_embeddings_ = max_key_val + 1;
  embedding_dim_  = values.shape(1);

  std::cout << "GPUKVStore: Lazily initializing storage on " << device_
            << " with shape [" << num_embeddings_ << ", " << embedding_dim_
            << "]" << std::endl;

  global_storage_ = torch::zeros(
      {num_embeddings_, embedding_dim_},
      torch::dtype(base::ToTorchDType(values.dtype())).device(device_));
}

void GPUKVStore::EmbRead(const RecTensor& keys, RecTensor& values) {
  c10::cuda::CUDAGuard device_guard(device_);

  if (!global_storage_.has_value() || keys.num_elements() == 0) {
    auto values_tensor = ToTorchTensor(values, device_);
    values_tensor.zero_();
    return;
  }

  auto keys_tensor   = ToTorchTensor(keys, device_);
  auto values_tensor = ToTorchTensor(values, device_);

  // Use PyTorch's highly optimized index_select.
  torch::index_select_out(
      values_tensor, global_storage_.value(), 0, keys_tensor);
}

void GPUKVStore::EmbWrite(const RecTensor& keys, const RecTensor& values) {
  c10::cuda::CUDAGuard device_guard(device_);

  if (keys.num_elements() == 0)
    return;
  initialize_storage_if_needed(keys, values);

  auto keys_tensor   = ToTorchTensor(keys, device_);
  auto values_tensor = ToTorchTensor(values, device_);

  // Use PyTorch's highly optimized index_put_.
  // This operation is performed entirely on the GPU.
  global_storage_->index_put_({keys_tensor}, values_tensor, false);
}

void GPUKVStore::EmbUpdate(const RecTensor& keys, const RecTensor& grads) {
  c10::cuda::CUDAGuard device_guard(device_);

  if (!global_storage_.has_value() || keys.num_elements() == 0) {
    // Cannot update if not initialized or keys为空
    std::cerr << "Warning: EmbUpdate called before any EmbWrite or with empty "
                 "keys. Gradients will be ignored."
              << std::endl;
    return;
  }

  auto keys_tensor  = ToTorchTensor(keys, device_);
  auto grads_tensor = ToTorchTensor(grads, device_);

  // Use PyTorch's index_add_ for efficient sparse updates.
  // This performs: storage[keys] += grads * (-learning_rate)
  global_storage_->index_add_(0, keys_tensor, grads_tensor * -learning_rate_);
}

void GPUKVStore::EmbInit(const RecTensor& keys, const RecTensor& init_values) {
  // For GPU store, Init is the same as a Write.
  EmbWrite(keys, init_values);
}

void GPUKVStore::EmbInit(const RecTensor& keys, const InitStrategy& strategy) {
  // This implementation initializes with zeros.
  // A more complete version would handle different strategies.
  c10::cuda::CUDAGuard device_guard(device_);
  initialize_storage_if_needed(keys, keys); // Pass dummy values to infer shape

  auto values     = torch::zeros({keys.shape(0), embedding_dim_},
                             torch::dtype(torch::kFloat32).device(device_));
  auto rec_values = base::RecTensor(
      values.data_ptr(),
      {keys.shape(0), embedding_dim_},
      base::DataType::FLOAT32);

  EmbWrite(keys, rec_values);
}

void GPUKVStore::barrier() {
  // For a single GPU, we can synchronize the device stream.
  c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
}

void GPUKVStore::SaveToFile(const std::string& path) {
  if (!global_storage_.has_value()) {
    throw std::runtime_error("No storage to save.");
  }
  auto cpu_tensor = global_storage_.value().to(torch::kCPU);
  torch::save(cpu_tensor, path);
}

void GPUKVStore::LoadFromFile(const std::string& path) {
  torch::Tensor loaded;
  torch::load(loaded, path);
  global_storage_ = loaded.to(device_);
  num_embeddings_ = global_storage_->size(0);
  embedding_dim_  = global_storage_->size(1);
}

#define NOT_IMPLEMENTED                                                        \
  throw std::runtime_error(                                                    \
      __FUNCTION__ + std::string(" is not implemented for GPUKVStore."))
bool GPUKVStore::EmbExists(const RecTensor& keys) { NOT_IMPLEMENTED; }
void GPUKVStore::EmbDelete(const RecTensor& keys) { NOT_IMPLEMENTED; }
uint64_t GPUKVStore::EmbPrefetch(const RecTensor& keys) { NOT_IMPLEMENTED; }
bool GPUKVStore::IsPrefetchDone(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
void GPUKVStore::WaitForPrefetch(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
uint64_t
GPUKVStore::EmbWriteAsync(const RecTensor& keys, const RecTensor& values) {
  NOT_IMPLEMENTED;
}
bool GPUKVStore::IsWriteDone(uint64_t write_id) { NOT_IMPLEMENTED; }
void GPUKVStore::WaitForWrite(uint64_t write_id) { NOT_IMPLEMENTED; }
#undef NOT_IMPLEMENTED

} // namespace framework
} // namespace recstore
