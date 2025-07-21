#include "gpu_kv_store.h"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <stdexcept>

namespace recstore {
namespace framework {

// Helper to correctly wrap a RecTensor's data pointer into a torch::Tensor
inline torch::Tensor ToTorchTensor(const base::RecTensor& tensor, torch::Device device) {
  auto options = torch::TensorOptions()
                     .dtype(base::ToTorchDType(tensor.dtype()))
                     .device(device);
  // FIX: Handle empty tensors correctly to avoid creating a tensor from a null/invalid pointer.
  if (tensor.num_elements() == 0) {
    return torch::empty(tensor.shape_as_vector(), options);
  }
  return torch::from_blob(tensor.data(), tensor.shape_as_vector(), options);
}

GPUKVStore::GPUKVStore(int device_id)
    : device_(torch::kCUDA, device_id), learning_rate_(0.01f) {
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("CUDA is not available, cannot create GPUKVStore.");
  }
  if (device_id >= torch::cuda::device_count()) {
    throw std::runtime_error("Invalid device_id: " + std::to_string(device_id));
  }
  std::cout << "GPUKVStore: Initialized on device cuda:" << device_id << std::endl;
}

void GPUKVStore::EmbRead(const std::string& name, const RecTensor& keys, RecTensor& values) {
  c10::cuda::CUDAGuard device_guard(device_);
  auto values_tensor = ToTorchTensor(values, device_);

  // If the named storage doesn't exist or input keys are empty, return zeros.
  if (storage_map_.find(name) == storage_map_.end() || keys.num_elements() == 0) {
    values_tensor.zero_();
    return;
  }

  auto keys_tensor = ToTorchTensor(keys, device_);
  torch::index_select_out(values_tensor, storage_map_.at(name), 0, keys_tensor);
}

void GPUKVStore::EmbWrite(const std::string& name, const RecTensor& keys, const RecTensor& values) {
  c10::cuda::CUDAGuard device_guard(device_);
  
  if (keys.num_elements() == 0) {
      return;
  }

  auto keys_tensor = ToTorchTensor(keys, device_).to(torch::kInt64);
  auto values_tensor = ToTorchTensor(values, device_);
  int64_t embedding_dim = values.shape(1);

  // If the named tensor doesn't exist, create it (lazy initialization).
  if (storage_map_.find(name) == storage_map_.end()) {
      int64_t num_embeddings = keys_tensor.max().item<int64_t>() + 1;
      std::cout << "GPUKVStore: Lazily initializing storage '" << name << "' on " << device_
                << " with shape [" << num_embeddings << ", " << embedding_dim << "]" << std::endl;
      storage_map_[name] = torch::zeros({num_embeddings, embedding_dim}, values_tensor.options());
  }

  // Check if the storage needs to be resized to accommodate new, larger keys.
  auto max_key = keys_tensor.max().item<int64_t>();
  if (max_key >= storage_map_[name].size(0)) {
      int64_t current_size = storage_map_[name].size(0);
      int64_t new_size = max_key + 1;
      std::cout << "GPUKVStore: Resizing storage '" << name << "' from " << current_size
                << " to " << new_size << std::endl;
      auto new_storage = torch::zeros({new_size, embedding_dim}, storage_map_[name].options());
      new_storage.slice(0, 0, current_size) = storage_map_[name];
      storage_map_[name] = new_storage;
  }

  storage_map_[name].index_put_({keys_tensor}, values_tensor, false);
}

void GPUKVStore::EmbUpdate(const std::string& name, const RecTensor& keys, const RecTensor& grads) {
  c10::cuda::CUDAGuard device_guard(device_);

  if (storage_map_.find(name) == storage_map_.end() || keys.num_elements() == 0) {
    return;
  }

  auto keys_tensor = ToTorchTensor(keys, device_);
  auto grads_tensor = ToTorchTensor(grads, device_);

  storage_map_[name].index_add_(0, keys_tensor, grads_tensor * -learning_rate_);
}

void GPUKVStore::EmbInit(const std::string& name, const RecTensor& keys, const RecTensor& init_values) {
  EmbWrite(name, keys, init_values);
}

void GPUKVStore::EmbInit(const std::string& name, const RecTensor& keys, const InitStrategy& strategy) {
    c10::cuda::CUDAGuard device_guard(device_);
    int64_t embedding_dim = keys.shape(1); // This seems incorrect, should be from context
    auto values = torch::zeros({keys.shape(0), embedding_dim},
                               torch::dtype(torch::kFloat32).device(device_));
    auto rec_values = base::RecTensor(values.data_ptr(), {keys.shape(0), embedding_dim}, base::DataType::FLOAT32);
    EmbWrite(name, keys, rec_values);
}

void GPUKVStore::barrier() {
  c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
}

// Stubs for other optional APIs
#define NOT_IMPLEMENTED_WITH_NAME(name) throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented for GPUKVStore for name " + name)
#define NOT_IMPLEMENTED throw std::runtime_error(std::string(__FUNCTION__) + " is not implemented for GPUKVStore.")
bool GPUKVStore::EmbExists(const std::string& name, const RecTensor& keys) { NOT_IMPLEMENTED_WITH_NAME(name); }
void GPUKVStore::EmbDelete(const std::string& name, const RecTensor& keys) { NOT_IMPLEMENTED_WITH_NAME(name); }
uint64_t GPUKVStore::EmbPrefetch(const std::string& name, const RecTensor& keys) { NOT_IMPLEMENTED_WITH_NAME(name); }
uint64_t GPUKVStore::EmbWriteAsync(const std::string& name, const RecTensor& keys, const RecTensor& values) { NOT_IMPLEMENTED_WITH_NAME(name); }
void GPUKVStore::SaveToFile(const std::string& name, const std::string& path) { NOT_IMPLEMENTED_WITH_NAME(name); }
void GPUKVStore::LoadFromFile(const std::string& name, const std::string& path) { NOT_IMPLEMENTED_WITH_NAME(name); }
bool GPUKVStore::IsPrefetchDone(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
void GPUKVStore::WaitForPrefetch(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
bool GPUKVStore::IsWriteDone(uint64_t write_id) { NOT_IMPLEMENTED; }
void GPUKVStore::WaitForWrite(uint64_t write_id) { NOT_IMPLEMENTED; }
#undef NOT_IMPLEMENTED_WITH_NAME
#undef NOT_IMPLEMENTED

} // namespace framework
} // namespace recstore
