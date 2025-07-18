#include "shm_kv_store.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>  // For std::getenv
#include <unistd.h> // For sleep
#include "ucc_context.h"

namespace recstore {
namespace framework {

SharedMemoryKVStore::SharedMemoryKVStore(const char* shm_name, size_t shm_size)
    : shm_name_(shm_name), learning_rate_(0.01f) {
  ucc_context_ = &UCCContext::GetInstance();

  try {
    if (ucc_context_->GetRank() == 0) {
      bip::shared_memory_object::remove(shm_name_.c_str());
      std::cout << "Master process (rank 0): Creating shared memory segment '"
                << shm_name_ << "'." << std::endl;
      segment_ = bip::managed_shared_memory(
          bip::create_only, shm_name_.c_str(), shm_size);
    }

    ucc_context_->Barrier();

    if (ucc_context_->GetRank() != 0) {
      std::cout << "Slave process (rank " << ucc_context_->GetRank()
                << "): Attaching to shared memory segment." << std::endl;
      segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
    }

    initialize_shm();

    ucc_context_->Barrier();

  } catch (const bip::interprocess_exception& ex) {
    std::cerr << "Rank " << ucc_context_->GetRank()
              << " Boost Interprocess Exception: " << ex.what() << std::endl;
    throw;
  }
}

SharedMemoryKVStore::~SharedMemoryKVStore() {
  ucc_context_->Barrier();
  if (ucc_context_->GetRank() == 0) {
    std::cout << "Master process (rank 0): Tearing down shared memory segment '"
              << shm_name_ << "'." << std::endl;
    bip::shared_memory_object::remove(shm_name_.c_str());
  }
}

void SharedMemoryKVStore::initialize_shm() {
  if (ucc_context_->GetRank() == 0) {
    ShmPairAllocator alloc_inst(segment_.get_segment_manager());
    store_ = segment_.construct<ShmUnorderedMap>("KVStore")(alloc_inst);
    mtx_   = segment_.construct<bip::interprocess_mutex>("KVStoreMutex")();
    embedding_dim_ = segment_.construct<int64_t>("EmbeddingDim")(-1);
  } else {
    store_ = segment_.find<ShmUnorderedMap>("KVStore").first;
    mtx_   = segment_.find<bip::interprocess_mutex>("KVStoreMutex").first;
    embedding_dim_ = segment_.find<int64_t>("EmbeddingDim").first;
    if (!store_ || !mtx_ || !embedding_dim_) {
      throw std::runtime_error(
          "Slave process could not find essential objects in shared memory.");
    }
  }
}

// bool SharedMemoryKVStore::is_master_process() {
//     // Using MPI/UCC to determine the rank of the process.
//     // If the environment variable does not exist, it is assumed to be a
//     single process mode, that is, the master process. const char* rank_str =
//     std::getenv("OMPI_COMM_WORLD_RANK"); if (rank_str == nullptr) {
//         rank_str = std::getenv("PMI_RANK"); // For other launchers like SLURM
//     }

//     if (rank_str == nullptr) {
//         return true; // cannot determine the rank, assume it is the master
//         process.
//     }
//     return std::stoi(rank_str) == 0;
// }

void SharedMemoryKVStore::EmbRead(const RecTensor& keys, RecTensor& values) {
  bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
  const int64_t emb_dim = values.shape(1);

  const uint64_t* key_data = keys.data_as<uint64_t>();
  float* value_data        = values.data_as<float>();
  const int64_t num_keys   = keys.shape(0);

  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    auto it      = store_->find(key);
    if (it == store_->end()) {
      // 如果 key 不存在，则填充 0
      std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
    } else {
      // 从共享内存的 ShmVector 拷贝到进程本地的 RecTensor
      std::copy(it->second.begin(), it->second.end(), value_data + i * emb_dim);
    }
  }
}

void SharedMemoryKVStore::EmbWrite(const RecTensor& keys,
                                   const RecTensor& values) {
  bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
  const int64_t emb_dim = values.shape(1);
  if (*embedding_dim_ == -1) {
    *embedding_dim_ = emb_dim;
  } else if (*embedding_dim_ != emb_dim) {
    throw std::runtime_error("SharedMemoryKVStore Error: Inconsistent "
                             "embedding dimension for write.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* value_data  = values.data_as<float>();
  const int64_t num_keys   = keys.shape(0);
  auto* segment_manager    = segment_.get_segment_manager();
  ShmFloatAllocator float_alloc(segment_manager);

  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key       = key_data[i];
    const float* start = value_data + i * emb_dim;
    const float* end   = start + emb_dim;
    store_->erase(key); // 先删除
    store_->emplace(std::piecewise_construct,
                    std::forward_as_tuple(key),
                    std::forward_as_tuple(start, end, float_alloc));
  }
}

void SharedMemoryKVStore::EmbUpdate(const RecTensor& keys,
                                    const RecTensor& grads) {
  bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
  const int64_t emb_dim = grads.shape(1);
  if (*embedding_dim_ == -1) {
    *embedding_dim_ = emb_dim;
  } else if (*embedding_dim_ != emb_dim) {
    throw std::runtime_error("SharedMemoryKVStore Error: Inconsistent "
                             "embedding dimension for update.");
  }

  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* grad_data   = grads.data_as<float>();
  const int64_t num_keys   = keys.shape(0);

  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    auto it      = store_->find(key);
    if (it != store_->end()) {
      // 在共享内存中直接进行梯度更新
      for (int64_t j = 0; j < emb_dim; ++j) {
        it->second[j] -= learning_rate_ * grad_data[i * emb_dim + j];
      }
    }
  }
}

void SharedMemoryKVStore::EmbInit(const RecTensor& keys,
                                  const RecTensor& init_values) {
  EmbWrite(keys, init_values);
}

void SharedMemoryKVStore::EmbInit(const RecTensor& keys,
                                  const InitStrategy& strategy) {
  bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
  if (*embedding_dim_ == -1) {
    throw std::runtime_error("SharedMemoryKVStore Error: Embedding dimension "
                             "must be set via EmbWrite before EmbInit.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const int64_t num_keys   = keys.shape(0);
  ShmFloatAllocator float_alloc(segment_.get_segment_manager());

  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    store_->erase(key); // 先删除
    store_->emplace(std::piecewise_construct,
                    std::forward_as_tuple(key),
                    std::forward_as_tuple(*embedding_dim_, 0.0f, float_alloc));
  }
}

// Not implemented interface
#define NOT_IMPLEMENTED                                                        \
  throw std::runtime_error(__FUNCTION__ + std::string(" is not implemented."))
bool SharedMemoryKVStore::EmbExists(const RecTensor& keys) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::EmbDelete(const RecTensor& keys) { NOT_IMPLEMENTED; }
uint64_t SharedMemoryKVStore::EmbPrefetch(const RecTensor& keys) {
  NOT_IMPLEMENTED;
}
bool SharedMemoryKVStore::IsPrefetchDone(uint64_t prefetch_id) {
  NOT_IMPLEMENTED;
}
void SharedMemoryKVStore::WaitForPrefetch(uint64_t prefetch_id) {
  NOT_IMPLEMENTED;
}
uint64_t SharedMemoryKVStore::EmbWriteAsync(const RecTensor& keys,
                                            const RecTensor& values) {
  NOT_IMPLEMENTED;
}
bool SharedMemoryKVStore::IsWriteDone(uint64_t write_id) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::WaitForWrite(uint64_t write_id) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::SaveToFile(const std::string& path) {
  NOT_IMPLEMENTED;
}
void SharedMemoryKVStore::LoadFromFile(const std::string& path) {
  NOT_IMPLEMENTED;
}
#undef NOT_IMPLEMENTED

} // namespace framework
} // namespace recstore
