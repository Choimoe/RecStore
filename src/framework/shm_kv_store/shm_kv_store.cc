#include "shm_kv_store.h"
#include "ucc_context.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <mpi.h> // Required for MPI_Abort
#include <thread> // Required for std::this_thread::sleep_for
#include <chrono> // Required for std::chrono::milliseconds

namespace recstore {
namespace framework {

SharedMemoryKVStore::SharedMemoryKVStore(const char* shm_name, size_t shm_size)
    : shm_name_(shm_name), learning_rate_(0.01f) {
    
    ucc_context_ = &UCCContext::GetInstance();

    // Simplified initialization sequence to avoid deadlocks
    try {
        // Step 1: Clean up any existing shared memory segment (only rank 0)
        if (ucc_context_->GetRank() == 0) {
            try {
                bip::shared_memory_object::remove(shm_name_.c_str());
                std::cout << "Master process (rank 0): Cleaned up existing shared memory segment." << std::endl;
            } catch (const bip::interprocess_exception& ex) {
                // Ignore errors if segment doesn't exist
                std::cout << "Master process (rank 0): No existing segment to clean up." << std::endl;
            }
        }
        
        // Step 2: Synchronize before creation
        std::cout << "Rank " << ucc_context_->GetRank() << ": Synchronizing before shared memory creation..." << std::endl;
        ucc_context_->Barrier();
        
        // Step 3: Create shared memory segment (only rank 0)
        if (ucc_context_->GetRank() == 0) {
            std::cout << "Master process (rank 0): Creating shared memory segment '" << shm_name_ << "' with size " << shm_size << " bytes..." << std::endl;
            segment_ = bip::managed_shared_memory(bip::create_only, shm_name_.c_str(), shm_size);
            std::cout << "Master process (rank 0): Shared memory segment created successfully." << std::endl;
        }
        
        // Step 4: All processes wait here to ensure creation is complete
        ucc_context_->Barrier();
        
        // Step 5: Non-master processes attach to the segment
        if (ucc_context_->GetRank() != 0) {
            std::cout << "Slave process (rank " << ucc_context_->GetRank() << "): Attaching to shared memory segment..." << std::endl;
            
            // Add retry logic for attachment
            int retry_count = 0;
            const int max_retries = 5;
            while (retry_count < max_retries) {
                try {
                    segment_ = bip::managed_shared_memory(bip::open_only, shm_name_.c_str());
                    std::cout << "Slave process (rank " << ucc_context_->GetRank() << "): Successfully attached to shared memory segment." << std::endl;
                    break;
                } catch (const bip::interprocess_exception& ex) {
                    retry_count++;
                    std::cerr << "Slave process (rank " << ucc_context_->GetRank() << "): Failed to attach (attempt " << retry_count << "/" << max_retries << "): " << ex.what() << std::endl;
                    if (retry_count >= max_retries) {
                        throw;
                    }
                    // Wait a bit before retrying
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }

        // Step 6: All processes initialize their internal pointers
        std::cout << "Rank " << ucc_context_->GetRank() << ": Initializing shared memory data structures..." << std::endl;
        initialize_shm();
        
        // Step 7: Always reset embedding dimension for new instances
        // This ensures that each test can use its own embedding dimension
        if (embedding_dim_) {
            *embedding_dim_ = -1;
            std::cout << "Rank " << ucc_context_->GetRank() << ": Reset embedding dimension to -1 for new instance." << std::endl;
        }
        
        // Step 8: Final synchronization to ensure all setup is complete
        ucc_context_->Barrier();
        std::cout << "Rank " << ucc_context_->GetRank() << ": SharedMemoryKVStore initialization completed successfully." << std::endl;

    } catch (const bip::interprocess_exception& ex) {
        std::cerr << "Rank " << ucc_context_->GetRank() << " Boost Interprocess Exception: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const std::exception& ex) {
        std::cerr << "Rank " << ucc_context_->GetRank() << " Exception during initialization: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

SharedMemoryKVStore::~SharedMemoryKVStore() {
    ucc_context_->Barrier();
    if (ucc_context_->GetRank() == 0) {
        std::cout << "Master process (rank 0): Tearing down shared memory segment '" << shm_name_ << "'." << std::endl;
        bip::shared_memory_object::remove(shm_name_.c_str());
    }
}

void SharedMemoryKVStore::initialize_shm() {
  if (ucc_context_->GetRank() == 0) {
    std::cout << "Rank 0: Creating shared memory data structures..." << std::endl;
    ShmPairAllocator alloc_inst(segment_.get_segment_manager());
    store_ = segment_.construct<ShmUnorderedMap>("KVStore")(alloc_inst);
    mtx_   = segment_.construct<bip::interprocess_mutex>("KVStoreMutex")();
    embedding_dim_ = segment_.construct<int64_t>("EmbeddingDim")(-1);
    std::cout << "Rank 0: Shared memory data structures created successfully." << std::endl;
  } else {
    std::cout << "Rank " << ucc_context_->GetRank() << ": Finding shared memory data structures..." << std::endl;
    
    // Add retry logic for finding objects
    int retry_count = 0;
    const int max_retries = 10;
    while (retry_count < max_retries) {
      store_ = segment_.find<ShmUnorderedMap>("KVStore").first;
      mtx_   = segment_.find<bip::interprocess_mutex>("KVStoreMutex").first;
      embedding_dim_ = segment_.find<int64_t>("EmbeddingDim").first;
      
      if (store_ && mtx_ && embedding_dim_) {
        std::cout << "Rank " << ucc_context_->GetRank() << ": Successfully found all shared memory objects." << std::endl;
        break;
      }
      
      retry_count++;
      std::cout << "Rank " << ucc_context_->GetRank() << ": Failed to find objects (attempt " << retry_count << "/" << max_retries << ")" << std::endl;
      std::cout << "  store_: " << (store_ ? "found" : "not found") << std::endl;
      std::cout << "  mtx_: " << (mtx_ ? "found" : "not found") << std::endl;
      std::cout << "  embedding_dim_: " << (embedding_dim_ ? "found" : "not found") << std::endl;
      
      if (retry_count >= max_retries) {
        throw std::runtime_error(
            "Slave process could not find essential objects in shared memory after " + 
            std::to_string(max_retries) + " attempts.");
      }
      
      // Wait before retrying
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}

void SharedMemoryKVStore::EmbRead(const RecTensor& keys, RecTensor& values) {
    bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
    const int64_t emb_dim = values.shape(1);
    const uint64_t* key_data = keys.data_as<uint64_t>();
    float* value_data = values.data_as<float>();
    const int64_t num_keys = keys.shape(0);

    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        auto it = store_->find(key);
        if (it == store_->end()) {
            std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
        } else {
            std::copy(it->second.begin(), it->second.end(), value_data + i * emb_dim);
        }
    }
}

void SharedMemoryKVStore::EmbWrite(const RecTensor& keys, const RecTensor& values) {
    bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
    const int64_t emb_dim = values.shape(1);
    
    // Check and set embedding dimension
    if (*embedding_dim_ == -1) {
        *embedding_dim_ = emb_dim;
        std::cout << "Rank " << ucc_context_->GetRank() << ": Set embedding dimension to " << emb_dim << std::endl;
    } else if (*embedding_dim_ != emb_dim) {
        std::cerr << "Rank " << ucc_context_->GetRank() << ": Embedding dimension mismatch! Expected " 
                  << *embedding_dim_ << ", got " << emb_dim << std::endl;
        throw std::runtime_error("Inconsistent embedding dimension for write. Expected " + 
                                std::to_string(*embedding_dim_) + ", got " + std::to_string(emb_dim));
    }
    
    const uint64_t* key_data = keys.data_as<uint64_t>();
    const float* value_data = values.data_as<float>();
    const int64_t num_keys = keys.shape(0);
    ShmFloatAllocator float_alloc(segment_.get_segment_manager());

    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        const float* start = value_data + i * emb_dim;
        const float* end = start + emb_dim;
        store_->erase(key);
        store_->emplace(std::piecewise_construct,
                        std::forward_as_tuple(key),
                        std::forward_as_tuple(start, end, float_alloc));
    }
}

// **FIX**: This is the correct, distributed implementation of EmbUpdate.
void SharedMemoryKVStore::EmbUpdate(const RecTensor& keys, const RecTensor& grads) {
    const int64_t num_keys = keys.shape(0);
    
    std::cout << "Rank " << ucc_context_->GetRank() << ": EmbUpdate called with num_keys=" << num_keys 
              << ", grads.shape=[" << grads.shape(0) << ", " << grads.shape(1) << "]" << std::endl;
    
    // Check if this is a valid update (non-zero gradients)
    bool has_valid_gradients = false;
    if (num_keys > 0) {
        const float* grad_data = grads.data_as<float>();
        const int64_t total_grad_elements = num_keys * grads.shape(1);
        
        // Check if any gradient is non-zero
        for (int64_t i = 0; i < total_grad_elements; ++i) {
            if (grad_data[i] != 0.0f) {
                has_valid_gradients = true;
                break;
            }
        }
        
        std::cout << "Rank " << ucc_context_->GetRank() << ": Gradient check: has_valid_gradients=" 
                  << (has_valid_gradients ? "true" : "false") << std::endl;
    }
    
    // 1. Serialize local gradients into a buffer.
    // Format: [int64_t num_keys][uint64_t key1...][float grad1...]
    std::vector<char> send_buffer;
    
    if (num_keys > 0 && has_valid_gradients) {
        int64_t emb_dim = grads.shape(1);
        size_t keys_size = num_keys * sizeof(uint64_t);
        size_t grads_size = num_keys * emb_dim * sizeof(float);
        send_buffer.resize(sizeof(int64_t) + keys_size + grads_size);

        memcpy(send_buffer.data(), &num_keys, sizeof(int64_t));
        memcpy(send_buffer.data() + sizeof(int64_t), keys.data(), keys_size);
        memcpy(send_buffer.data() + sizeof(int64_t) + keys_size, grads.data(), grads_size);
        
        std::cout << "Rank " << ucc_context_->GetRank() << ": EmbUpdate: num_keys=" << num_keys 
                  << ", emb_dim=" << emb_dim << ", buffer_size=" << send_buffer.size() << std::endl;
    } else {
        // Empty update or zero gradients: just send the count (0)
        send_buffer.resize(sizeof(int64_t));
        int64_t zero_keys = 0;
        memcpy(send_buffer.data(), &zero_keys, sizeof(int64_t));
        
        std::cout << "Rank " << ucc_context_->GetRank() << ": EmbUpdate: empty/zero update, buffer_size=" << send_buffer.size() << std::endl;
    }
    
    // 2. Use UCC to gather all gradient buffers from all processes.
    std::vector<char> recv_buffer = ucc_context_->Allgatherv(send_buffer);

    // 3. Let Rank 0 apply all updates to the shared memory to avoid lock contention.
    if (ucc_context_->GetRank() == 0) {
        bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);

        const char* current_ptr = recv_buffer.data();
        const std::vector<int>& counts = ucc_context_->GetLastRecvCounts();

        // Iterate through the data received from each rank
        for (int rank_idx = 0; rank_idx < ucc_context_->GetWorldSize(); ++rank_idx) {
            if (counts[rank_idx] == 0) continue; // Skip ranks with no updates

            // Deserialize the data block from this rank
            int64_t remote_num_keys = 0;
            memcpy(&remote_num_keys, current_ptr, sizeof(int64_t));
            const char* data_start = current_ptr + sizeof(int64_t);

            if (remote_num_keys == 0) {
                current_ptr += counts[rank_idx];
                continue;
            }

            // Infer embedding dimension from the first non-empty update
            if (*embedding_dim_ == -1) {
                size_t keys_bytes = remote_num_keys * sizeof(uint64_t);
                size_t grads_bytes = counts[rank_idx] - sizeof(int64_t) - keys_bytes;
                *embedding_dim_ = grads_bytes / (remote_num_keys * sizeof(float));
            }
            
            const uint64_t* remote_key_data = reinterpret_cast<const uint64_t*>(data_start);
            const float* remote_grad_data = reinterpret_cast<const float*>(data_start + remote_num_keys * sizeof(uint64_t));

            // Apply gradients
            for (int64_t i = 0; i < remote_num_keys; ++i) {
                auto it = store_->find(remote_key_data[i]);
                if (it != store_->end()) {
                    for (int64_t j = 0; j < *embedding_dim_; ++j) {
                        it->second[j] -= learning_rate_ * remote_grad_data[i * (*embedding_dim_) + j];
                    }
                }
            }
            current_ptr += counts[rank_idx];
        }
    }

    // 4. Final barrier to ensure all processes wait for rank 0 to finish updates.
    ucc_context_->Barrier();
}

void SharedMemoryKVStore::EmbInit(const RecTensor& keys, const RecTensor& init_values) {
    EmbWrite(keys, init_values);
}

void SharedMemoryKVStore::EmbInit(const RecTensor& keys, const InitStrategy& strategy) {
    bip::scoped_lock<bip::interprocess_mutex> lock(*mtx_);
    if (*embedding_dim_ == -1) {
        throw std::runtime_error("Embedding dimension must be set before init.");
    }
    const uint64_t* key_data = keys.data_as<uint64_t>();
    const int64_t num_keys = keys.shape(0);
    ShmFloatAllocator float_alloc(segment_.get_segment_manager());

    for (int64_t i = 0; i < num_keys; ++i) {
        uint64_t key = key_data[i];
        store_->erase(key);
        store_->emplace(std::piecewise_construct,
                        std::forward_as_tuple(key),
                        std::forward_as_tuple(*embedding_dim_, 0.0f, float_alloc));
    }
}

void SharedMemoryKVStore::barrier() {
    if (ucc_context_) {
        ucc_context_->Barrier();
    }
}

void SharedMemoryKVStore::reset_embedding_dimension() {
    if (embedding_dim_) {
        *embedding_dim_ = -1;
        std::cout << "Rank " << ucc_context_->GetRank() << ": Reset embedding dimension to -1." << std::endl;
    }
}

#define NOT_IMPLEMENTED throw std::runtime_error(__FUNCTION__ + std::string(" is not implemented."))
bool SharedMemoryKVStore::EmbExists(const RecTensor& keys) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::EmbDelete(const RecTensor& keys) { NOT_IMPLEMENTED; }
uint64_t SharedMemoryKVStore::EmbPrefetch(const RecTensor& keys) { NOT_IMPLEMENTED; }
bool SharedMemoryKVStore::IsPrefetchDone(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::WaitForPrefetch(uint64_t prefetch_id) { NOT_IMPLEMENTED; }
uint64_t SharedMemoryKVStore::EmbWriteAsync(const RecTensor& keys, const RecTensor& values) { NOT_IMPLEMENTED; }
bool SharedMemoryKVStore::IsWriteDone(uint64_t write_id) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::WaitForWrite(uint64_t write_id) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::SaveToFile(const std::string& path) { NOT_IMPLEMENTED; }
void SharedMemoryKVStore::LoadFromFile(const std::string& path) { NOT_IMPLEMENTED; }
#undef NOT_IMPLEMENTED

} // namespace framework
} // namespace recstore
