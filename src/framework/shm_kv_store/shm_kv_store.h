#pragma once

#include "framework/op.h"
#include <string>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace recstore {
namespace framework {

namespace bip = boost::interprocess;

using ShmFloatAllocator = bip::allocator<float, bip::managed_shared_memory::segment_manager>;
using ShmVector = bip::vector<float, ShmFloatAllocator>;
using ShmPair = std::pair<const uint64_t, ShmVector>;
using ShmPairAllocator = bip::allocator<ShmPair, bip::managed_shared_memory::segment_manager>;
using ShmUnorderedMap = boost::unordered_map<uint64_t, ShmVector, std::hash<uint64_t>, std::equal_to<uint64_t>, ShmPairAllocator>;

class SharedMemoryKVStore : public CommonOp {
public:
    SharedMemoryKVStore(const char* shm_name, size_t shm_size);
    virtual ~SharedMemoryKVStore();

    // ===================================================================
    // CommonOp core interface
    // ===================================================================
    void EmbInit(const RecTensor& keys, const RecTensor& init_values) override;
    void EmbInit(const RecTensor& keys, const InitStrategy& strategy) override;
    void EmbRead(const RecTensor& keys, RecTensor& values) override;
    void EmbWrite(const RecTensor& keys, const RecTensor& values) override;
    void EmbUpdate(const RecTensor& keys, const RecTensor& grads) override;

    // ===================================================================
    // Not implemented interface
    // ===================================================================
    bool EmbExists(const RecTensor& keys) override;
    void EmbDelete(const RecTensor& keys) override;
    uint64_t EmbPrefetch(const RecTensor& keys) override;
    bool IsPrefetchDone(uint64_t prefetch_id) override;
    void WaitForPrefetch(uint64_t prefetch_id) override;
    uint64_t EmbWriteAsync(const RecTensor& keys, const RecTensor& values) override;
    bool IsWriteDone(uint64_t write_id) override;
    void WaitForWrite(uint64_t write_id) override;
    void SaveToFile(const std::string& path) override;
    void LoadFromFile(const std::string& path) override;

private:
    void initialize_shm();
    bool is_master_process();

    std::string shm_name_;
    bip::managed_shared_memory segment_;
    
    ShmUnorderedMap* store_ = nullptr;
    bip::interprocess_mutex* mtx_ = nullptr;
    int64_t* embedding_dim_ = nullptr;
    
    float learning_rate_;
};

} // namespace framework
} // namespace recstore
