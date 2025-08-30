#pragma once

#include "../../memory/persist_malloc.h"
#include "pointer.h"
#include "index.h"
#include "../dram/extendible_hash.h"
#include "lru_cache.h"     
#include <string>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/uio.h>

class ValueManager {
public:
    using LRUCache = tstarling::ThreadSafeLRUCache<Key_t, uint8_t>;

    UnifiedPointer last_promoted_ptr_ = UnifiedPointer();
    Index* index;

    ValueManager(
        const std::string& shm_file_path, size_t shm_capacity,
        const std::string& ssd_file_path, size_t ssd_capacity,
        const IndexConfig& index_config)
        : shm_manage(shm_file_path, shm_capacity),
          ssd_manage(ssd_file_path, ssd_capacity),
          index(new ExtendibleHash(index_config)),
          lru_(std::max<size_t>(4096, shm_capacity / 64))
    {
        fd_ssd = open(ssd_file_path.c_str(), O_RDWR);
        if (fd_ssd < 0) {
            LOG(ERROR) << "ssd open error";
        }
    }
    ValueManager(const ValueManager&) = delete;
    ValueManager& operator=(const ValueManager&) = delete;

    ~ValueManager() {
        if (fd_ssd >= 0) {
            close(fd_ssd);
        }
        delete index;
    }

    void DeleteValue(Key_t key, const UnifiedPointer& p) {
        std::lock_guard<std::mutex> lock(mutex_);
        switch (p.type()) {
            case UnifiedPointer::Type::Memory: {
                void* mem_ptr = p.asMemoryPointer();
                shm_manage.Free(mem_ptr); 
                // 从内存驻留集移除
                {
                    std::unique_lock<std::shared_mutex> g(resident_mu_);
                    resident_keys_.erase(key);
                }
                return;
            }
            case UnifiedPointer::Type::Disk: {
                uint64_t offset = p.asDiskPageId();
                char* disk_ptr = ssd_manage.GetMallocData(static_cast<int64>(offset));
                ssd_manage.Free(disk_ptr);
                return;
            }
            case UnifiedPointer::Type::PMem: {
                throw std::runtime_error("PMem not implemented");
            }
            default:
                LOG(ERROR) << "Invalid pointer type during deletion";
                return;
        }
    }

    void WriteValue(Key_t key, const std::string_view& value, unsigned tid) {
        // 先尝试内存
        UnifiedPointer p_write = WriteMem(key, value, tid);
        if (!p_write) {
            // 触发一次淘汰
            EvictLRUItem();
            p_write = WriteMem(key, value, tid);
        }
        if (!p_write) {
            p_write = WriteDisk(value);
        }
        index->Put(key, p_write.RawValue(), tid);
    }

    std::string RetrieveValue(Key_t key, unsigned tid) {
        std::string value_;
        uint64_t pointer = 0;
        index->Get(key, pointer, tid);
        UnifiedPointer p = UnifiedPointer::FromRaw(pointer);

        switch (p.type()) {
            case UnifiedPointer::Type::Memory: {
                value_ = ReadFromMem(key, p);
                break;
            }
            case UnifiedPointer::Type::Disk: {
                value_ = ReadFromDisk(key, p);
                break;
            }
            case UnifiedPointer::Type::PMem: {
                return "";
            }
            case UnifiedPointer::Type::Invalid:
            default:
                return "";
        }
        return value_;
    }

private:
    base::PersistLoopShmMalloc shm_manage;
    base::PersistLoopShmMalloc ssd_manage;
    int fd_ssd = -1;

    std::mutex mutex_;          
    std::mutex mutex_mem;       // shm 分配/释放
    std::mutex mutex_disk;      // ssd 分配/写盘

    LRUCache lru_;                                  
    std::unordered_set<Key_t> resident_keys_;       
    std::shared_mutex resident_mu_;                 

    void TouchLRU(Key_t key) {
        typename LRUCache::ConstAccessor ac;
        if (!lru_.find(ac, key)) {
            (void)lru_.insert(key, static_cast<uint8_t>(0));
        }
    }

    UnifiedPointer WriteMem(Key_t key, const std::string_view& value, unsigned /*tid*/) {
        std::lock_guard<std::mutex> lock(mutex_mem);
        if (value.size() > std::numeric_limits<uint16_t>::max()) {
            LOG(ERROR) << "Value too large for 16-bit length header";
            return UnifiedPointer();
        }

        const uint16_t data_len = static_cast<uint16_t>(value.size());
        const size_t total_size = sizeof(data_len) + data_len;

        char* ptr = shm_manage.New(static_cast<int>(total_size));
        if (!ptr) return UnifiedPointer();

        if (shm_manage.GetMallocOffset(ptr) == -1) {
            shm_manage.Free(ptr);
            return UnifiedPointer();
        }

        memcpy(ptr, &data_len, sizeof(data_len));
        if (data_len) {
            memcpy(ptr + sizeof(data_len), value.data(), data_len);
        }
        {
            std::unique_lock<std::shared_mutex> g(resident_mu_);
            resident_keys_.insert(key);
        }
        TouchLRU(key);

        return UnifiedPointer::FromMemory(ptr);
    }

    UnifiedPointer WriteDisk(const std::string_view& value) {
        std::lock_guard<std::mutex> lock(mutex_disk);

        if (value.size() > std::numeric_limits<uint16_t>::max()) {
            LOG(ERROR) << "Value too large for 16-bit length header (disk)";
            return UnifiedPointer();
        }

        const uint16_t data_len = static_cast<uint16_t>(value.size());
        const size_t total_size = sizeof(data_len) + data_len;

        char* ptr = ssd_manage.New(static_cast<int>(total_size));
        if (!ptr) return UnifiedPointer();

        const off_t pool_off = static_cast<off_t>(ssd_manage.GetMallocOffset(ptr));
        if (pool_off == -1) {
            ssd_manage.Free(ptr);
            return UnifiedPointer();
        }
        const off_t file_off = static_cast<off_t>(ssd_manage.DataBaseOffset()) + pool_off;

        uint8_t lenle[2] = {
            static_cast<uint8_t>(data_len & 0xFF),
            static_cast<uint8_t>((data_len >> 8) & 0xFF)
        };

        if (pwrite(fd_ssd, lenle, 2, file_off) != 2 ||
            (data_len &&
             pwrite(fd_ssd, value.data(), data_len, file_off + 2) != static_cast<ssize_t>(data_len))) {
            ssd_manage.Free(ptr);
            return UnifiedPointer();
        }
        if (fdatasync(fd_ssd) < 0) {
            ssd_manage.Free(ptr);
            return UnifiedPointer();
        }
        return UnifiedPointer::FromDiskPageId(static_cast<uint64_t>(pool_off));
    }

    std::string ReadFromMem(Key_t key, const UnifiedPointer& p) {
        std::string value_;
        const void* mem_ptr = p.asMemoryPointer();
        const uint8_t* bytes = static_cast<const uint8_t*>(mem_ptr);

        const uint16_t len = static_cast<uint16_t>(bytes[0]) |
                             static_cast<uint16_t>(bytes[1] << 8);
        const char* str_data = reinterpret_cast<const char*>(bytes + 2);
        value_.assign(str_data, len);

        // 更新 LRU
        TouchLRU(key);
        return value_;
    }

    std::string ReadFromDisk(Key_t key, const UnifiedPointer& p) {
        std::string value_;

        const off_t pool_off = static_cast<off_t>(p.asDiskPageId());
        const off_t file_off = static_cast<off_t>(ssd_manage.DataBaseOffset()) + pool_off;

        uint8_t lenle[2];
        if (pread(fd_ssd, lenle, 2, file_off) != 2) {
            LOG(ERROR) << "Read length failed";
            return {};
        }
        const uint16_t value_len = static_cast<uint16_t>(lenle[0]) |
                                   static_cast<uint16_t>(lenle[1] << 8);

        value_.resize(value_len);
        if (value_len &&
            pread(fd_ssd, value_.data(), value_len, file_off + 2) != static_cast<ssize_t>(value_len)) {
            LOG(ERROR) << "Read payload failed";
            return {};
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto mem_ptr = WriteMem(key, value_, /*tid*/0);
            if (!mem_ptr) {
                EvictLRUItem();
                mem_ptr = WriteMem(key, value_, /*tid*/0);
            }
            if (mem_ptr) {
                index->Delete(key);
                char* disk_ptr = ssd_manage.GetMallocData(static_cast<int64>(pool_off));
                ssd_manage.Free(disk_ptr);
                index->Put(key, mem_ptr.RawValue(), 0);
            }
        }
        // 访问更新 LRU
        TouchLRU(key);
        return value_;
    }

    void DeleteMemValue(Key_t /*key*/, const UnifiedPointer& p) {
        
    }

    // 选择一个key 进行淘汰
    bool EvictLRUItem() {
        std::vector<Key_t> keys;
        lru_.snapshotKeys(keys);
        if (keys.empty()) return false;

        Key_t victim = 0;
        bool found = false;

        {
            std::shared_lock<std::shared_mutex> g(resident_mu_);
            for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
                if (resident_keys_.find(*it) != resident_keys_.end()) {
                    victim = *it;
                    found = true;
                    break;
                }
            }
        }
        if (!found) return false;

        // 读取索引并确定其确实在内存
        uint64_t old_pointer = 0;
        index->Get(victim, old_pointer, 0);
        UnifiedPointer old_p = UnifiedPointer::FromRaw(old_pointer);
        if (old_p.type() != UnifiedPointer::Type::Memory) {
            // 非内存，移出驻留集
            std::unique_lock<std::shared_mutex> g(resident_mu_);
            resident_keys_.erase(victim);
            return false;
        }

        // 将其写入 SSD 并更新索引
        void* mem_ptr = old_p.asMemoryPointer();
        uint16_t data_len = 0;
        memcpy(&data_len, mem_ptr, sizeof(data_len));
        std::string value(static_cast<char*>(mem_ptr) + sizeof(data_len), data_len);

        auto ssd_ptr = WriteDisk(std::string_view(value.data(), value.size()));
        if (!ssd_ptr) return false;

        index->Delete(victim);
        shm_manage.Free(mem_ptr);
        index->Put(victim, ssd_ptr.RawValue(), 0);

        {
            std::unique_lock<std::shared_mutex> g(resident_mu_);
            resident_keys_.erase(victim);
        }
        return true;
    }
};
