#pragma once

#include "../../memory/persist_malloc.h"
#include "pointer.h"
#include <string_view>
#include <mutex>

class ValueManager {
public:
    ValueManager(
        // const std::string& shm_file_path, size_t shm_capacity,
            const std::string& ssd_file_path, size_t ssd_capacity):
        // shm_manage(shm_file_path, shm_capacity),
        ssd_manage(ssd_file_path, ssd_capacity)
    {
        fd_ssd = open(ssd_file_path.c_str(),O_RDWR);
        if(fd_ssd < 0){
            LOG(ERROR) << "ssd open error";
        }
    }

    ~ValueManager() {
        if (fd_ssd >= 0) {
            close(fd_ssd);
        }
    }

    // void record_value_access(const std::string& value) {
    //     hot_value_cms.access_a_key(Slice(value));
    // }

    std::string DeleteValue(const UnifiedPointer& p) {
        std::string old_value = this->RetrieveValue(p);
        std::lock_guard<std::mutex> lock(mutex_);
        switch (p.type()) {
            case UnifiedPointer::Type::Memory: {
                // void* mem_ptr = p.asMemoryPointer();
                // // 获取实际分配的内存起始地址（包含长度前缀）
                // if(shm_manage.Free(mem_ptr)){
                //     return old_value;
                // }
                return nullptr;
            }
            case UnifiedPointer::Type::Disk: {
                uint64_t offset = p.asDiskPageId();
                // 通过偏移量获取原始指针
                char* disk_ptr = ssd_manage.GetMallocData(static_cast<int64>(offset));
                if(ssd_manage.Free(disk_ptr)){
                    return old_value;
                }
                return nullptr;
            }
            case UnifiedPointer::Type::PMem: {
                throw std::runtime_error("PMem not implemented");
                return nullptr;
            }
            default:
                LOG(ERROR) << "Invalid pointer type during deletion";
                return nullptr;
        }
    }

    UnifiedPointer WriteValue(const std::string_view& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        // 优先尝试在内存中写入
        // if (auto ptr = WriteMem(value); ptr) {
        //     return ptr;
        // }
        // 内存写入失败时尝试SSD
        return WriteDisk(value);
    }

    std::string RetrieveValue(const UnifiedPointer& p) {
        // UnifiedPointer p = UnifiedPointer::FromRaw(raw_value);
        if (p.type() != UnifiedPointer::Type::Invalid) {
            // std::cout << "Raw pointer info: " << p.toString() << std::endl;
        } else {
            // std::cout << "Raw pointer info: [Invalid Pointer]" << std::endl;
        }
        std::string value_;
        switch (p.type()) { // TODO: retrieve value from different tier
            case UnifiedPointer::Type::Memory: {
                void* mem_ptr = p.asMemoryPointer();
                const uint8_t* bytes = static_cast<const uint8_t*>(mem_ptr);
                // 从首部两个字节解析长度（小端序）
                uint16_t len = static_cast<uint16_t>(bytes[0]) | 
                            (static_cast<uint16_t>(bytes[1]) << 8);
                const char* str_data = reinterpret_cast<const char*>(bytes + 2);
                
                value_.assign(str_data, len);
                break;
            }
            case UnifiedPointer::Type::Disk: {
                off_t pointer = p.asDiskPageId();
                uint16_t value_len;
                ssize_t bytes_read = pread(fd_ssd, &value_len, sizeof(value_len), pointer);
                if (bytes_read != sizeof(value_len)) {
                    value_.clear();  // 返回空字符串表示错误
                    LOG(ERROR) << "Read failed";
                }
                // 读取实际数据
                value_.resize(value_len);
                bytes_read = pread(fd_ssd, value_.data(), value_len, pointer + sizeof(value_len));
                if (bytes_read != value_len) {
                    if (bytes_read < 0) {
                        perror("pread data failed");
                    } else {
                        perror("Incomplete data read at pointer ");
                    }
                }
                break;
            }
            case UnifiedPointer::Type::PMem: {
                // throw std::runtime_error("PMem not implemented"); 
                return "";  
            }
            case UnifiedPointer::Type::Invalid:
                // throw std::runtime_error("Invalid pointer type");
                return "";  
            default:
                // throw std::runtime_error("Unknown pointer type");
                return "";  
            }
        return value_;
    }

private:
    // base::PersistLoopShmMalloc shm_manage;
    base::PersistLoopShmMalloc ssd_manage;
    int fd_ssd = -1;
    std::mutex mutex_;

    // UnifiedPointer WriteMem(const std::string_view& value) {
    //     uint16_t data_len = value.size();
    //     size_t total_size = data_len + sizeof(data_len);
    //     char* ptr = shm_manage.New(total_size);
    //     if (!ptr) return UnifiedPointer(); // 分配失败

    //     // 检查偏移量是否有效
    //     if (shm_manage.GetMallocOffset(ptr) == -1) {
    //         LOG(ERROR) << "WriteMem failed: invalid offset";
    //         shm_manage.Free(ptr);
    //         return UnifiedPointer();
    //     }

    //     // 写入长度前缀和数据
    //     memcpy(ptr, &data_len, sizeof(data_len));
    //     memcpy(ptr + sizeof(data_len), value.data(), data_len);
    //     return UnifiedPointer::FromMemory(ptr);
    // }

    UnifiedPointer WriteDisk(const std::string_view& value) {
        uint16_t data_len = value.size();
        size_t total_size = data_len + sizeof(data_len);
        char* ptr = ssd_manage.New(total_size);
        if (!ptr) return UnifiedPointer(); // 分配失败

        off_t offset = static_cast<off_t>(ssd_manage.GetMallocOffset(ptr));
        if (offset == -1) {
            LOG(ERROR) << "WriteSSD failed: invalid offset";
            ssd_manage.Free(ptr);
            return UnifiedPointer();
        }
        std::vector<char> buffer(total_size);
        memcpy(buffer.data(), &data_len, sizeof(data_len));
        memcpy(buffer.data() + sizeof(data_len), value.data(), data_len);

        ssize_t bytes_written = pwrite(fd_ssd, buffer.data(), total_size, offset);
        if (bytes_written != static_cast<ssize_t>(total_size)) {
            LOG(ERROR) << "pwrite failed: " << strerror(errno)
                       << " expected: " << total_size << " actual: " << bytes_written;
            ssd_manage.Free(ptr);
            throw std::runtime_error("SSD write failed");
        }
        
        if (fsync(fd_ssd) < 0) {
            perror("fsync failed");
            ssd_manage.Free(ptr);
            return UnifiedPointer();
        }
        return UnifiedPointer::FromDiskPageId(static_cast<uint64_t>(offset));
    }
};