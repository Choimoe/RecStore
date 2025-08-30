#include "persist_malloc.h"

#include <algorithm>
#include <iostream>

DEFINE_double(shm_malloc_healthy_rate, 0.95, "shm malloc healthy rate");

DECLARE_double(shm_malloc_healthy_rate);

namespace base {
PersistLoopShmMalloc::PersistLoopShmMalloc(const std::string &filename,
                                           int64 memory_size) {
  block_num_ = 0;
  while (((block_num_ + 64) >> 3) + (block_num_ + 64) * sizeof(uint64) <=
         memory_size) {
    block_num_ += 64;
  }
  healthy_used_ = block_num_ * FLAGS_shm_malloc_healthy_rate;
  bool file_exists = base::file_util::PathExists(filename);
  if (!shm_file_.Initialize(filename, memory_size)) {
    file_exists = false;
    CHECK(base::file_util::Delete(filename, false));
    CHECK(shm_file_.Initialize(filename, memory_size))
        << filename << " " << memory_size;
  }

  used_bits_ = reinterpret_cast<uint64 *>(shm_file_.Data());
  data_ = shm_file_.Data() + (block_num_ >> 3);
  R2AllocMaster.init(data_, memory_size);
  load_success_ = file_exists;
  Initialize();
  if (!file_exists) {
    LOG(INFO) << "PersistLoopShmMalloc: first initialization.";
  } else {
    LOG(INFO) << "PersistLoopShmMalloc: Recovery from shutdown.";
  }
}

void PersistLoopShmMalloc::AddMallocs4Recovery(int64_t shm_offset) {
  // NOTE(xieminhui) use atomic instr to reduce conflict between threads.
  // But the code below has redundancy with the function UseBlock
  __sync_fetch_and_add(&total_malloc_, 1);  // NOLINT
  int64_t offset = shm_offset;
  CHECK_GE(offset, 0);
  CHECK_LT(offset, block_num_ * 8L);
  int memory_size = GetMallocSize(offset);
  int memory_block_num = BlockNum(memory_size);

  __sync_fetch_and_add(&total_used_, memory_block_num + 1);  // NOLINT
  int64_t sizeBlockIdx = BlockIndex(offset) - 1;

  for (int i = 1; i <= memory_block_num; i++) {
    int64 index = sizeBlockIdx + i;
    // Use Block [index]
    uint64 temp;
    do {
      temp = used_bits_[index >> 6];
    } while (!__sync_bool_compare_and_swap(&used_bits_[index >> 6],  // NOLINT
                                           temp, temp | (1ul << (index & 63))));
  }
}

bool PersistLoopShmMalloc::UnusedMemoryValid(int64 block_index) const {
  int64 memory_block_num = BlockNum(Block()[block_index]);
  if (memory_block_num < 0) {
    LOG(ERROR) << "memory_block_num < 0";
    return false;
  }
  block_index++;
  if (block_index + memory_block_num > block_num_) {
    LOG(ERROR) << "block_index + memory_block_num > block_num_";
    return false;
  }
  for (int i = 0; i < memory_block_num; ++i) {
    if (!Used(block_index + i)) {
      // LOG(ERROR) << "!Used(block_index + i)\t"
      //            << "block_index=" << block_index << "; i=" << i;
      return false;
    }
  }
  return true;
}

void PersistLoopShmMalloc::GetMallocsAppend(
    std::vector<char *> *mallocs_data) const {
  for (int64 i = 0; i < block_num_ - 1; ++i) {
    if (!Used(i) && Used(i + 1)) {
      CHECK_GT(Block()[i], 0) << "block id = " << i;
      mallocs_data->push_back(data_ + ((i + 1) << 3));
    }
  }
}

void PersistLoopShmMalloc::GetMallocsAppend(
    std::vector<int64> *mallocs_offset) const {
  for (int64 i = 0; i < block_num_ - 1; ++i) {
    if (!Used(i) && Used(i + 1)) {
      CHECK_GT(Block()[i], 0) << "block id = " << i;
      mallocs_offset->push_back((i + 1) << 3);
    }
  }
}

bool PersistLoopShmMalloc::MemoryValid() const {
  int64 block_index = 0;
  while (block_index < block_num_) {
    if (Used(block_index)) return false;
    if (!UnusedMemoryValid(block_index)) return false;
    block_index += 1 + BlockNum(Block()[block_index]);
  }
  return block_index == block_num_;
}


char* PersistLoopShmMalloc::New(int memory_size) {
  if (memory_size <= 0) return nullptr;
  r2::Allocator* alloc = R2AllocMaster.get_thread_allocator();
  if (!alloc){
    LOG(ERROR) << "PersistLoopShmMalloc::New get_thread_allocator() failed";
    return nullptr;
  }
  void* p = alloc->alloc(static_cast<size_t>(memory_size)); 
  return static_cast<char*>(p);                          
}

bool PersistLoopShmMalloc::Free(void* user_ptr) {
  LOG(INFO) << "PersistLoopShmMalloc::Free ptr=" << user_ptr;
  if (!user_ptr) return true;
  r2::Allocator* alloc = R2AllocMaster.get_thread_allocator();
  LOG(INFO) << "PersistLoopShmMalloc::Free ptr=" << user_ptr;
  alloc->dealloc(user_ptr);
  return true;
}

}  // namespace base