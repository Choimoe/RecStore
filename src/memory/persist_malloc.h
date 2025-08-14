#pragma once

#include <fcntl.h>
#include <sys/mman.h>

#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/async_time.h"
#include "base/bitmap.h"
#include "base/counter.h"
#include "base/hashtable.h"
#include "malloc.h"
#include "shm_file.h"

namespace base {

class PersistSimpleMalloc : public MallocApi {
 public:
  PersistSimpleMalloc(const std::string &filename, int64 memory_size,
                      int slab_size)
      : memory_size_(memory_size), slab_size_(slab_size) {
    bool file_exists = base::file_util::PathExists(filename);

    meta_data_memory_size_ = memory_size_ / slab_size_ * sizeof(uint64_t);

    if (!shm_file_.Initialize(filename, memory_size + meta_data_memory_size_)) {
      file_exists = false;
      CHECK(base::file_util::Delete(filename, false));
      CHECK(
          shm_file_.Initialize(filename, memory_size + meta_data_memory_size_))
          << filename << " " << memory_size;
    }
    Initialize();
    meta_data_block_ = (uint64_t *)shm_file_.Data();
    data_ = shm_file_.Data() + meta_data_memory_size_;
    if (!file_exists) {
      LOG(INFO) << "PersistSimpleMalloc: first initialization.";
    } else {
      LOG(INFO) << "PersistSimpleMalloc: Recovery from shutdown.";
    }

    LOG(WARNING) << "skip the first allocate";
    allocated_.fetch_add(slab_size_);
  }

  char *New(int malloc_size) override {
    CHECK_EQ(malloc_size, slab_size_)<< "Simple Malloc, size inconsistent";
    auto offset = allocated_.fetch_add(malloc_size);
    nr_malloc_++;
    // TODO allocate
    if (offset + slab_size_ >= memory_size_) {
      offset = 0;
      allocated_ = 0;
    }
    CHECK_LE(offset + slab_size_, memory_size_);
    meta_data_block_[offset / slab_size_] = malloc_size;
    return data_ + offset;
  }

  bool Free(void *memory_data) override {
    LOG(FATAL) << "not implement";
    return false;
  }

  void AddMallocs4Recovery(int64_t shm_offset) override {
    LOG(FATAL) << "not implement";
  }

  std::string GetInfo() const override {
    std::string info;
    info.append(folly::stringPrintf("allocated/memory_size/util: %ld/%ld/%ld\n",
                                    allocated_.load(), memory_size_,
                                    allocated_.load() * 100 / memory_size_));
    return info;
  }

  uint64_t total_malloc() const override { return nr_malloc_; }

  bool Healthy() const override { return true; }

  void GetMallocsAppend(std::vector<char *> *mallocs_data) const override {
    LOG(FATAL) << "not implement";
  }
  void GetMallocsAppend(std::vector<int64> *mallocs_offset) const override {
    LOG(FATAL) << "not implement";
  }
  void Initialize() override { allocated_.store(0); }

  char *GetMallocData(int64 offset) const override { return data_ + offset; }

  int GetMallocSize(int64 offset) const override {
    return meta_data_block_[offset / slab_size_];
  }

  int64 GetMallocOffset(const char *data) const override {
    int64 offset = data - data_;
    CHECK(data_ <= data);
    CHECK(data + slab_size_ < data_ + memory_size_);
    return offset;
  }
  int GetMallocSize(const char *data) const override {
    return GetMallocSize(GetMallocOffset(data));
  }

 private:
  ShmFile shm_file_;
  std::atomic<uint64_t> allocated_{0};
  std::atomic<uint64_t> nr_malloc_{0};
  char *data_;
  uint64_t *meta_data_block_;
  int64 memory_size_;
  int64 meta_data_memory_size_;
  int64 slab_size_;
  DISALLOW_COPY_AND_ASSIGN(PersistSimpleMalloc);
};

template <bool PERFECT_FIT_MOD = true>
class PersistMemoryPool : public MallocApi {
  static constexpr uint64_t kChunkSize = 2 * 1024 * 1024LL;
  static constexpr int kMetaDataSize = 8;

  class Chunk {
   public:
    Chunk(char *chunk_start, uint64 chunk_id) {
      header_ = (ChunkHeader *)chunk_start;
      data_ = nullptr;
      chunk_id_ = chunk_id;
      allocated_entries_ = 0;
    }

    void Initialize() { header_->is_used = false; }

    void Use(int slab_size) {
      // header(nr_entries) + nr_entries * slab_size < kChunkSize
      int nr_entries = kChunkSize / slab_size;
      nr_entries -= (nr_entries % 64);

      while (ChunkHeader::HeaderSize(nr_entries) + nr_entries * slab_size >
             kChunkSize) {
        nr_entries -= 64;
      }

      CHECK_GT(nr_entries, 0);

      header_->Initialize(slab_size, nr_entries);
      data_ = (char *)header_ + header_->HeaderSize();
      // TODO: flush the meta data
      allocated_entries_ = 0;
      is_recovered = true;
    }

    bool ValidPointerRange(void *start, void *end) {
      if ((char *)end - (char *)start != header_->slab_size_) {
        return false;
      }
      if (start < data_ ||
          end > data_ + header_->slab_size_ * header_->nr_entries_) {
        return false;
      }
      return true;
    }

    void Recovery() {
      if (UNLIKELY(!is_recovered.load())) {
        allocated_entries_ = header_->bitmap_->NumberOfOnes();
        data_ = (char *)header_ + header_->HeaderSize();
        is_recovered = true;
      }
    }

    bool IsChunkUsed() const { return header_->is_used; }

    bool Full() const {
#ifdef DEBUG
      CHECK_EQ(header_->bitmap_->FirstZeroPos() == -1,
               allocated_entries_ == MaxEntryNumber());
#endif
      return allocated_entries_ == MaxEntryNumber();
      // return header_->bitmap_->FirstZeroPos() == -1;
    }

    char *Malloc() {
      base::AutoLock lock(lock_);
      Recovery();
      int entry_id = header_->bitmap_->SetZeroPos();
      if (entry_id == -1) return nullptr;
      allocated_entries_++;
      return Entry(entry_id);
    }

    void Free(void *memory_data) {
      base::AutoLock lock(lock_);
      Recovery();
#ifdef DEBUG
      CHECK_LE(slab_size_ + (char *)memory_data, data_ + kChunkSize);
      CHECK_GE((char *)memory_data, data_);
      CHECK(header_->bitmap_->Get(EntryId(memory_data)) == false);
#endif
      allocated_entries_--;
      header_->bitmap_->Clear(EntryId(memory_data));
    }

    int SlabSize() const { return header_->slab_size_; }

    int AllocatedEntryNumber() const { return allocated_entries_; }

    std::vector<int> GetMallocedIds() const {
      std::vector<int> return_id;
      for (int i = 0; i < MaxEntryNumber(); i++)
        if (header_->bitmap_->Get(i)) {
          return_id.push_back(i);
        }
      CHECK_EQ(return_id.size(), allocated_entries_)
          << fmt::format("chunk id is {}", chunk_id_);
      return return_id;
    }

    std::vector<char *> GetMallocedData() const {
      std::vector<char *> return_data;
      for (int i = 0; i < MaxEntryNumber(); i++)
        if (header_->bitmap_->Get(i)) {
          return_data.push_back(Entry(i));
        }
      CHECK_EQ(return_data.size(), allocated_entries_)
          << fmt::format("chunk id is {}", chunk_id_);
      return return_data;
    }

   private:
    FOLLY_ALWAYS_INLINE int MaxEntryNumber() const {
      return header_->nr_entries_;
    }

    int EntryId(void *memory_data) {
#ifdef DEBUG
      CHECK_EQ((memory_data - data_) % SlabSize(), 0);
#endif
      return ((char *)memory_data - data_) / SlabSize();
    }

    char *Entry(int entry_id) const {
#ifdef DEBUG
      CHECK_GE(entry_id, 0);
      CHECK_LT(entry_id, MaxEntryNumber());
#endif
      return data_ + entry_id * SlabSize();
    }

    struct ChunkHeader {
      void Initialize(int slab_size, int nr_entries) {
        slab_size_ = slab_size;
        nr_entries_ = nr_entries;
        new (bitmap_) BitMap(nr_entries);
        bitmap_->Clear();
        is_used = true;
      }

      bool is_used = false;
      int slab_size_;
      int nr_entries_;
      bool allocated;
      base::BitMap bitmap_[0];  // bitmap_ must be always in the tail

      size_t HeaderSize() const {
        return sizeof(ChunkHeader) + base::BitMap::MemorySize(nr_entries_);
      }
      static size_t HeaderSize(int nr_entires) {
        return sizeof(ChunkHeader) + base::BitMap::MemorySize(nr_entires);
      }
      DISALLOW_COPY_AND_ASSIGN(ChunkHeader);
    };
    ChunkHeader *header_;
    char *data_;
    std::atomic_bool is_recovered = false;
    int allocated_entries_;
    uint64 chunk_id_;
    base::Lock lock_;
    DISALLOW_COPY_AND_ASSIGN(Chunk);
  };

 public:
  PersistMemoryPool(const std::string &filename, int64 memory_size,
                    const std::vector<int> &slab_sizes)
      : allocated_slab_sizes_(slab_sizes.begin(), slab_sizes.end()) {
    bool file_exists = base::file_util::PathExists(filename);
    if (!shm_file_.Initialize(filename, memory_size)) {
      file_exists = false;
      CHECK(base::file_util::Delete(filename, false));
      CHECK(shm_file_.Initialize(filename, memory_size))
          << filename << " " << memory_size;
    }

    memory_size -= memory_size % kChunkSize;
    nr_chunks_ = memory_size / kChunkSize;

    CHECK_GE(nr_chunks_, slab_sizes.size());

    for (int64 i = 0; i < nr_chunks_; i++) {
      chunks_.push_back(new Chunk(shm_file_.Data() + i * kChunkSize, i));
    }

    for (auto slab : slab_sizes) {
      size_to_chunks_[slab] = new std::deque<Chunk *>();
    }

    if (!file_exists || !Valid()) {
      LOG(INFO) << "PersistMemoryPool: first initialization.";
      for (int i = 0; i < nr_chunks_; i++) {
        chunks_[i]->Initialize();
        free_chunks_.push_back(chunks_[i]);
      }
      return;
    }
    LOG(INFO) << "PersistMemoryPool: Recovery from shutdown.";
    for (int i = 0; i < nr_chunks_; i++) {
      auto each = chunks_[i];
      if (each->IsChunkUsed()) {
        auto it = size_to_chunks_.find(each->SlabSize());
        size_to_chunks_[each->SlabSize()]->push_back(each);
        each->Recovery();
        LOG(INFO) << "each->AllocatedEntryNumber() = "
                  << each->AllocatedEntryNumber();
        total_malloc_ += each->AllocatedEntryNumber();
      } else {
        free_chunks_.push_back(each);
      }
    }
  }

  bool Valid() const { return true; }

  char *New(int memory_size) override {
    if (PERFECT_FIT_MOD) {
#ifdef DEBUG
      CHECK(allocated_slab_sizes_.find(memory_size) !=
            allocated_slab_sizes_.end());
#endif
      return NewInternal(memory_size);
    } else {
      auto iter =
          allocated_slab_sizes_.lower_bound(kMetaDataSize + memory_size);
      CHECK(iter != allocated_slab_sizes_.end());
      char *ptr = NewInternal(*iter);
      *(int *)ptr = memory_size;
      return ptr + kMetaDataSize;
    }
  }

  char *NewInternal(int slab_size) {
    // TODO: fine grained lock
    base::AutoLock lock(lock_);
    void *return_ptr = nullptr;

    Chunk *last_used_chunk = nullptr;
    auto iter = size_to_last_used_chunk_.find(slab_size);
    if (iter != size_to_last_used_chunk_.end()) last_used_chunk = iter->second;

    if (last_used_chunk && !last_used_chunk->Full()) {
      return_ptr = last_used_chunk->Malloc();
      CHECK(return_ptr);
    } else if (!free_chunks_.empty()) {
      auto front = free_chunks_.front();
      free_chunks_.pop_front();
      front->Use(slab_size);
      auto chunks = size_to_chunks_[slab_size];
      chunks->push_back(front);
      size_to_last_used_chunk_[slab_size] = front;
      return_ptr = front->Malloc();
    } else {
      auto chunks = size_to_chunks_[slab_size];
      for (auto chunk : *chunks) {
        if (chunk->Full()) continue;
        return_ptr = chunk->Malloc();
        goto final;
      }
      LOG(FATAL) << fmt::format("Persist Memory Pool OOM, total_malloc={}",
                                total_malloc_.load());
    }
  final:
    CHECK(return_ptr);
    total_malloc_++;
    return (char *)return_ptr;
  }

  bool Free(void *memory_data) override {
    if (!PERFECT_FIT_MOD) memory_data = (char *)memory_data - kMetaDataSize;

    int chunk_id = GetChunkID(memory_data);
    if (0 <= chunk_id && chunk_id < chunks_.size()) {
      chunks_[chunk_id]->Free(memory_data);
      total_malloc_--;
      return true;
    } else {
      return false;
    }
  }
  void GetMallocsAppend(std::vector<char *> *mallocs_data) const override {
    for (auto chunk : chunks_) {
      auto chunk_data = chunk->GetMallocedData();
      for (char *each : chunk_data) {
        if (PERFECT_FIT_MOD)
          mallocs_data->push_back(each);
        else
          mallocs_data->push_back((char *)each + kMetaDataSize);
      }
    }
  }
  void GetMallocsAppend(std::vector<int64> *mallocs_offset) const override {
    std::vector<char *> mallocs_data;
    GetMallocsAppend(&mallocs_data);
    for (auto each : mallocs_data) {
      mallocs_offset->push_back(each - shm_file_.Data());
    }
  }
  void Initialize() override {}

  std::string GetInfo() const override {
    uint64 malloced_bytes = 0;
    uint64 malloced_chunk = 0;
    for (auto each : chunks_) {
      if (each->IsChunkUsed()) {
        malloced_bytes += each->AllocatedEntryNumber() * each->SlabSize();
        malloced_chunk++;
      }
    }
    return folly::sformat(
        "[PersistMemoryPool] "
        "Use {} of {} chunks,  Util {} %",
        malloced_chunk, chunks_.size(),
        100 * malloced_bytes / (kChunkSize * chunks_.size()));
  }

  char *GetMallocData(int64 offset) const override {
    return shm_file_.Data() + offset;
  }

  int GetMallocSize(int64 offset) const override {
    return GetMallocSize(GetMallocData(offset));
  }

  int64 GetMallocOffset(const char *data) const override {
    auto ret = data - shm_file_.Data();
    // see cache.SetShmMallocOffset
    CHECK_EQ((ret >> 3) << 3, ret);
    return ret;
  }

  int GetMallocSize(const char *data) const override {
    if (PERFECT_FIT_MOD)
      return chunks_[GetChunkID(data)]->SlabSize();
    else
      return *(int *)(data - kMetaDataSize);
  }

  uint64_t total_malloc() const override { return total_malloc_; }

  ~PersistMemoryPool() override {
    for (auto p : chunks_) {
      delete p;
    }
    for (auto [size, q] : size_to_chunks_) {
      delete q;
    }
  }

  bool Healthy() const { return true; }
  void AddMallocs4Recovery(int64_t shm_offset) {
    FB_LOG_EVERY_MS(ERROR, 1000) << "AddMallocs4Recovery not implement";
  }

 private:
  int64 GetChunkID(void *data) const { return GetChunkID((const char *)data); }
  int64 GetChunkID(const char *data) const {
    return GetChunkID(data - shm_file_.Data());
  }

  int64 GetChunkID(int64 offset) const { return offset / kChunkSize; }

 private:
  ShmFile shm_file_;
  int64 nr_chunks_;
  std::vector<Chunk *> chunks_;
  std::deque<Chunk *> free_chunks_;

  std::unordered_map<int, std::deque<Chunk *> *> size_to_chunks_;
  std::unordered_map<int, Chunk *> size_to_last_used_chunk_;

  std::set<int> allocated_slab_sizes_;

  std::atomic<int64> total_malloc_ = 0;

  base::Lock lock_;
  // different slab
};

class PersistLoopShmMalloc : public MallocApi {
 public:
  static const int max_fast_list_type = 32;
  static const int max_fast_list_num = 1 << 20;
  // filename: 文件名, 内存大小
  PersistLoopShmMalloc(const std::string &filename, int64 memory_size);
  // 如果分配内存不是固定的几种大小，且我们需要利用循环首次适应的 LRU 特性,
  // 关闭这个功能
  void DisableFastMalloc() { enable_fast_malloc_ = false; }
  char *New(int memory_size);
  bool Free(void *memory_data);

  bool load_success() const { return load_success_; }
  void GetMallocsAppend(std::vector<char *> *mallocs_data) const;
  void GetMallocsAppend(std::vector<int64> *mallocs_offset) const;
  std::string GetInfo() const {
    std::string info;
    info.append(folly::stringPrintf("used/healthy/total block: %ld/%ld/%ld\n",
                                    total_used_, healthy_used_, block_num_));
    info.append(total_fast_malloc_.Display() + "\n");
    info.append(total_loop_malloc_.Display() + "\n");
    return info;
  }
  void Initialize() {
    total_used_ = 0;
    total_malloc_ = 0;
    last_malloc_block_ = 0;
    memset(used_bits_, 0, block_num_ >> 3);
    clflushopt_range(used_bits_, block_num_ >> 3);
  }
  // 一共使用了多少 8 字节的 block
  int64 total_used() const { return total_used_; }
  // 一共分配了多少块内存, 和 GetUsedBlockAppend 对应
  uint64 total_malloc() const { return total_malloc_; }
  bool Healthy() const { return total_used_ <= healthy_used_; }
  int64 DataBaseOffset()const {return data_ - shm_file_.Data();}
  char *GetMallocData(int64 offset) const {
    if (offset < 8 || (offset & 7) != 0 || offset > block_num_ * 8L)
      return NULL;
    return data_ + offset;
  }
  int GetMallocSize(int64 offset) const {
    if (offset == block_num_ * 8L) return 0;
    if (offset < 8 || (offset & 7) != 0 || offset > block_num_ * 8L) return -1;
    return Block()[BlockIndex(offset) - 1];
  }
  int64 GetMallocOffset(const char *data) const {
    int64 offset = data - data_;
    if (offset < 8 || (offset & 7) != 0 || offset > block_num_ * 8L) return -1;
    return offset;
  }
  int GetMallocSize(const char *data) const {
    int64 offset = GetMallocOffset(data);
    if (offset == block_num_ * 8L) return 0;
    if (offset < 0) return -1;
    return Block()[BlockIndex(offset) - 1];
  }

  void AddMallocs4Recovery(int64_t shm_offset);

 private:
  static int BlockNum(int memory_size) { return (memory_size + 7) >> 3; }
  static int64 BlockIndex(int64 offset) { return offset >> 3; }
  bool UnusedMemoryValid(int64 block_index) const;
  bool MemoryValid() const;
  bool Used(int64 index) const {
    return (used_bits_[index >> 6] & (1ul << (index & 63))) != 0;
  }
  const uint64 *Block() const {
    return reinterpret_cast<const uint64 *>(data_);
  }
  uint64 *Block() { return reinterpret_cast<uint64 *>(data_); }
  void UseBlock(int64 index) { used_bits_[index >> 6] |= 1ul << (index & 63); }
  void FreeBlock(int64 index) {
    used_bits_[index >> 6] &= ~(1ul << (index & 63));
  }
  ShmFile shm_file_;
  base::Lock lock_;
  uint64 *used_bits_;
  char *data_;
  char *start_address_;
  int64 memory_size_;
  int64 block_num_;
  int64 last_malloc_block_;
  int64 total_used_;
  int64 total_malloc_;
  int64 healthy_used_;
  bool load_success_;
  bool enable_fast_malloc_ = true;
  StdAutoDeleteHash<std::deque<int64>> fast_malloc_lists_;
  Counter total_fast_malloc_{"total_fast_malloc"};
  Counter total_loop_malloc_{"total_loop_malloc"};

  DISALLOW_COPY_AND_ASSIGN(PersistLoopShmMalloc);
};

}  // namespace base
