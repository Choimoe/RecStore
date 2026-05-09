#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "storage/allocator/ssd/ssd_block_allocator.h"
#include "storage/io_backend/io_backend.h"

class SsdSlabAllocator : public SsdBlockAllocator {
public:
  SsdSlabAllocator(IOBackend* backend,
                   const std::vector<int>& size_classes,
                   uint64_t total_capacity_bytes,
                   uint64_t base_byte_offset)
      : backend_(backend) {
    if (backend_ == nullptr) {
      throw std::invalid_argument("SsdSlabAllocator backend is null");
    }
    size_classes_ =
        size_classes.empty() ? std::vector<int>{128, 256, 512, 1024, 4096}
                             : size_classes;
    std::sort(size_classes_.begin(), size_classes_.end());
    const uint64_t per_slab = std::max<uint64_t>(
        PAGE_SIZE, total_capacity_bytes / size_classes_.size());
    uint64_t cursor = base_byte_offset;
    for (int cls : size_classes_) {
      const uint64_t slot_size = static_cast<uint64_t>(cls) + kHeaderSize;
      SlabPool pool;
      pool.slot_size        = slot_size;
      pool.base_byte_offset = cursor;
      pool.capacity_slots   = std::max<uint64_t>(1, per_slab / slot_size);
      slabs_.push_back(std::move(pool));
      cursor += pool.capacity_slots * slot_size;
    }
  }

  uint64_t Alloc(size_t data_size) override {
    const uint16_t slab_idx = SelectSlab(data_size);
    auto& slab              = slabs_.at(slab_idx);
    uint64_t slot_id        = 0;
    {
      std::lock_guard<std::mutex> lock(slab.free_mu);
      if (!slab.free_list.empty()) {
        slot_id = slab.free_list.back();
        slab.free_list.pop_back();
        return Encode(slab_idx, slot_id);
      }
    }
    slot_id = slab.next_slot.fetch_add(1, std::memory_order_relaxed);
    if (slot_id >= slab.capacity_slots) {
      return kInvalidHandle;
    }
    return Encode(slab_idx, slot_id);
  }

  void Free(uint64_t handle) override {
    uint16_t slab_idx;
    uint64_t slot_id;
    Decode(handle, slab_idx, slot_id);
    auto& slab = slabs_.at(slab_idx);
    std::lock_guard<std::mutex> lock(slab.free_mu);
    slab.free_list.push_back(slot_id);
  }

  void Write(uint64_t handle, const void* data, size_t data_size) override {
    uint16_t slab_idx;
    uint64_t slot_id;
    Decode(handle, slab_idx, slot_id);
    const auto& slab = slabs_.at(slab_idx);
    if (data_size + kHeaderSize > slab.slot_size) {
      throw std::invalid_argument("SsdSlabAllocator write exceeds slot size");
    }
    WriteBytes(slab.SlotByteOffset(slot_id), slab.slot_size, data, data_size);
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    uint16_t slab_idx;
    uint64_t slot_id;
    Decode(handle, slab_idx, slot_id);
    const auto& slab = slabs_.at(slab_idx);
    return ReadBytes(
        slab.SlotByteOffset(slot_id), slab.slot_size, out_buf, buf_size);
  }

  uint64_t AllocAndWrite(const void* data, size_t data_size) override {
    const uint64_t handle = Alloc(data_size);
    if (handle != kInvalidHandle) {
      Write(handle, data, data_size);
    }
    return handle;
  }

  size_t SlotCapacity(uint64_t handle) const override {
    uint16_t slab_idx;
    uint64_t slot_id;
    Decode(handle, slab_idx, slot_id);
    (void)slot_id;
    return slabs_.at(slab_idx).slot_size - kHeaderSize;
  }

private:
  struct SlabPool {
    uint64_t slot_size        = 0;
    uint64_t base_byte_offset = 0;
    uint64_t capacity_slots   = 0;
    std::atomic<uint64_t> next_slot{0};
    std::mutex free_mu;
    std::vector<uint64_t> free_list;

    SlabPool() = default;
    SlabPool(SlabPool&& other) noexcept
        : slot_size(other.slot_size),
          base_byte_offset(other.base_byte_offset),
          capacity_slots(other.capacity_slots),
          next_slot(other.next_slot.load(std::memory_order_relaxed)),
          free_list(std::move(other.free_list)) {}
    SlabPool& operator=(SlabPool&&) = delete;

    uint64_t SlotByteOffset(uint64_t slot_id) const {
      return base_byte_offset + slot_id * slot_size;
    }
  };

  static uint64_t Encode(uint16_t slab_idx, uint64_t slot_id) {
    return (static_cast<uint64_t>(slab_idx) << 48) |
           ((slot_id + 1) & 0x0000FFFFFFFFFFFFULL);
  }

  static void Decode(uint64_t handle, uint16_t& slab_idx, uint64_t& slot_id) {
    slab_idx = static_cast<uint16_t>(handle >> 48);
    slot_id  = (handle & 0x0000FFFFFFFFFFFFULL) - 1;
  }

  uint16_t SelectSlab(size_t data_size) const {
    for (uint16_t i = 0; i < size_classes_.size(); ++i) {
      if (data_size <= static_cast<size_t>(size_classes_[i])) {
        return i;
      }
    }
    throw std::invalid_argument("SsdSlabAllocator value too large");
  }

  void WriteBytes(uint64_t byte_offset,
                  uint64_t slot_size,
                  const void* data,
                  size_t data_size) {
    std::lock_guard<std::mutex> lock(io_mu_);
    const PageID_t start    = byte_offset / PAGE_SIZE;
    const uint64_t page_off = byte_offset % PAGE_SIZE;
    const uint64_t total    = page_off + slot_size;
    const uint64_t pages    = (total + PAGE_SIZE - 1) / PAGE_SIZE;
    char* buf               = backend_->AllocateBuffer(pages);
    backend_->BatchReadPages({{start, buf, pages}});
    const uint32_t n = static_cast<uint32_t>(data_size);
    std::memcpy(buf + page_off, &n, sizeof(n));
    std::memcpy(buf + page_off + kHeaderSize, data, data_size);
    backend_->BatchWritePages({{start, buf, pages}});
    backend_->FreeBuffer(buf);
  }

  size_t ReadBytes(uint64_t byte_offset,
                   uint64_t slot_size,
                   void* out_buf,
                   size_t buf_size) {
    std::lock_guard<std::mutex> lock(io_mu_);
    const PageID_t start    = byte_offset / PAGE_SIZE;
    const uint64_t page_off = byte_offset % PAGE_SIZE;
    const uint64_t total    = page_off + slot_size;
    const uint64_t pages    = (total + PAGE_SIZE - 1) / PAGE_SIZE;
    char* buf               = backend_->AllocateBuffer(pages);
    backend_->BatchReadPages({{start, buf, pages}});
    uint32_t n = 0;
    std::memcpy(&n, buf + page_off, sizeof(n));
    const size_t actual = std::min(buf_size, static_cast<size_t>(n));
    std::memcpy(out_buf, buf + page_off + kHeaderSize, actual);
    backend_->FreeBuffer(buf);
    return actual;
  }

  static constexpr uint64_t kHeaderSize = 4;
  IOBackend* backend_;
  std::mutex io_mu_;
  std::vector<int> size_classes_;
  std::vector<SlabPool> slabs_;
};

FACTORY_REGISTER(SsdBlockAllocator,
                 SSD_SLAB,
                 SsdSlabAllocator,
                 IOBackend*,
                 const std::vector<int>&,
                 uint64_t,
                 uint64_t);
