#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "storage/index/index.h"

class ValueStore {
public:
  struct ReadResult {
    std::vector<char> data;
  };

  struct WriteSpec {
    const void* data;
    size_t size;
  };

  virtual ~ValueStore() = default;

  virtual uint64_t Alloc(size_t size)                                  = 0;
  virtual void Write(uint64_t handle, const void* data, size_t size)   = 0;
  virtual uint64_t AllocAndWrite(const void* data, size_t size)        = 0;
  virtual size_t Read(uint64_t handle, void* out_buf, size_t buf_size) = 0;
  virtual void Free(uint64_t handle)                                   = 0;
  virtual const char* DirectPtr(uint64_t handle) const { return nullptr; }
  virtual size_t SlotCapacity(uint64_t handle) const = 0;

  virtual void BatchRead(const std::vector<uint64_t>& handles,
                         std::vector<ReadResult>& out_results) {
    out_results.resize(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
        continue;
      }
      const size_t cap = SlotCapacity(handles[i]);
      out_results[i].data.resize(cap);
      const size_t actual = Read(
          handles[i], out_results[i].data.data(), out_results[i].data.size());
      out_results[i].data.resize(actual);
    }
  }

  virtual std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteSpec>& specs) {
    std::vector<uint64_t> handles;
    handles.reserve(specs.size());
    for (const auto& spec : specs) {
      handles.push_back(AllocAndWrite(spec.data, spec.size));
    }
    return handles;
  }

  virtual std::string GetInfo() const { return "(no info)"; }
};
