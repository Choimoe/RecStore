#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace recstore::storage::fasterkv {

class FasterKVBackend {
public:
  using MissCallback = std::function<void(size_t)>;

  FasterKVBackend(uint64_t capacity, size_t value_size);
  ~FasterKVBackend();

  FasterKVBackend(const FasterKVBackend&)            = delete;
  FasterKVBackend& operator=(const FasterKVBackend&) = delete;

  void Insert(size_t num_keys, const long long* keys, const char* values);

  void Fetch(size_t num_keys,
             const long long* keys,
             char* values,
             const MissCallback& on_miss);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace recstore::storage::fasterkv
