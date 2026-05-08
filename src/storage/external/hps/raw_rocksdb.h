#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace recstore::storage {

class RawRocksDBBackend {
public:
  using MissCallback = std::function<void(size_t)>;

  RawRocksDBBackend(
      const std::string& path, size_t value_size, bool use_mem_env = false);
  ~RawRocksDBBackend();

  RawRocksDBBackend(const RawRocksDBBackend&)            = delete;
  RawRocksDBBackend& operator=(const RawRocksDBBackend&) = delete;

  void Insert(size_t num_keys, const long long* keys, const char* values);

  void Fetch(size_t num_keys,
             const long long* keys,
             char* values,
             const MissCallback& on_miss);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace recstore::storage
