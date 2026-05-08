#include "storage/external/hps/raw_rocksdb.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"
#include "rocksdb/write_batch.h"

namespace recstore::storage {
namespace {

std::string EncodeKey(long long key) {
  static_assert(sizeof(long long) == sizeof(uint64_t));
  uint64_t encoded = static_cast<uint64_t>(key);
  std::string out(sizeof(encoded), '\0');
  std::memcpy(out.data(), &encoded, sizeof(encoded));
  return out;
}

void CheckStatus(const rocksdb::Status& status, const char* operation) {
  if (!status.ok()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.ToString());
  }
}

} // namespace

class RawRocksDBBackend::Impl {
public:
  Impl(const std::string& path, size_t value_size, bool use_mem_env)
      : value_size_(value_size) {
    if (value_size_ == 0) {
      throw std::invalid_argument("RawRocksDBBackend value_size must be > 0");
    }

    rocksdb::Options options;
    options.create_if_missing = true;
    if (use_mem_env) {
      mem_env_.reset(rocksdb::NewMemEnv(rocksdb::Env::Default()));
      if (!mem_env_) {
        throw std::runtime_error("RocksDB NewMemEnv returned null");
      }
      options.env = mem_env_.get();
    }
    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();

    rocksdb::DB* db = nullptr;
    CheckStatus(rocksdb::DB::Open(options, path, &db), "RocksDB::Open");
    db_.reset(db);
  }

  void Insert(size_t num_keys, const long long* keys, const char* values) {
    rocksdb::WriteBatch batch;
    std::vector<std::string> key_storage;
    key_storage.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      key_storage.push_back(EncodeKey(keys[i]));
      batch.Put(rocksdb::Slice(key_storage.back()),
                rocksdb::Slice(values + i * value_size_, value_size_));
    }
    CheckStatus(db_->Write(write_options_, &batch), "RocksDB::WriteBatch");
  }

  void Fetch(size_t num_keys,
             const long long* keys,
             char* values,
             const MissCallback& on_miss) {
    std::vector<std::string> key_storage;
    key_storage.reserve(num_keys);
    std::vector<rocksdb::Slice> key_slices;
    key_slices.reserve(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      key_storage.push_back(EncodeKey(keys[i]));
      key_slices.emplace_back(key_storage.back());
    }

    std::vector<std::string> result_storage(num_keys);
    const std::vector<rocksdb::Status> statuses =
        db_->MultiGet(read_options_, key_slices, &result_storage);
    for (size_t i = 0; i < num_keys; ++i) {
      if (statuses[i].IsNotFound()) {
        if (on_miss) {
          on_miss(i);
        }
        continue;
      }
      CheckStatus(statuses[i], "RocksDB::MultiGet");
      if (result_storage[i].size() != value_size_) {
        throw std::runtime_error("RocksDB value size mismatch");
      }
      std::memcpy(
          values + i * value_size_, result_storage[i].data(), value_size_);
    }
  }

private:
  struct DBDeleter {
    void operator()(rocksdb::DB* db) const { delete db; }
  };

  size_t value_size_;
  rocksdb::ReadOptions read_options_;
  rocksdb::WriteOptions write_options_;
  std::unique_ptr<rocksdb::Env> mem_env_;
  std::unique_ptr<rocksdb::DB, DBDeleter> db_;
};

RawRocksDBBackend::RawRocksDBBackend(
    const std::string& path, size_t value_size, bool use_mem_env)
    : impl_(std::make_unique<Impl>(path, value_size, use_mem_env)) {}

RawRocksDBBackend::~RawRocksDBBackend() = default;

void RawRocksDBBackend::Insert(
    size_t num_keys, const long long* keys, const char* values) {
  impl_->Insert(num_keys, keys, values);
}

void RawRocksDBBackend::Fetch(size_t num_keys,
                              const long long* keys,
                              char* values,
                              const MissCallback& on_miss) {
  impl_->Fetch(num_keys, keys, values, on_miss);
}

} // namespace recstore::storage
