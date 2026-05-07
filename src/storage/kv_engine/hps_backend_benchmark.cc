#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/bind_core.h"
#include "base/init.h"
#include "base/log.h"
#include "storage/kv_engine/hps_recstore_backend.h"
#include "third_party/HugeCTR/HugeCTR/include/hps/hash_map_backend.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/rocksdb_backend.hpp"

DEFINE_string(backend, "hps_hash_map", "hps_hash_map|hps_rocksdb|recstore");
DEFINE_string(path, "", "RecStore data path");
DEFINE_string(index_type, "DRAM_EXTENDIBLE_HASH", "RecStore index.type");
DEFINE_string(value_store_type, "DRAM_VALUE_STORE", "RecStore value.type");
DEFINE_string(dram_allocator, "PERSIST_LOOP_SLAB", "RecStore DRAM allocator");
DEFINE_int64(dram_capacity_bytes, 0, "override RecStore DRAM capacity bytes");
DEFINE_int64(record_count, 1000000, "record count");
DEFINE_int32(value_size, 512, "value size bytes");
DEFINE_int32(batch_size, 1024, "keys per HPS fetch/insert call");
DEFINE_int32(thread_num, 16, "worker thread count");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");
DEFINE_string(distribution, "uniform", "uniform|zipfian");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_string(mode, "fetch", "fetch|insert|mixed");
DEFINE_int32(read_ratio, 100, "read percentage for mixed mode");
DEFINE_string(table_name, "hps_recstore_bench_table", "HPS table name");

namespace {

struct PhaseStats {
  uint64_t batches = 0;
  uint64_t key_ops = 0;
  uint64_t misses  = 0;
};

class FastRandom {
public:
  explicit FastRandom(uint64_t seed)
      : state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t Next() {
    uint64_t x = state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state_ = x;
    return x * 2685821657736338717ULL;
  }

  double Uniform01() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }
  uint64_t Uniform(uint64_t n) { return n == 0 ? 0 : Next() % n; }

private:
  uint64_t state_;
};

class KeyGenerator {
public:
  KeyGenerator(std::string distribution,
               uint64_t record_count,
               double alpha,
               uint64_t seed)
      : distribution_(std::move(distribution)),
        record_count_(record_count),
        alpha_(alpha),
        rng_(seed) {
    if (record_count_ == 0) {
      throw std::invalid_argument("record_count must be positive");
    }
    if (distribution_ == "zipfian") {
      if (std::abs(alpha_ - 1.0) < 1e-9) {
        log_n_ = std::log(static_cast<double>(record_count_));
      } else {
        pow_n_ = std::pow(static_cast<double>(record_count_), 1.0 - alpha_);
      }
    } else if (distribution_ != "uniform") {
      throw std::invalid_argument("distribution must be uniform or zipfian");
    }
  }

  uint64_t NextKey() {
    if (distribution_ == "uniform") {
      return rng_.Uniform(record_count_) + 1;
    }
    return NextZipfian() + 1;
  }

  uint64_t NextUint(uint64_t n) { return rng_.Uniform(n); }

private:
  uint64_t NextZipfian() {
    const double u = std::max(rng_.Uniform01(), 1e-12);
    double rank    = 1.0;
    if (std::abs(alpha_ - 1.0) < 1e-9) {
      rank = std::exp(u * log_n_);
    } else {
      rank = std::pow(1.0 + u * (pow_n_ - 1.0), 1.0 / (1.0 - alpha_));
    }
    uint64_t key = static_cast<uint64_t>(rank);
    if (key >= record_count_) {
      key = record_count_ - 1;
    }
    return key;
  }

  std::string distribution_;
  uint64_t record_count_;
  double alpha_;
  double pow_n_ = 1.0;
  double log_n_ = 0.0;
  FastRandom rng_;
};

using HpsBackend = HugeCTR::DatabaseBackendBase<long long>;

std::unique_ptr<HpsBackend> CreateBackend() {
  if (FLAGS_backend == "hps_hash_map") {
    HugeCTR::HashMapBackendParams params;
    params.max_batch_size  = static_cast<size_t>(FLAGS_batch_size);
    params.num_partitions  = static_cast<size_t>(std::max(1, FLAGS_thread_num));
    params.overflow_margin = std::numeric_limits<size_t>::max();
    return std::make_unique<HugeCTR::HashMapBackend<long long>>(params);
  }
  if (FLAGS_backend == "hps_rocksdb") {
    HugeCTR::RocksDBBackendParams params;
    params.path           = FLAGS_path;
    params.max_batch_size = static_cast<size_t>(FLAGS_batch_size);
    params.num_threads    = static_cast<size_t>(std::max(1, FLAGS_thread_num));
    return std::make_unique<HugeCTR::RocksDBBackend<long long>>(params);
  }
  if (FLAGS_backend == "recstore") {
    recstore::storage::HpsRecStoreBackendParams params;
    params.path             = FLAGS_path;
    params.capacity         = static_cast<uint64_t>(FLAGS_record_count);
    params.value_size       = static_cast<uint32_t>(FLAGS_value_size);
    params.max_batch_size   = static_cast<size_t>(FLAGS_batch_size);
    params.num_partitions   = static_cast<size_t>(std::max(1, FLAGS_thread_num));
    params.index_type       = FLAGS_index_type;
    params.value_store_type = FLAGS_value_store_type;
    params.dram_allocator   = FLAGS_dram_allocator;
    params.dram_capacity_bytes =
        FLAGS_dram_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_dram_capacity_bytes)
            : 0;
    params.num_threads = FLAGS_thread_num;
    return std::make_unique<recstore::storage::HpsRecStoreBackend<long long>>(
        params);
  }
  throw std::invalid_argument("unsupported backend: " + FLAGS_backend);
}

std::vector<char> MakeValues(size_t rows, size_t value_size, int seed) {
  std::vector<char> values(rows * value_size);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<char>('a' + ((i + seed) % 26));
  }
  return values;
}

PhaseStats LoadRecords(HpsBackend* backend, int load_threads) {
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(load_threads);
  const uint64_t record_count = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t per_thread =
      (record_count + static_cast<uint64_t>(load_threads) - 1) /
      static_cast<uint64_t>(load_threads);

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      std::vector<long long> keys;
      keys.reserve(static_cast<size_t>(FLAGS_batch_size));
      std::vector<char> values =
          MakeValues(static_cast<size_t>(FLAGS_batch_size), FLAGS_value_size, tid);
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread + 1;
      const uint64_t end = std::min(record_count + 1, begin + per_thread);
      PhaseStats local;
      for (uint64_t key = begin; key < end;) {
        keys.clear();
        while (key < end && keys.size() < static_cast<size_t>(FLAGS_batch_size)) {
          keys.push_back(static_cast<long long>(key++));
        }
        backend->insert(FLAGS_table_name,
                        keys.size(),
                        keys.data(),
                        values.data(),
                        static_cast<uint32_t>(FLAGS_value_size),
                        static_cast<size_t>(FLAGS_value_size));
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  return total;
}

void PrepareBackendForLoad(HpsBackend* backend) {
  if (FLAGS_backend != "hps_rocksdb") {
    return;
  }

  const long long key = 1;
  std::vector<char> value =
      MakeValues(1, static_cast<size_t>(FLAGS_value_size), 0);
  backend->insert(FLAGS_table_name,
                  1,
                  &key,
                  value.data(),
                  static_cast<uint32_t>(FLAGS_value_size),
                  static_cast<size_t>(FLAGS_value_size));
}

PhaseStats RunTransactions(HpsBackend* backend) {
  const bool fetch_only  = FLAGS_mode == "fetch";
  const bool insert_only = FLAGS_mode == "insert";
  const bool mixed       = FLAGS_mode == "mixed";
  if (!fetch_only && !insert_only && !mixed) {
    throw std::invalid_argument("mode must be fetch|insert|mixed");
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      KeyGenerator key_gen(FLAGS_distribution,
                           static_cast<uint64_t>(FLAGS_record_count),
                           FLAGS_zipfian_alpha,
                           0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(tid));
      std::vector<long long> keys(static_cast<size_t>(FLAGS_batch_size));
      std::vector<char> values =
          MakeValues(static_cast<size_t>(FLAGS_batch_size), FLAGS_value_size, tid);
      std::vector<char> out(static_cast<size_t>(FLAGS_batch_size) *
                            static_cast<size_t>(FLAGS_value_size));
      PhaseStats local;
      while (!start.load(std::memory_order_acquire)) {
      }
      while (!stop.load(std::memory_order_relaxed)) {
        for (auto& key : keys) {
          key = static_cast<long long>(key_gen.NextKey());
        }
        const bool do_fetch =
            fetch_only ||
            (mixed && static_cast<int>(key_gen.NextUint(100)) < FLAGS_read_ratio);
        if (do_fetch) {
          size_t misses = 0;
          backend->fetch(FLAGS_table_name,
                         keys.size(),
                         keys.data(),
                         out.data(),
                         static_cast<size_t>(FLAGS_value_size),
                         [&](size_t) { ++misses; },
                         std::chrono::nanoseconds::zero());
          local.misses += misses;
        } else {
          backend->insert(FLAGS_table_name,
                          keys.size(),
                          keys.data(),
                          values.data(),
                          static_cast<uint32_t>(FLAGS_value_size),
                          static_cast<size_t>(FLAGS_value_size));
        }
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }

  start.store(true, std::memory_order_release);
  sleep(FLAGS_running_seconds);
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
    total.misses += each.misses;
  }
  return total;
}

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

void PrintResult(const char* phase, const PhaseStats& stats, double seconds) {
  const double batch_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.batches) / seconds : 0.0;
  const double key_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.key_ops) / seconds : 0.0;
  std::printf(
      "HPS_BACKEND_RESULT phase=%s backend=%s index_type=%s "
      "value_store_type=%s mode=%s distribution=%s zipfian_alpha=%.6f "
      "threads=%d batch_size=%d records=%ld runtime_s=%.6f batches=%lu "
      "key_ops=%lu misses=%lu throughput_batches_sec=%.6f "
      "throughput_keys_sec=%.6f\n",
      phase,
      FLAGS_backend.c_str(),
      FLAGS_index_type.c_str(),
      FLAGS_value_store_type.c_str(),
      FLAGS_mode.c_str(),
      FLAGS_distribution.c_str(),
      FLAGS_zipfian_alpha,
      FLAGS_thread_num,
      FLAGS_batch_size,
      FLAGS_record_count,
      seconds,
      stats.batches,
      stats.key_ops,
      stats.misses,
      batch_ops_sec,
      key_ops_sec);
}

} // namespace

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  CHECK_GT(FLAGS_record_count, 0);
  CHECK_GT(FLAGS_value_size, 0);
  CHECK_GT(FLAGS_batch_size, 0);
  CHECK_GT(FLAGS_thread_num, 0);
  CHECK_GT(FLAGS_running_seconds, 0);
  if (FLAGS_backend == "recstore" || FLAGS_backend == "hps_rocksdb") {
    CHECK(!FLAGS_path.empty()) << "--path is required for " << FLAGS_backend
                               << " backend";
  }

  const int load_threads =
      FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;
  auto backend = CreateBackend();
  PrepareBackendForLoad(backend.get());

  const auto load_begin = std::chrono::steady_clock::now();
  const PhaseStats load = LoadRecords(backend.get(), load_threads);
  const auto load_end   = std::chrono::steady_clock::now();
  PrintResult("load", load, SecondsSince(load_begin, load_end));

  const auto run_begin = std::chrono::steady_clock::now();
  const PhaseStats run = RunTransactions(backend.get());
  const auto run_end   = std::chrono::steady_clock::now();
  PrintResult("run", run, SecondsSince(run_begin, run_end));
  return 0;
}
