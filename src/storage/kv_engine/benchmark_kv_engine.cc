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
#include <string_view>
#include <thread>
#include <vector>

#include "base/bind_core.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/log.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_cceh.h"
#include "storage/kv_engine/engine_extendible_hash.h"
#include "storage/kv_engine/engine_selector.h"

DEFINE_string(
    dram_path,
    "",
    "DRAM data directory; empty only works with anonymous DRAM allocators");
DEFINE_string(ssd_path, "", "SSD data directory");
DEFINE_string(index_type, "DRAM_EXTENDIBLE_HASH", "index.type");
DEFINE_string(value_store_type, "DRAM_VALUE_STORE", "value.type");
DEFINE_string(engine_class, "KVEngineComposite", "BaseKV factory class name");
DEFINE_string(dram_allocator, "PERSIST_LOOP_SLAB", "value.dram_allocator.type");
DEFINE_string(ssd_io_backend, "IOURING", "SSD IO backend");
DEFINE_int32(ssd_queue_depth, 512, "SSD IO queue depth");
DEFINE_int64(dram_capacity_bytes, 0, "override DRAM allocator capacity bytes");
DEFINE_int64(ssd_capacity_bytes, 0, "override SSD allocator capacity bytes");

DEFINE_int64(record_count, 100000000, "YCSB record count");
DEFINE_string(workload,
              "c",
              "YCSB workload: a/b/c or workloada/workloadb/workloadc");
DEFINE_string(distribution, "uniform", "key distribution: uniform/zipfian");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_string(read_mode, "exists", "read mode: exists/get");
DEFINE_bool(load, true, "run load phase");
DEFINE_bool(run, true, "run transaction phase");
DEFINE_bool(print_util, false, "print KVEngine utilization after load");

DEFINE_int32(thread_num, 16, "worker thread count");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(value_size, 128, "value size bytes");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");

namespace {

struct PhaseStats {
  uint64_t total_ops  = 0;
  uint64_t read_ops   = 0;
  uint64_t update_ops = 0;
};

struct WorkloadMix {
  int read_percent = 100;
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
               double zipfian_alpha,
               uint64_t seed)
      : distribution_(std::move(distribution)),
        record_count_(record_count),
        alpha_(zipfian_alpha),
        rng_(seed) {
    if (record_count_ == 0) {
      throw std::invalid_argument("record_count must be positive");
    }
    if (distribution_ == "zipfian") {
      if (alpha_ < 0.0) {
        throw std::invalid_argument("zipfian_alpha must be non-negative");
      }
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

std::string NormalizeWorkload(std::string workload) {
  std::transform(workload.begin(), workload.end(), workload.begin(), ::tolower);
  if (workload == "workloada")
    return "a";
  if (workload == "workloadb")
    return "b";
  if (workload == "workloadc")
    return "c";
  return workload;
}

WorkloadMix GetWorkloadMix(const std::string& workload) {
  if (workload == "a")
    return WorkloadMix{50};
  if (workload == "b")
    return WorkloadMix{95};
  if (workload == "c")
    return WorkloadMix{100};
  throw std::invalid_argument(
      "workload must be a/b/c or workloada/workloadb/workloadc");
}

bool IsSsdIndexType(const std::string& type) {
  return type == "SSD" || type == "SSD_EXTENDIBLE_HASH";
}

bool HasDramValueStore(const std::string& type) {
  return type == "DRAM_VALUE_STORE" || type == "TIERED_VALUE_STORE";
}

bool HasSsdValueStore(const std::string& type) {
  return type == "SSD_VALUE_STORE" || type == "TIERED_VALUE_STORE";
}

bool IsLegacyDirectEngine(const std::string& engine_class) {
  return engine_class == "KVEngineExtendibleHash" ||
         engine_class == "KVEngineCCEH";
}

BaseKVConfig BuildConfig() {
  BaseKVConfig config;
  const uint64_t capacity = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t value_slot_bytes =
      static_cast<uint64_t>(std::max(FLAGS_value_size, 1)) + sizeof(uint64_t);
  const uint64_t value_capacity           = capacity * value_slot_bytes * 6 / 5;
  constexpr uint64_t kMinSsdCapacityBytes = 256ULL * 1024ULL * 1024ULL;
  if (FLAGS_engine_class == "KVEngineExtendibleHash") {
    const uint64_t dram_capacity =
        FLAGS_dram_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_dram_capacity_bytes)
            : value_capacity;
    const uint64_t ssd_capacity =
        FLAGS_ssd_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_ssd_capacity_bytes)
            : std::max(value_capacity, kMinSsdCapacityBytes);
    std::string value_medium = "DRAM";
    if (FLAGS_value_store_type == "SSD_VALUE_STORE") {
      value_medium = "SSD";
    } else if (FLAGS_value_store_type != "DRAM_VALUE_STORE") {
      throw std::invalid_argument("KVEngineExtendibleHash only supports "
                                  "DRAM_VALUE_STORE/SSD_VALUE_STORE");
    }
    const std::string engine_path =
        (value_medium == "SSD" && !FLAGS_ssd_path.empty())
            ? FLAGS_ssd_path
            : FLAGS_dram_path;
    config.json_config_ = {
        {"capacity", capacity},
        {"path", engine_path},
        {"value_size", FLAGS_value_size},
        {"value_type", value_medium},
        {"DRAM_SIZE", dram_capacity},
        {"SSD_SIZE", ssd_capacity}};
    config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
    return config;
  }
  if (FLAGS_engine_class == "KVEngineCCEH") {
    const uint64_t ssd_capacity =
        FLAGS_ssd_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_ssd_capacity_bytes)
            : std::max(value_capacity, kMinSsdCapacityBytes);
    config.json_config_ = {
        {"capacity", capacity},
        {"path", FLAGS_ssd_path},
        {"value_size", FLAGS_value_size},
        {"queue_cnt", std::max(FLAGS_thread_num, 1)},
        {"io_backend_type", FLAGS_ssd_io_backend},
        {"SSD_SIZE", ssd_capacity}};
    config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
    return config;
  }

  const bool ssd_index = IsSsdIndexType(FLAGS_index_type);

  config.json_config_ = {
      {"capacity", capacity},
      {"index", {{"type", FLAGS_index_type}}},
      {"value",
       {{"type", FLAGS_value_store_type},
        {"default_value_size_hint", FLAGS_value_size}}}};

  if (ssd_index) {
    config.json_config_["index"]["path"] = FLAGS_ssd_path + "/index.db";
    config.json_config_["index"]["io"]   = {
        {"type", FLAGS_ssd_io_backend},
        {"queue_depth", FLAGS_ssd_queue_depth},
        {"base_offset_bytes", 0}};
  }

  const uint64_t dram_capacity =
      FLAGS_dram_capacity_bytes > 0
          ? static_cast<uint64_t>(FLAGS_dram_capacity_bytes)
          : value_capacity;
  const uint64_t ssd_capacity =
      FLAGS_ssd_capacity_bytes > 0
          ? static_cast<uint64_t>(FLAGS_ssd_capacity_bytes)
          : std::max(value_capacity, kMinSsdCapacityBytes);

  if (FLAGS_value_store_type == "DRAM_VALUE_STORE") {
    if (!FLAGS_dram_path.empty()) {
      config.json_config_["value"]["path"] = FLAGS_dram_path + "/value";
    }
    config.json_config_["value"]["dram_allocator"] = {
        {"type", FLAGS_dram_allocator}, {"capacity_bytes", dram_capacity}};
  }
  if (FLAGS_value_store_type == "SSD_VALUE_STORE") {
    config.json_config_["value"]["path"] = FLAGS_ssd_path + "/value.db";
    config.json_config_["value"]["ssd_allocator"] = {
        {"type", "SSD_SLAB"},
        {"capacity_bytes", ssd_capacity},
        {"min_block_size", 128},
        {"max_block_size", 4096},
        {"io",
         {{"type", FLAGS_ssd_io_backend},
          {"queue_depth", FLAGS_ssd_queue_depth},
          {"base_offset_bytes", 4096}}}};
  } else if (FLAGS_value_store_type == "TIERED_VALUE_STORE") {
    config.json_config_["value"]["dram_allocator"] = {
        {"type", FLAGS_dram_allocator},
        {"capacity_bytes", dram_capacity},
        {"path", FLAGS_dram_path + "/dram"}};
    config.json_config_["value"]["ssd_allocator"] = {
        {"type", "SSD_SLAB"},
        {"capacity_bytes", ssd_capacity},
        {"min_block_size", 128},
        {"max_block_size", 4096},
        {"path", FLAGS_ssd_path + "/ssd.db"},
        {"io",
         {{"type", FLAGS_ssd_io_backend},
          {"queue_depth", FLAGS_ssd_queue_depth},
          {"base_offset_bytes", 4096}}}};
    config.json_config_["value"]["tiering"] = {{"cache_policy", "LRU"}};
  }

  LOG(INFO) << config.json_config_.dump(2);

  config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
  return config;
}

PhaseStats LoadRecords(BaseKV* kv, int load_threads, uint64_t record_count) {
  std::vector<std::thread> threads;
  std::vector<uint64_t> counts(load_threads, 0);
  const uint64_t per_thread = (record_count + load_threads - 1) / load_threads;

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([kv, tid, per_thread, record_count, &counts]() {
      base::auto_bind_core();
      std::string value(FLAGS_value_size, static_cast<char>('a' + (tid % 26)));
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread + 1;
      const uint64_t end   = std::min(record_count + 1, begin + per_thread);
      for (uint64_t key = begin; key < end; ++key) {
        kv->Put(key, std::string_view(value.data(), value.size()), tid);
        ++counts[tid];
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats stats;
  for (uint64_t count : counts) {
    stats.total_ops += count;
    stats.update_ops += count;
  }
  return stats;
}

PhaseStats RunTransactions(
    BaseKV* kv,
    const std::string& workload,
    const std::string& distribution,
    int threads_num,
    uint64_t record_count,
    int seconds) {
  const WorkloadMix mix = GetWorkloadMix(workload);
  const bool use_exists = FLAGS_read_mode == "exists";
  if (!use_exists && FLAGS_read_mode != "get") {
    throw std::invalid_argument("read_mode must be exists or get");
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(threads_num);

  for (int tid = 0; tid < threads_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      KeyGenerator key_gen(
          distribution,
          record_count,
          FLAGS_zipfian_alpha,
          0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(tid));
      std::string value(FLAGS_value_size, static_cast<char>('A' + (tid % 26)));
      std::string read_value;
      PhaseStats local;
      while (!start.load(std::memory_order_acquire)) {
      }
      while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t key = key_gen.NextKey();
        const bool do_read =
            static_cast<int>(key_gen.NextUint(100)) < mix.read_percent;
        if (do_read) {
          if (use_exists) {
            (void)kv->Exists(key, tid);
          } else {
            kv->Get(key, read_value, tid);
          }
          ++local.read_ops;
        } else {
          kv->Put(key, std::string_view(value.data(), value.size()), tid);
          ++local.update_ops;
        }
        ++local.total_ops;
      }
      stats[tid] = local;
    });
  }

  start.store(true, std::memory_order_release);
  sleep(seconds);
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.total_ops += each.total_ops;
    total.read_ops += each.read_ops;
    total.update_ops += each.update_ops;
  }
  return total;
}

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

} // namespace

int main(int argc, char* argv[]) {
  base::Init(&argc, &argv);

  if (FLAGS_record_count <= 0) {
    LOG(FATAL) << "record_count must be positive";
  }
  if (FLAGS_thread_num <= 0) {
    LOG(FATAL) << "thread_num must be positive";
  }
  if (FLAGS_value_size <= 0) {
    LOG(FATAL) << "value_size must be positive";
  }
  if (FLAGS_running_seconds <= 0) {
    LOG(FATAL) << "running_seconds must be positive";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      IsSsdIndexType(FLAGS_index_type) && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for SSD index";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      HasSsdValueStore(FLAGS_value_store_type) && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for SSD/TIERED value store";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      FLAGS_value_store_type == "TIERED_VALUE_STORE" &&
      FLAGS_dram_path.empty()) {
    LOG(FATAL) << "dram_path must be set for TIERED value store";
  }
  if (FLAGS_engine_class == "KVEngineCCEH" && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for KVEngineCCEH";
  }
  if (FLAGS_engine_class == "KVEngineExtendibleHash" &&
      FLAGS_dram_path.empty() && FLAGS_ssd_path.empty()) {
    LOG(FATAL)
        << "dram_path or ssd_path must be set for KVEngineExtendibleHash";
  }

  const std::string workload = NormalizeWorkload(FLAGS_workload);
  (void)GetWorkloadMix(workload);
  const int load_threads =
      FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;

  base::PMMmapRegisterCenter::GetConfig().use_dram =
      HasDramValueStore(FLAGS_value_store_type);

  BaseKVConfig config = BuildConfig();
  std::unique_ptr<BaseKV> kv;
  if (FLAGS_engine_class == "KVEngine") {
    auto resolved = base::ResolveEngine(config);
    kv.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        resolved.engine, resolved.cfg));
  } else {
    kv.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        FLAGS_engine_class, config));
  }
  if (!kv) {
    LOG(FATAL) << "failed to create KVEngine";
  }

  if (FLAGS_load) {
    const auto begin      = std::chrono::steady_clock::now();
    PhaseStats load_stats = LoadRecords(
        kv.get(), load_threads, static_cast<uint64_t>(FLAGS_record_count));
    const auto end       = std::chrono::steady_clock::now();
    const double seconds = SecondsSince(begin, end);
    const double throughput =
        seconds > 0.0 ? static_cast<double>(load_stats.total_ops) / seconds
                      : 0.0;
    std::printf(
        "YCSB_LOAD_RESULT records=%ld threads=%d seconds=%.6f ops=%lu "
        "throughput_ops_sec=%.6f\n",
        FLAGS_record_count,
        load_threads,
        seconds,
        load_stats.total_ops,
        throughput);
    if (FLAGS_print_util) {
      kv->Util();
    }
  }

  if (FLAGS_run) {
    const auto begin     = std::chrono::steady_clock::now();
    PhaseStats run_stats = RunTransactions(
        kv.get(),
        workload,
        FLAGS_distribution,
        FLAGS_thread_num,
        static_cast<uint64_t>(FLAGS_record_count),
        FLAGS_running_seconds);
    const auto end       = std::chrono::steady_clock::now();
    const double seconds = SecondsSince(begin, end);
    const double throughput =
        seconds > 0.0 ? static_cast<double>(run_stats.total_ops) / seconds
                      : 0.0;
    std::printf(
        "YCSB_RESULT workload=%s distribution=%s zipfian_alpha=%.6f "
        "read_mode=%s threads=%d records=%ld runtime_s=%.6f ops=%lu "
        "throughput_ops_sec=%.6f read_ops=%lu update_ops=%lu\n",
        workload.c_str(),
        FLAGS_distribution.c_str(),
        FLAGS_zipfian_alpha,
        FLAGS_read_mode.c_str(),
        FLAGS_thread_num,
        FLAGS_record_count,
        seconds,
        run_stats.total_ops,
        throughput,
        run_stats.read_ops,
        run_stats.update_ops);
  }

  LOG(INFO) << "gracefully exit";
  return 0;
}
