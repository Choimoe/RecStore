#include <folly/init/Init.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/bind_core.h"
#include "benchmark/ps_transport_benchmark_config.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/base/base_client.h"
#include "ps/client_factory.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/brpc/dist_brpc_ps_client.h"
#include "ps/grpc/dist_grpc_ps_client.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

DEFINE_string(transport, "rdma", "rdma|grpc|brpc");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 25000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_string(config_path, "", "full RecStore config path for PS client");
DEFINE_int32(iterations, 100, "number of get/put iterations");
DEFINE_int32(rounds, 1, "number of measured rounds");
DEFINE_int32(warmup_rounds, 0, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 4, "number of keys per put/get RPC");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DEFINE_string(workload, "micro", "benchmark workload: micro|transactions");
DEFINE_string(mode,
              "fetch",
              "transactions mode: fetch|insert|mixed|fetch_insert");
DEFINE_int64(record_count, 1000000, "record count for transactions workload");
DEFINE_int32(thread_num, 16, "worker thread count for transactions workload");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");
DEFINE_string(distribution, "uniform", "uniform|zipfian");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_int32(read_ratio, 100, "read percentage for mixed mode");
DECLARE_int32(value_size);

namespace {

struct PhaseStats {
  uint64_t batches = 0;
  uint64_t key_ops = 0;
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

std::vector<uint64_t> MakeKeys(int batch_keys) {
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    keys.push_back(static_cast<uint64_t>(1001 + i));
  }
  return keys;
}

std::vector<std::vector<float>> MakeValues(const std::vector<uint64_t>& keys) {
  const int dim = FLAGS_value_size / sizeof(float);
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row(dim, static_cast<float>(key));
    values.push_back(std::move(row));
  }
  return values;
}

std::vector<float> MakeFlatValues(size_t rows, int dim, int seed) {
  std::vector<float> values(rows * static_cast<size_t>(dim));
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((static_cast<int>(i) + seed) % 101);
  }
  return values;
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

void MaybePrintPerRound(
    const std::string& transport,
    const std::string& op,
    const std::string& report_mode,
    bool is_warmup,
    int round,
    int warmup_rounds,
    int rounds,
    int64_t elapsed_us) {
  if (!ShouldPrintPerRound(report_mode)) {
    return;
  }
  std::cout << "transport=" << transport << " op=" << op
            << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
            << (is_warmup ? (round + 1) : (round - warmup_rounds + 1)) << "/"
            << (is_warmup ? warmup_rounds : rounds)
            << " elapsed_us=" << elapsed_us << std::endl;
}

template <typename IterationFn>
void RunOperationRounds(
    const std::string& transport,
    const std::string& op,
    int total_rounds,
    int warmup_rounds,
    int rounds,
    int iterations,
    const std::string& report_mode,
    IterationFn&& run_iteration,
    std::vector<int64_t>* warmup_samples_us,
    std::vector<int64_t>* measure_samples_us) {
  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < warmup_rounds;
    auto start           = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      run_iteration(i);
    }
    auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? *warmup_samples_us : *measure_samples_us)
        .push_back(elapsed_us);
    MaybePrintPerRound(
        transport,
        op,
        report_mode,
        is_warmup,
        round,
        warmup_rounds,
        rounds,
        elapsed_us);
  }
}

double PercentileUs(std::vector<int64_t> samples, double ratio) {
  CHECK(!samples.empty());
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(std::min<int64_t>(
      samples.size() - 1,
      static_cast<int64_t>(std::ceil(ratio * samples.size()) - 1)));
  return static_cast<double>(samples[idx]);
}

void PrintSummary(
    const std::string& transport,
    const std::string& op,
    const std::string& phase,
    const std::vector<int64_t>& elapsed_us_samples,
    int iterations_per_round,
    int batch_keys,
    std::size_t keys_per_iteration) {
  if (elapsed_us_samples.empty()) {
    return;
  }

  const double total_us = std::accumulate(
      elapsed_us_samples.begin(), elapsed_us_samples.end(), 0.0);
  const double mean_us =
      total_us / static_cast<double>(elapsed_us_samples.size());
  const double p50_us = PercentileUs(elapsed_us_samples, 0.50);
  const double p95_us = PercentileUs(elapsed_us_samples, 0.95);
  const double p99_us = PercentileUs(elapsed_us_samples, 0.99);

  const double ops_per_round = static_cast<double>(iterations_per_round);
  const double key_ops_per_round =
      ops_per_round * static_cast<double>(keys_per_iteration);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec  = (ops_per_round * total_rounds) / (total_us / 1e6);
  const double key_ops_per_sec =
      (key_ops_per_round * total_rounds) / (total_us / 1e6);

  std::cout << "transport=" << transport << " op=" << op << " phase=" << phase
            << " summary rounds=" << elapsed_us_samples.size()
            << " iterations=" << iterations_per_round
            << " batch_keys=" << batch_keys << " elapsed_us_mean=" << mean_us
            << " elapsed_us_p50=" << p50_us << " elapsed_us_p95=" << p95_us
            << " elapsed_us_p99=" << p99_us << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

nlohmann::json LoadClientConfig(const std::string& transport) {
  if (!FLAGS_config_path.empty()) {
    std::ifstream in(FLAGS_config_path);
    CHECK(in.good()) << "failed to open --config_path=" << FLAGS_config_path;
    nlohmann::json config;
    in >> config;
    return config;
  }
  return BuildRpcBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
}

std::unique_ptr<recstore::BasePSClient>
CreateBenchmarkClient(const std::string& transport) {
  auto config           = LoadClientConfig(transport);
  const auto normalized = NormalizeBenchmarkTransport(transport);
  if (FLAGS_num_shards > 1 && normalized == "BRPC") {
    return std::make_unique<recstore::DistributedBRPCParameterClient>(config);
  }
  if (FLAGS_num_shards > 1 && normalized == "GRPC") {
    return std::make_unique<recstore::DistributedGRPCParameterClient>(config);
  }
  return recstore::CreatePSClient(
      recstore::ResolvePSClientOptionsFromFrameworkConfig(config));
}

bool PutFlat(recstore::BasePSClient* client,
             const std::string& transport,
             const std::vector<uint64_t>& keys,
             const std::vector<float>& flat_values,
             int dim) {
  std::vector<std::vector<float>> rows;
  rows.reserve(keys.size());
  for (size_t row = 0; row < keys.size(); ++row) {
    const float* begin = flat_values.data() + row * static_cast<size_t>(dim);
    rows.emplace_back(begin, begin + dim);
  }
  return BenchmarkWriteSucceeded(
      transport, client->PutParameter(base::ConstArray<uint64_t>(keys), rows));
}

bool GetFlat(recstore::BasePSClient* client,
             const std::string& transport,
             const std::vector<uint64_t>& keys,
             std::vector<float>* output) {
  const auto key_array = base::ConstArray<uint64_t>(keys);
  if (BenchmarkUsesVectorGet(transport)) {
    auto* brpc_client = dynamic_cast<BRPCParameterClient*>(client);
    if (brpc_client != nullptr) {
      std::vector<std::vector<float>> vectors;
      return BenchmarkReadSucceeded(
          transport, brpc_client->GetParameter(key_array, &vectors));
    }
  }
  return BenchmarkReadSucceeded(
      transport, client->GetParameter(key_array, output->data()));
}

PhaseStats
LoadRecords(const std::string& transport, int load_threads, int dim) {
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(load_threads);
  const uint64_t record_count = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t per_thread =
      (record_count + static_cast<uint64_t>(load_threads) - 1) /
      static_cast<uint64_t>(load_threads);

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      auto client = CreateBenchmarkClient(transport);
      std::vector<uint64_t> keys;
      keys.reserve(static_cast<size_t>(FLAGS_batch_keys));
      std::vector<float> values =
          MakeFlatValues(static_cast<size_t>(FLAGS_batch_keys), dim, tid);
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread + 1;
      const uint64_t end   = std::min(record_count + 1, begin + per_thread);
      PhaseStats local;
      for (uint64_t key = begin; key < end;) {
        keys.clear();
        while (key < end &&
               keys.size() < static_cast<size_t>(FLAGS_batch_keys)) {
          keys.push_back(key++);
        }
        CHECK(PutFlat(client.get(), transport, keys, values, dim))
            << transport << " preload PutParameter failed";
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

PhaseStats RunTransactions(const std::string& transport, int dim) {
  const bool fetch_only   = FLAGS_mode == "fetch";
  const bool insert_only  = FLAGS_mode == "insert";
  const bool mixed        = FLAGS_mode == "mixed";
  const bool fetch_insert = FLAGS_mode == "fetch_insert";
  CHECK(fetch_only || insert_only || mixed || fetch_insert)
      << "mode must be fetch|insert|mixed|fetch_insert";

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      auto client = CreateBenchmarkClient(transport);
      KeyGenerator key_gen(
          FLAGS_distribution,
          static_cast<uint64_t>(FLAGS_record_count),
          FLAGS_zipfian_alpha,
          0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(tid));
      std::vector<uint64_t> keys(static_cast<size_t>(FLAGS_batch_keys));
      std::vector<float> values =
          MakeFlatValues(static_cast<size_t>(FLAGS_batch_keys), dim, tid);
      std::vector<float> output(
          static_cast<size_t>(FLAGS_batch_keys) * static_cast<size_t>(dim));
      PhaseStats local;
      while (!start.load(std::memory_order_acquire)) {
      }
      while (!stop.load(std::memory_order_relaxed)) {
        for (auto& key : keys) {
          key = key_gen.NextKey();
        }
        const bool do_fetch =
            fetch_only || fetch_insert ||
            (mixed &&
             static_cast<int>(key_gen.NextUint(100)) < FLAGS_read_ratio);
        if (do_fetch) {
          CHECK(GetFlat(client.get(), transport, keys, &output))
              << transport << " GetParameter failed";
        }
        if (insert_only || fetch_insert || (mixed && !do_fetch)) {
          CHECK(PutFlat(client.get(), transport, keys, values, dim))
              << transport << " PutParameter failed";
        }
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }

  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::seconds(FLAGS_running_seconds));
  stop.store(true, std::memory_order_relaxed);
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

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

void PrintTransactionResult(const char* phase,
                            const std::string& transport,
                            const PhaseStats& stats,
                            double seconds) {
  const double batch_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.batches) / seconds : 0.0;
  const double key_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.key_ops) / seconds : 0.0;
  std::cout << "PS_BENCHMARK_RESULT phase=" << phase
            << " transport=" << transport << " mode=" << FLAGS_mode
            << " distribution=" << FLAGS_distribution << " zipfian_alpha="
            << FLAGS_zipfian_alpha << " threads=" << FLAGS_thread_num
            << " batch_size=" << FLAGS_batch_keys
            << " records=" << FLAGS_record_count << " runtime_s=" << seconds
            << " batches=" << stats.batches << " key_ops=" << stats.key_ops
            << " throughput_batches_sec=" << batch_ops_sec
            << " throughput_keys_sec=" << key_ops_sec << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);

  const std::string transport   = NormalizeBenchmarkTransport(FLAGS_transport);
  const std::string report_mode = FLAGS_report_mode;
  CHECK(report_mode == "summary" || report_mode == "per_round" ||
        report_mode == "both")
      << "Invalid --report_mode: " << report_mode
      << ", expected summary|per_round|both";
  CHECK_GT(FLAGS_value_size, 0) << "--value_size must be positive";
  CHECK_EQ(FLAGS_value_size % static_cast<int>(sizeof(float)), 0)
      << "--value_size must be divisible by sizeof(float)";
  const int dim = FLAGS_value_size / sizeof(float);

  if (FLAGS_workload == "transactions") {
    CHECK_GT(FLAGS_record_count, 0);
    CHECK_GT(FLAGS_batch_keys, 0);
    CHECK_GT(FLAGS_thread_num, 0);
    CHECK_GT(FLAGS_running_seconds, 0);
    const int load_threads =
        FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;
    const auto load_begin = std::chrono::steady_clock::now();
    const PhaseStats load = LoadRecords(transport, load_threads, dim);
    const auto load_end   = std::chrono::steady_clock::now();
    PrintTransactionResult(
        "load", transport, load, SecondsSince(load_begin, load_end));

    const auto run_begin = std::chrono::steady_clock::now();
    const PhaseStats run = RunTransactions(transport, dim);
    const auto run_end   = std::chrono::steady_clock::now();
    PrintTransactionResult(
        "run", transport, run, SecondsSince(run_begin, run_end));
    return 0;
  }
  CHECK_EQ(FLAGS_workload, "micro") << "workload must be micro|transactions";

  const auto keys        = MakeKeys(FLAGS_batch_keys);
  const auto values      = MakeValues(keys);
  const auto key_array   = base::ConstArray<uint64_t>(keys);
  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;

  if (transport == "RDMA") {
    if (FLAGS_num_shards == 1) {
      petps::PetPSClient client(FLAGS_host, FLAGS_port, 0);
      client.InitThread();
      void* recv_buffer =
          client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
      std::vector<int64_t> put_warmup_samples_us;
      std::vector<int64_t> put_measure_samples_us;
      put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
      put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
      RunOperationRounds(
          "RDMA",
          "put",
          total_rounds,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          FLAGS_iterations,
          report_mode,
          [&](int iteration) {
            CHECK_EQ(client.PutParameter(keys, values), 0)
                << "RDMA PutParameter failed at iteration=" << iteration;
          },
          &put_warmup_samples_us,
          &put_measure_samples_us);

      std::vector<int64_t> get_warmup_samples_us;
      std::vector<int64_t> get_measure_samples_us;
      get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
      get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
      RunOperationRounds(
          "RDMA",
          "get",
          total_rounds,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          FLAGS_iterations,
          report_mode,
          [&](int iteration) {
            int rpc_id = client.GetParameter(
                key_array, static_cast<float*>(recv_buffer), false, 0);
            client.WaitRPCFinish(rpc_id);
            client.RevokeRPCResource(rpc_id);
            (void)iteration;
          },
          &get_warmup_samples_us,
          &get_measure_samples_us);
      if (ShouldPrintSummary(report_mode)) {
        PrintSummary(
            "RDMA",
            "put",
            "warmup",
            put_warmup_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "put",
            "measure",
            put_measure_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "get",
            "warmup",
            get_warmup_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "get",
            "measure",
            get_measure_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
      }
      return 0;
    }

    std::vector<std::unique_ptr<petps::PetPSClient>> owned;
    std::vector<BaseParameterClient*> clients;
    for (int shard = 0; shard < FLAGS_num_shards; ++shard) {
      owned.push_back(
          std::make_unique<petps::PetPSClient>(FLAGS_host, FLAGS_port, shard));
      owned.back()->InitThread();
      clients.push_back(owned.back().get());
    }

    AllShardsParameterClientWrapper client(clients, FLAGS_num_shards);
    client.InitThread();
    std::vector<int64_t> put_warmup_samples_us;
    std::vector<int64_t> put_measure_samples_us;
    put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
    put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
    RunOperationRounds(
        "RDMA",
        "put",
        total_rounds,
        FLAGS_warmup_rounds,
        FLAGS_rounds,
        FLAGS_iterations,
        report_mode,
        [&](int iteration) {
          CHECK_EQ(client.PutParameter(keys, values), 0)
              << "RDMA(all-shards) PutParameter failed at iteration="
              << iteration;
        },
        &put_warmup_samples_us,
        &put_measure_samples_us);

    std::vector<int64_t> get_warmup_samples_us;
    std::vector<int64_t> get_measure_samples_us;
    get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
    get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
    RunOperationRounds(
        "RDMA",
        "get",
        total_rounds,
        FLAGS_warmup_rounds,
        FLAGS_rounds,
        FLAGS_iterations,
        report_mode,
        [&](int iteration) {
          std::vector<float> output(
              keys.size() * (FLAGS_value_size / sizeof(float)) + 1, 0.0f);
          int rpc_id = client.GetParameter(key_array, output.data(), false, 0);
          client.WaitRPCFinish(rpc_id);
          client.RevokeRPCResource(rpc_id);
          (void)iteration;
        },
        &get_warmup_samples_us,
        &get_measure_samples_us);
    if (ShouldPrintSummary(report_mode)) {
      PrintSummary(
          "RDMA",
          "put",
          "warmup",
          put_warmup_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "put",
          "measure",
          put_measure_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "get",
          "warmup",
          get_warmup_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "get",
          "measure",
          get_measure_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
    }
    return 0;
  }

  std::unique_ptr<recstore::BasePSClient> client =
      CreateBenchmarkClient(transport);

  std::vector<int64_t> put_warmup_samples_us;
  std::vector<int64_t> put_measure_samples_us;
  put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
  RunOperationRounds(
      transport,
      "put",
      total_rounds,
      FLAGS_warmup_rounds,
      FLAGS_rounds,
      FLAGS_iterations,
      report_mode,
      [&](int iteration) {
        const int ret = client->PutParameter(key_array, values);
        CHECK(BenchmarkWriteSucceeded(transport, ret))
            << transport << " PutParameter failed at iteration=" << iteration;
      },
      &put_warmup_samples_us,
      &put_measure_samples_us);

  std::vector<int64_t> get_warmup_samples_us;
  std::vector<int64_t> get_measure_samples_us;
  get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
  RunOperationRounds(
      transport,
      "get",
      total_rounds,
      FLAGS_warmup_rounds,
      FLAGS_rounds,
      FLAGS_iterations,
      report_mode,
      [&](int iteration) {
        if (BenchmarkUsesVectorGet(transport)) {
          auto* brpc_client = dynamic_cast<BRPCParameterClient*>(client.get());
          CHECK_NE(brpc_client, nullptr);
          std::vector<std::vector<float>> output;
          const int ret = brpc_client->GetParameter(key_array, &output);
          CHECK(BenchmarkReadSucceeded(transport, ret))
              << transport << " GetParameter failed at iteration=" << iteration;
        } else {
          std::vector<float> output(
              keys.size() * (FLAGS_value_size / sizeof(float)), 0.0f);
          const int ret = client->GetParameter(key_array, output.data());
          CHECK(BenchmarkReadSucceeded(transport, ret))
              << transport << " GetParameter failed at iteration=" << iteration;
        }
      },
      &get_warmup_samples_us,
      &get_measure_samples_us);
  if (ShouldPrintSummary(report_mode)) {
    PrintSummary(
        transport,
        "put",
        "warmup",
        put_warmup_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "put",
        "measure",
        put_measure_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "get",
        "warmup",
        get_warmup_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "get",
        "measure",
        get_measure_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
  }
  return 0;
}
