#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "base/array.h"
#include "benchmark/ps_transport_benchmark_config.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

DEFINE_string(transport, "rdma", "rdma|grpc|brpc");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 25000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_int32(iterations, 100, "number of get/put iterations");
DEFINE_int32(rounds, 1, "number of measured rounds");
DEFINE_int32(warmup_rounds, 0, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 4, "number of keys per put/get RPC");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DECLARE_int32(value_size);

namespace {

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

} // namespace

int main(int argc, char** argv) {
  const std::string transport   = NormalizeBenchmarkTransport(FLAGS_transport);
  const std::string report_mode = FLAGS_report_mode;
  CHECK(report_mode == "summary" || report_mode == "per_round" ||
        report_mode == "both")
      << "Invalid --report_mode: " << report_mode
      << ", expected summary|per_round|both";
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

  auto config = BuildRpcBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
  std::unique_ptr<recstore::BasePSClient> client(recstore::CreatePSClient(
      recstore::ResolvePSClientOptionsFromFrameworkConfig(config)));

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
        CHECK_EQ(client->PutParameter(key_array, values), 0)
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
          brpc_client->GetParameter(key_array, &output);
        } else {
          std::vector<float> output(
              keys.size() * (FLAGS_value_size / sizeof(float)), 0.0f);
          client->GetParameter(key_array, output.data());
        }
        (void)iteration;
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
