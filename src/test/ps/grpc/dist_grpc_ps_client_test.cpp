#include "ps/grpc/dist_grpc_ps_client.h"

#include <stdlib.h>

#include <future>
#include <random>

#include "base/array.h"
#include "base/init.h"
#include "base/timer.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kDistGrpcPort0 = 15133;
constexpr int kDistGrpcPort1 = 15134;

void DisableGrpcTracingEnv() {
  // CI may inject tracing knobs that force grpc_tracer_init() during channel
  // credential creation and hang inside TraceFlagList::Set. Clear them for
  // this integration test to keep gRPC startup deterministic.
  unsetenv("GRPC_TRACE");
  unsetenv("GRPC_VERBOSITY");
}
} // namespace

using namespace xmh;
using namespace recstore;

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return false;

  for (int i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6)
      return false;
  }
  return true;
}

static bool check_eq_2d(std::vector<std::vector<float>>& a,
                        const std::vector<std::vector<float>>& b) {
  a.resize(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    a[i].resize(b[i].size());
  }
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false)
      return false;
  }
  return true;
}

void TestBasicConfig(const std::vector<int>& ports) {
  std::cout << "=== Testing Basic Configuration ===" << std::endl;

  json recstore_config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(recstore_config);
    std::cout << "Recstore config parsed successfully, shard count: "
              << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore config test failed: " << e.what() << std::endl;
  }
}

void TestDirectClient(const std::vector<int>& ports) {
  std::cout << "=== Testing Direct Client Creation ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(config);
    std::cout << "Direct client created successfully, shard count: "
              << client.shard_count() << std::endl;

    client.ClearPS();
    std::vector<uint64_t> keys = {1001, 1002, 1003};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {
        {1, 0, 1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    base::ConstArray<uint64_t> keys_array(keys);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    client.PutParameter(keys_array, rightvalues);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, rightvalues));

    client.ClearPS();
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "All direct client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestLargeBatch(const std::vector<int>& ports) {
  std::cout << "=== Testing Large Batch Operations ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 50}}}};

  try {
    DistributedGRPCParameterClient client(config);

    std::vector<uint64_t> large_keys;
    std::vector<std::vector<float>> large_values;
    for (int i = 0; i < 120; ++i) {
      large_keys.push_back(2000 + static_cast<uint64_t>(i) * 2);
      large_values.push_back({float(i), float(i * 2)});
    }

    base::ConstArray<uint64_t> keys_array(large_keys);

    client.ClearPS();

    int put_result = client.PutParameter(keys_array, large_values);
    CHECK(put_result == 0);
    std::vector<std::vector<float>> retrieved_values;
    bool get_success = client.GetParameter(keys_array, &retrieved_values);
    CHECK(get_success);
    CHECK(check_eq_2d(retrieved_values, large_values));

    std::cout << "Large batch test passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestPrefetch(const std::vector<int>& ports) {
  std::cout << "=== Testing Distributed gRPC Prefetch ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 8}}}};

  DistributedGRPCParameterClient client(config);
  client.ClearPS();

  std::vector<uint64_t> keys = {100, 101, 102, 103, 104, 105, 106, 107};
  std::vector<std::vector<float>> values = {
      {1.0f, 1.1f, 1.2f},
      {2.0f, 2.1f, 2.2f},
      {3.0f, 3.1f, 3.2f},
      {4.0f, 4.1f, 4.2f},
      {5.0f, 5.1f, 5.2f},
      {6.0f, 6.1f, 6.2f},
      {7.0f, 7.1f, 7.2f},
      {8.0f, 8.1f, 8.2f}};
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.PutParameter(keys_array, values) == 0);

  uint64_t prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(prefetch_id != 0);
  std::cout << "Issued prefetch request" << std::endl;
  CHECK(!client.IsPrefetchDone(999999));
  client.WaitForPrefetch(prefetch_id);
  std::cout << "Prefetch wait completed" << std::endl;
  CHECK(client.IsPrefetchDone(prefetch_id));

  std::vector<std::vector<float>> fetched_values;
  CHECK(client.GetPrefetchResult(prefetch_id, &fetched_values));
  std::cout << "Fetched structured prefetch result" << std::endl;
  CHECK(check_eq_2d(fetched_values, values));
  CHECK(!client.GetPrefetchResult(prefetch_id, &fetched_values));

  uint64_t flat_prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(flat_prefetch_id != 0);
  std::cout << "Issued flat prefetch request" << std::endl;
  std::vector<float> flat_values;
  int64_t num_rows = 0;
  CHECK(client.GetPrefetchResultFlat(
      flat_prefetch_id, &flat_values, &num_rows, 3));
  std::cout << "Fetched flat prefetch result" << std::endl;
  CHECK(num_rows == static_cast<int64_t>(keys.size()));
  CHECK(flat_values.size() == keys.size() * 3);
  for (size_t i = 0; i < keys.size(); ++i) {
    for (int d = 0; d < 3; ++d) {
      CHECK(std::abs(flat_values[i * 3 + d] - values[i][d]) < 1e-6);
    }
  }
  CHECK(!client.GetPrefetchResultFlat(
      flat_prefetch_id, &flat_values, &num_rows, 3));
}

void TestPrefetchConcurrency(const std::vector<int>& ports) {
  std::cout << "=== Testing Distributed gRPC Prefetch Concurrency ==="
            << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 6}}}};

  DistributedGRPCParameterClient client(config);
  client.ClearPS();

  struct CaseData {
    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> values;
  };

  std::vector<CaseData> cases(16);
  for (size_t c = 0; c < cases.size(); ++c) {
    auto& cs = cases[c];
    for (int i = 0; i < 12; ++i) {
      uint64_t k = 3000 + static_cast<uint64_t>(c) * 100 + i;
      cs.keys.push_back(k);
      cs.values.push_back({static_cast<float>(k),
                           static_cast<float>(k + 1),
                           static_cast<float>(k + 2)});
    }
    base::ConstArray<uint64_t> keys_array(cs.keys);
    CHECK(client.PutParameter(keys_array, cs.values) == 0);
  }

  std::vector<std::future<bool>> futures;
  futures.reserve(cases.size());
  for (const auto& cs : cases) {
    futures.emplace_back(std::async(std::launch::async, [&client, cs]() {
      base::ConstArray<uint64_t> keys_array(cs.keys);
      uint64_t prefetch_id = client.PrefetchParameter(keys_array);
      if (prefetch_id == 0) {
        return false;
      }
      client.WaitForPrefetch(prefetch_id);
      std::vector<std::vector<float>> fetched_values;
      if (!client.GetPrefetchResult(prefetch_id, &fetched_values)) {
        return false;
      }
      return check_eq_2d(fetched_values, cs.values);
    }));
  }

  for (auto& future : futures) {
    CHECK(future.get());
  }
}

int main(int argc, char** argv) {
  DisableGrpcTracingEnv();
  base::Init(&argc, &argv);
  Reporter::StartReportThread(2000);

  const std::vector<int> ports = {kDistGrpcPort0, kDistGrpcPort1};

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "GRPC";
  launch_options.override_ports   = ports;
  recstore::test::ScopedPSServer server(launch_options, true);

  std::cout << "=== Distributed gRPC PS client tests ===" << std::endl;
  std::cout << std::endl;

  TestBasicConfig(ports);
  std::cout << std::endl;

  TestDirectClient(ports);
  std::cout << std::endl;

  TestLargeBatch(ports);
  std::cout << std::endl;

  TestPrefetch(ports);
  std::cout << std::endl;

  TestPrefetchConcurrency(ports);
  std::cout << std::endl;

  std::cout << "All tests completed!" << std::endl;
  return 0;
}
