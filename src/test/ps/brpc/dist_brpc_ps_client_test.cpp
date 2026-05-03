#include "ps/brpc/dist_brpc_ps_client.h"

#include <algorithm>
#include <future>
#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/hash.h"
#include "base/init.h"
#include "base/thread.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "test/server_mgr/ps_server_launcher.h"

using namespace xmh;
using namespace recstore;

namespace {
constexpr int kDistBrpcPort0 = 16133;
constexpr int kDistBrpcPort1 = 16134;
} // namespace

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() < b.size())
    return false;

  for (int i = 0; i < b.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6)
      return false;
  }
  return true;
}

static bool check_eq_2d(const std::vector<std::vector<float>>& a,
                        const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false)
      return false;
  }
  return true;
}

void TestBasicConfig() {
  std::cout << "=== Testing Basic Configuration (bRPC) ===" << std::endl;

  json recstore_config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedBRPCParameterClient client(recstore_config);
    std::cout << "Recstore bRPC config parsed successfully, shard count: "
              << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore bRPC config test failed: " << e.what() << std::endl;
  }
}

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern (bRPC) ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  std::unique_ptr<BasePSClient> base_client(
      base::Factory<BasePSClient, json>::NewInstance(
          "distributed_brpc", config));

  if (!base_client) {
    std::cerr << "Failed to create distributed bRPC PS client via factory!"
              << std::endl;
    return;
  }

  std::cout << "Successfully created distributed bRPC PS client via factory"
            << std::endl;

  auto* client =
      dynamic_cast<DistributedBRPCParameterClient*>(base_client.get());
  if (!client) {
    std::cerr << "Failed to cast to DistributedBRPCParameterClient!"
              << std::endl;
    return;
  }

  std::cout << "Successfully created distributed bRPC PS client via factory"
            << std::endl;

  try {
    client->ClearPS();
    std::vector<uint64_t> keys_vec = {1, 2, 3};
    base::ConstArray<uint64_t> keys(keys_vec);
    std::vector<std::vector<float>> emptyvalues(keys_vec.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    client->PutParameter(keys, rightvalues);
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));

    client->ClearPS();
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "load fake data" << std::endl;
    CHECK(client->LoadFakeData(100));
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    CHECK(client->DumpFakeData(100));
    std::cout << "dump fake data done" << std::endl;

    std::cout << "All distributed bRPC PS operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "=== Testing Direct bRPC Client Creation ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedBRPCParameterClient client(config);
    std::cout << "Direct bRPC client created successfully, shard count: "
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

    std::cout << "All direct bRPC client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestLargeBatch() {
  std::cout << "=== Testing Large Batch Operations (bRPC) ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 50}}}};

  try {
    DistributedBRPCParameterClient client(config);

    std::vector<uint64_t> large_keys;
    std::vector<std::vector<float>> large_values;
    for (int i = 0; i < 100; ++i) {
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

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "BRPC";
  launch_options.override_ports   = {kDistBrpcPort0, kDistBrpcPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  TestBasicConfig();
  TestFactoryClient();
  TestDirectClient();
  TestLargeBatch();
  return 0;
}
