#include <folly/executors/CPUThreadPoolExecutor.h>

#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "base_ps/base_client.h"
#include "grpc_ps_client.h"

static bool check_eq_1d(const std::vector<float> &a,
                        const std::vector<float> &b) {
  if (a.size() != b.size()) return false;

  for (int i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6) return false;
  }
  return true;
}

static bool check_eq_2d(const std::vector<std::vector<float>> &a,
                        const std::vector<std::vector<float>> &b) {
  if (a.size() != b.size()) return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false) return false;
  }
  return true;
}


void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern ===" << std::endl;


  json config = {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 1}};

  std::unique_ptr<recstore::BasePSClient> client(
      base::Factory<recstore::BasePSClient, json>::NewInstance("grpc", config));

  if (!client) {
    std::cerr << "Failed to create PS client via factory!" << std::endl;
    return;
  }

  std::cout << "Successfully created PS client via factory" << std::endl;

  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(1, 200LL * 1e6);

  // while (1) {
  //   int perf_count = 500;
  //   std::vector<uint64_t> keys(perf_count);
  //   for (int i = 0; i < perf_count; i++) {
  //     keys[i] = distrib(gen);
  //   }
  //   std::vector<std::vector<float>> values;
  //   xmh::Timer timer_client("client get");
  //   ConstArray<uint64_t> keys_array(keys);
  //   client.GetParameter(keys_array, &values);
  //   timer_client.end();
  // }

  client->ClearPS();
  // assert empty
  std::vector<uint64_t> keys = {1, 2, 3};
  std::vector<std::vector<float>> emptyvalues(keys.size());
  std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
  std::vector<std::vector<float>> values;
  client->GetParameter(keys, &values);
  CHECK(check_eq_2d(values, emptyvalues));

  // insert something
  client->PutParameter(keys, rightvalues);
  // read those
  client->GetParameter(keys, &values);
  CHECK(check_eq_2d(values, rightvalues));

  // clear all
  client->ClearPS();
  // read those
  client->GetParameter(keys, &values);
  CHECK(check_eq_2d(values, emptyvalues));
  
}

int main(int argc, char **argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  TestFactoryClient();

  std::cout << "\n=== Testing Original Implementation ===" << std::endl;

  // GRPCParameterClient client("127.0.0.1", 15000, 1);
  // std::random_device
  //     rd;  // Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  // std::uniform_int_distribution<> distrib(1, 200LL * 1e6);

  // // while (1) {
  // //   int perf_count = 500;
  // //   std::vector<uint64_t> keys(perf_count);
  // //   for (int i = 0; i < perf_count; i++) {
  // //     keys[i] = distrib(gen);
  // //   }
  // //   std::vector<std::vector<float>> values;
  // //   xmh::Timer timer_client("client get");
  // //   ConstArray<uint64_t> keys_array(keys);
  // //   client.GetParameter(keys_array, &values);
  // //   timer_client.end();
  // // }

  // client.ClearPS();
  // // assert empty
  // std::vector<uint64_t> keys = {1, 2, 3};
  // std::vector<std::vector<float>> emptyvalues(keys.size());
  // std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
  // std::vector<std::vector<float>> values;
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, emptyvalues));

  // // insert something
  // client.PutParameter(keys, rightvalues);
  // // read those
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, rightvalues));

  // // clear all
  // client.ClearPS();
  // // read those
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, emptyvalues));

  return 0;
}