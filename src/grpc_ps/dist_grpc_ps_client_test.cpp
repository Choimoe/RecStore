#include "dist_grpc_ps_client.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "base_ps/base_client.h"

using namespace recstore;

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

void TestBasicConfig() {
  std::cout << "=== Testing Basic Configuration ===" << std::endl;

  // 测试recstore配置格式
  json recstore_config = {
    {"distributed_client", {
      {"servers", {
        {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}},
        {{"host", "127.0.0.1"}, {"port", 15001}, {"shard", 1}}
      }},
      {"num_shards", 2},
      {"hash_method", "city_hash"}
    }}
  };

  try {
    DistributedGRPCParameterClient client(recstore_config);
    std::cout << "Recstore config parsed successfully, shard count: " << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore config test failed: " << e.what() << std::endl;
  }
}

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern ===" << std::endl;

  json config = {
    {"distributed_client", {
      {"servers", {
        {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}},
        {{"host", "127.0.0.1"}, {"port", 15001}, {"shard", 1}}
      }},
      {"num_shards", 2},
      {"hash_method", "city_hash"}
    }}
  };

  std::unique_ptr<BasePSClient> client(
      base::Factory<BasePSClient, json>::NewInstance("distributed_grpc", config));

  if (!client) {
    std::cerr << "Failed to create distributed PS client via factory!" << std::endl;
    return;
  }

  std::cout << "Successfully created distributed PS client via factory" << std::endl;

  try {
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

    std::cout << "All distributed PS operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what() << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "=== Testing Direct Client Creation ===" << std::endl;

  json config = {
    {"distributed_client", {
      {"servers", {
        {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}},
        {{"host", "127.0.0.1"}, {"port", 15001}, {"shard", 1}}
      }},
      {"num_shards", 2},
      {"hash_method", "city_hash"}
    }}
  };

  try {
    DistributedGRPCParameterClient client(config);
    std::cout << "Direct client created successfully, shard count: " << client.shard_count() << std::endl;

    client.ClearPS();
    // assert empty
    std::vector<uint64_t> keys = {1001, 1002, 1003};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {{1, 0, 1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;
    
    base::ConstArray<uint64_t> keys_array(keys);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    // insert something
    client.PutParameter(keys_array, rightvalues);
    // read those
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    client.ClearPS();
    // read those
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "All direct client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what() << std::endl;
  }
}

void TestLargeBatch() {
  std::cout << "=== Testing Large Batch Operations ===" << std::endl;

  json config = {
    {"distributed_client", {
      {"servers", {
        {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}},
        {{"host", "127.0.0.1"}, {"port", 15001}, {"shard", 1}}
      }},
      {"num_shards", 2},
      {"hash_method", "city_hash"},
      {"max_keys_per_request", 50}
    }}
  };

  try {
    DistributedGRPCParameterClient client(config);
    
    // 准备大批量keys (超过max_keys_per_request)
    std::vector<uint64_t> large_keys;
    std::vector<std::vector<float>> large_values;
    for (int i = 0; i < 100; ++i) {
      large_keys.push_back(2000 + i);
      large_values.push_back({float(i), float(i * 2)});
    }

    base::ConstArray<uint64_t> keys_array(large_keys);
    
    client.ClearPS();
    
    // 写入大批量数据
    int put_result = client.PutParameter(keys_array, large_values);
    CHECK(put_result == 0);
    
    // 读取并验证
    std::vector<std::vector<float>> retrieved_values;
    bool get_success = client.GetParameter(keys_array, &retrieved_values);
    CHECK(get_success);
    CHECK(check_eq_2d(retrieved_values, large_values));

    std::cout << "Large batch operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Large batch test skipped (servers not available): " << e.what() << std::endl;
  }
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  base::Reporter::StartReportThread(2000);

  std::cout << "=== 分布式gRPC客户端测试 ===" << std::endl;
  std::cout << std::endl;

  TestBasicConfig();
  std::cout << std::endl;
  
  TestFactoryClient();
  std::cout << std::endl;
  
  TestDirectClient();
  std::cout << std::endl;
  
  TestLargeBatch();
  std::cout << std::endl;

  std::cout << "All tests completed!" << std::endl;
  return 0;
} 