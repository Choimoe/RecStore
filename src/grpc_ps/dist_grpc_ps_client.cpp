#include "dist_grpc_ps_client.h"

#include <algorithm>
#include <future>
#include <thread>
#include <fstream>

#include "base/factory.h"
#include "base/log.h"
#include "base/timer.h"

FACTORY_REGISTER(recstore::BasePSClient, distributed_grpc, recstore::DistributedGRPCParameterClient, json);

namespace recstore {

DistributedGRPCParameterClient::DistributedGRPCParameterClient(json config) : BasePSClient(config) {
  json client_config;
  
  if (config.contains("distributed_client")) {
    LOG(INFO) << "Detected recstore config format, extracting distributed_client section";
    client_config = config["distributed_client"];
  }
  else {
    LOG(FATAL) << "Invalid config format. Expected either recstore config with 'distributed_client' section "
               << "or direct client config with 'servers' and 'num_shards' fields";
  }

  // 验证必要字段
  if (!client_config.contains("servers") || !client_config["servers"].is_array()) {
    LOG(FATAL) << "Missing or invalid 'servers' field in distributed client config";
  }

  if (!client_config.contains("num_shards") || !client_config["num_shards"].is_number_integer()) {
    LOG(FATAL) << "Missing or invalid 'num_shards' field in distributed client config";
  }

  num_shards_           = client_config["num_shards"].get<int>();
  max_keys_per_request_ = client_config.value("max_keys_per_request", 500);
  hash_method_          = client_config.value("hash_method", "city_hash");

  // 解析服务器配置
  auto servers = client_config["servers"];
  server_configs_.reserve(servers.size());

  for (size_t i = 0; i < servers.size(); ++i) {
    const auto& server = servers[i];
    if (!server.contains("host") || !server.contains("port") || !server.contains("shard")) {
      LOG(FATAL) << "Server config " << i << " missing required fields (host, port, shard)";
    }

    ServerConfig cfg;
    cfg.host  = server["host"].get<std::string>();
    cfg.port  = server["port"].get<int>();
    cfg.shard = server["shard"].get<int>();

    server_configs_.push_back(cfg);
    shard_to_client_index_[cfg.shard] = i;
  }

  if (server_configs_.size() != static_cast<size_t>(num_shards_)) {
    LOG(WARNING) << "Number of servers (" << server_configs_.size() << ") doesn't match num_shards (" << num_shards_
                 << ")";
  }

  partitioned_key_buffer_.resize(num_shards_);
  key_index_mapping_.resize(num_shards_);

  InitializeClients();

  LOG(INFO) << "Initialized DistributedGRPCParameterClient with " << num_shards_
            << " shards, hash method: " << hash_method_;
}

DistributedGRPCParameterClient::~DistributedGRPCParameterClient() {}

void DistributedGRPCParameterClient::InitializeClients() {
  clients_.clear();
  clients_.reserve(server_configs_.size());

  for (const auto& server_config : server_configs_) {
    // 为每个服务器创建独立的配置
    json client_config = {{"host", server_config.host}, {"port", server_config.port}, {"shard", server_config.shard}};

    auto client = std::make_unique<GRPCParameterClient>(client_config);
    clients_.push_back(std::move(client));

    LOG(INFO) << "Created gRPC client for shard " << server_config.shard << " at " << server_config.host << ":"
              << server_config.port;
  }
}

int DistributedGRPCParameterClient::GetShardId(uint64_t key) const {
  if (hash_method_ == "city_hash") {
    return GetHash(key) % num_shards_;
  } else if (hash_method_ == "simple_mod") {
    return key % num_shards_;
  } else {
    LOG(ERROR) << "Unknown hash method: " << hash_method_ << ", using city_hash";
    return GetHash(key) % num_shards_;
  }
}

void DistributedGRPCParameterClient::PartitionKeys(const base::ConstArray<uint64_t>& keys,
                                                   std::vector<std::vector<uint64_t>>& partitioned_keys) const {

  for (auto& partition : partitioned_key_buffer_) {
    partition.clear();
  }
  for (auto& mapping : key_index_mapping_) {
    mapping.clear();
  }


  for (size_t i = 0; i < keys.Size(); ++i) {
    uint64_t key = keys[i];
    int shard_id = GetShardId(key);

    partitioned_key_buffer_[shard_id].push_back(key);
    key_index_mapping_[shard_id].push_back(i);


    if (partitioned_key_buffer_[shard_id].size() > static_cast<size_t>(max_keys_per_request_)) {
      partitioned_key_buffer_[shard_id].resize(max_keys_per_request_);
      key_index_mapping_[shard_id].resize(max_keys_per_request_);
      LOG(WARNING) << "Truncated keys for shard " << shard_id << " to " << max_keys_per_request_;
    }
  }


  partitioned_keys = partitioned_key_buffer_;
}

bool DistributedGRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                                  std::vector<std::vector<float>>* values) {
  if (keys.Size() == 0) {
    values->clear();
    return true;
  }

  xmh::Timer timer("DistributedGRPCParameterClient::GetParameter");


  std::vector<std::vector<uint64_t>> partitioned_keys;
  PartitionKeys(keys, partitioned_keys);


  std::vector<std::future<bool>> futures;
  std::vector<std::vector<std::vector<float>>> partitioned_results(num_shards_);

  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (partitioned_keys[shard_id].empty()) {
      continue;
    }

    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      return false;
    }

    int client_index = it->second;
    auto* client = clients_[client_index].get();

    // 异步请求
    futures.push_back(std::async(std::launch::async, [=, &partitioned_results]() {
      base::ConstArray<uint64_t> shard_keys(partitioned_keys[shard_id]);
      return client->GetParameter(shard_keys, &partitioned_results[shard_id]);
    }));
  }

  // 3. 等待所有请求完成
  for (auto& future : futures) {
    if (!future.get()) {
      LOG(ERROR) << "Failed to get parameters from one of the shards";
      return false;
    }
  }

  // 4. 合并结果
  MergeResults(keys, partitioned_results, values);

  return true;
}

void DistributedGRPCParameterClient::MergeResults(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<std::vector<float>>>& partitioned_results,
    std::vector<std::vector<float>>* values) const {
  values->clear();
  values->resize(keys.Size());

  // 重建key -> index映射
  std::unordered_map<uint64_t, size_t> key_to_result_index;
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      if (i < partitioned_results[shard_id].size()) {
        (*values)[original_index] = partitioned_results[shard_id][i];
      }
    }
  }
}

void DistributedGRPCParameterClient::MergeResultsToArray(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<std::vector<float>>>& partitioned_results,
    float* values) const {
  int emb_dim = 0;
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (!partitioned_results[shard_id].empty() && !partitioned_results[shard_id][0].empty()) {
      emb_dim = partitioned_results[shard_id][0].size();
      break;
    }
  }

  if (emb_dim == 0) {
    LOG(WARNING) << "No valid embeddings found";
    return;
  }

  // 合并结果到连续内存
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      if (i < partitioned_results[shard_id].size()) {
        const auto& embedding = partitioned_results[shard_id][i];
        std::copy(embedding.begin(), embedding.end(), values + original_index * emb_dim);
      }
    }
  }
}

// 实现BasePSClient接口
int DistributedGRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys, float* values) {
  std::vector<std::vector<float>> result_vectors;
  bool success = GetParameter(keys, &result_vectors);

  if (!success) {
    return -1;
  }

  // 将vector结果复制到连续内存
  MergeResultsToArray(keys, {{result_vectors}}, values);
  return 0;
}

int DistributedGRPCParameterClient::AsyncGetParameter(const base::ConstArray<uint64_t>& keys, float* values) {
  return GetParameter(keys, values);
}

int DistributedGRPCParameterClient::PutParameter(const base::ConstArray<uint64_t>& keys,
                                                 const std::vector<std::vector<float>>& values) {
  if (keys.Size() != values.size()) {
    LOG(ERROR) << "Keys and values size mismatch: " << keys.Size() << " vs " << values.size();
    return -1;
  }

  std::vector<std::vector<uint64_t>> partitioned_keys;
  PartitionKeys(keys, partitioned_keys);


  std::vector<std::vector<std::vector<float>>> partitioned_values(num_shards_);
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      partitioned_values[shard_id].push_back(values[original_index]);
    }
  }

  // 并发put到各个分片
  std::vector<std::future<bool>> futures;

  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (partitioned_keys[shard_id].empty()) {
      continue;
    }

    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      return -1;
    }

    int client_index = it->second;
    auto* client = clients_[client_index].get();

    futures.push_back(std::async(std::launch::async, [=, &partitioned_keys, &partitioned_values]() {
      base::ConstArray<uint64_t> shard_keys(partitioned_keys[shard_id]);
      return client->PutParameter(shard_keys, partitioned_values[shard_id]) == 0;
    }));
  }

  // 等待所有请求完成
  for (auto& future : futures) {
    if (!future.get()) {
      LOG(ERROR) << "Failed to put parameters to one of the shards";
      return -1;
    }
  }

  return 0;
}

void DistributedGRPCParameterClient::Command(PSCommand command) {

  std::vector<std::future<void>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(std::launch::async, [&client, command]() { client->Command(command); }));
  }


  for (auto& future : futures) {
    future.wait();
  }
}

bool DistributedGRPCParameterClient::ClearPS() {
  std::vector<std::future<bool>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(std::launch::async, [&client]() { return client->ClearPS(); }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

bool DistributedGRPCParameterClient::LoadCkpt(const std::vector<std::string>& model_config_path,
                                              const std::vector<std::string>& emb_file_path) {
  std::vector<std::future<bool>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(std::launch::async, [&client, &model_config_path, &emb_file_path]() {
      return client->LoadCkpt(model_config_path, emb_file_path);
    }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

} // namespace recstore
