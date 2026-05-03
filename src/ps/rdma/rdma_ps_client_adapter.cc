#include "ps/rdma/rdma_ps_client_adapter.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <gflags/gflags.h>

#include "base/init.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/base/config.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_string(rdma_transport_mode);

namespace recstore {

namespace {
constexpr float kRdmaUpdateLearningRate = 0.01f;

std::vector<std::string> ReadProcessArgv() {
  std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
  std::vector<std::string> argv;
  if (!cmdline.is_open()) {
    return argv;
  }

  std::string current;
  char ch = '\0';
  while (cmdline.get(ch)) {
    if (ch == '\0') {
      if (!current.empty()) {
        argv.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    argv.push_back(current);
  }
  return argv;
}
} // namespace

void InitializeRdmaProcessRuntime() {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    std::vector<std::string> argv_strings = ReadProcessArgv();
    if (argv_strings.empty()) {
      argv_strings.emplace_back("recstore_rdma_client");
    }

    std::vector<char*> argv_storage;
    argv_storage.reserve(argv_strings.size() + 1);
    for (auto& arg : argv_strings) {
      argv_storage.push_back(arg.data());
    }
    argv_storage.push_back(nullptr);

    int argc    = static_cast<int>(argv_strings.size());
    char** argv = argv_storage.data();
    base::Init(&argc, &argv);
  });
}

RDMAPSClientAdapter::RDMAPSClientAdapter(json config)
    : BasePSClient(config), config_(std::move(config)) {}

void RDMAPSClientAdapter::EnsureClientInitialized() {
  std::lock_guard<std::mutex> guard(init_mu_);
  if (initialized_) {
    return;
  }

  const json cache_ps_cfg =
      config_.contains("cache_ps") ? config_["cache_ps"] : json::object();
  const json client_cfg =
      config_.contains("client") ? config_["client"] : json::object();
  const json dist_cfg = ResolveFrameworkDistributedClientConfig(config_);

  const int num_shards       = dist_cfg.value("num_shards", 1);
  FLAGS_num_server_processes = num_shards;
  FLAGS_num_client_processes = 1;
  FLAGS_global_id            = num_shards;
  FLAGS_value_size =
      cache_ps_cfg.contains("base_kv_config")
          ? cache_ps_cfg["base_kv_config"].value("value_size", FLAGS_value_size)
          : FLAGS_value_size;
  FLAGS_max_kv_num_per_request =
      dist_cfg.value("max_keys_per_request", FLAGS_max_kv_num_per_request);
  if (const char* mode = std::getenv("RECSTORE_RDMA_TRANSPORT_MODE")) {
    FLAGS_rdma_transport_mode = mode;
  }

  if (num_shards <= 1) {
    shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
        client_cfg.value("host", std::string("127.0.0.1")),
        client_cfg.value("port", 25000),
        client_cfg.value("shard", 0)));
    client_ = shard_clients_.front().get();
  } else {
    const auto servers_it = dist_cfg.find("servers");
    if (servers_it == dist_cfg.end() || !servers_it->is_array() ||
        servers_it->empty()) {
      throw std::runtime_error(
          "RDMA distributed_client.servers must be provided for multi-shard "
          "configuration");
    }

    std::vector<BaseParameterClient*> raw_clients;
    raw_clients.reserve(servers_it->size());
    CHECK_EQ(static_cast<int>(servers_it->size()), num_shards)
        << "RDMA distributed_client.servers size must equal num_shards";
    for (std::size_t shard_idx = 0; shard_idx < servers_it->size();
         ++shard_idx) {
      const auto& server = (*servers_it)[shard_idx];
      const int shard    = server.value("shard", static_cast<int>(shard_idx));
      shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
          server.value("host", std::string("127.0.0.1")),
          server.value("port", 25000),
          shard));
      raw_clients.push_back(shard_clients_.back().get());
    }
    multi_client_ = std::make_unique<AllShardsParameterClientWrapper>(
        raw_clients, num_shards);
    client_ = multi_client_.get();
  }

  initialized_ = true;
}

void RDMAPSClientAdapter::EnsureThreadInitialized() {
  EnsureClientInitialized();
  const std::thread::id tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> guard(thread_init_mu_);
  if (initialized_threads_.find(tid) != initialized_threads_.end()) {
    return;
  }

  if (multi_client_ != nullptr) {
    multi_client_->InitThread();
  } else if (client_ != nullptr) {
    client_->InitThread();
  }

  initialized_threads_.insert(tid);
}

void RDMAPSClientAdapter::EnsureTableReady(const std::string& table_name,
                                           int64_t embedding_dim) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = tables_.find(table_name);
  if (it == tables_.end()) {
    throw std::runtime_error("RDMA table is not initialized: " + table_name);
  }
  if (static_cast<int64_t>(it->second.config.embedding_dim) != embedding_dim) {
    throw std::runtime_error(
        "RDMA embedding dimension mismatch for table " + table_name);
  }
}

int64_t RDMAPSClientAdapter::DefaultEmbeddingDimOrThrow() const {
  if (tables_.empty()) {
    throw std::runtime_error(
        "RDMA table metadata is empty; call InitEmbeddingTable first");
  }
  return static_cast<int64_t>(tables_.begin()->second.config.embedding_dim);
}

RDMAPSClientAdapter::PrefetchState
RDMAPSClientAdapter::GetPrefetchState(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = prefetches_.find(prefetch_id);
  if (it == prefetches_.end()) {
    throw std::runtime_error(
        "Unknown RDMA prefetch id: " + std::to_string(prefetch_id));
  }
  return it->second;
}

void RDMAPSClientAdapter::MarkPrefetchConsumed(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  prefetches_.erase(prefetch_id);
}

int RDMAPSClientAdapter::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      float* values) {
  EnsureThreadInitialized();
  if (client_ == nullptr) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  std::vector<float> heap_recv;
  float* recv = nullptr;
  if (multi_client_ != nullptr) {
    heap_recv.resize(response_bytes / sizeof(float), 0.0f);
    recv = heap_recv.data();
  } else {
    recv = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
    std::memset(recv, 0, response_bytes);
  }

  int rpc_id = client_->GetParameter(keys, recv, false, 0);
  client_->RevokeRPCResource(rpc_id);

  const auto* status_word = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(recv) +
      keys.Size() * static_cast<std::size_t>(FLAGS_value_size));
  if (*status_word != 0) {
    return -1;
  }

  std::memcpy(
      values, recv, keys.Size() * static_cast<std::size_t>(FLAGS_value_size));
  return 0;
}

int RDMAPSClientAdapter::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  EnsureThreadInitialized();
  if (client_ == nullptr) {
    return -1;
  }
  return client_->PutParameter(keys.ToVector(), values);
}

int RDMAPSClientAdapter::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (grads->empty()) {
    return 0;
  }

  const int64_t embedding_dim = static_cast<int64_t>(grads->front().size());
  std::vector<float> flat;
  flat.reserve(grads->size() * static_cast<std::size_t>(embedding_dim));
  for (const auto& row : *grads) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return UpdateParameterFlat(
      table_name,
      keys,
      flat.data(),
      static_cast<int64_t>(grads->size()),
      embedding_dim);
}

int RDMAPSClientAdapter::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  EnsureTableReady(table_name, embedding_dim);
  if (num_rows == 0) {
    return 0;
  }

  std::vector<float> current(
      static_cast<std::size_t>(num_rows) *
          static_cast<std::size_t>(embedding_dim),
      0.0f);
  if (GetParameter(keys, current.data()) != 0) {
    return -1;
  }

  std::vector<std::vector<float>> updated;
  updated.reserve(static_cast<std::size_t>(num_rows));
  for (int64_t row = 0; row < num_rows; ++row) {
    std::vector<float> values(static_cast<std::size_t>(embedding_dim), 0.0f);
    for (int64_t col = 0; col < embedding_dim; ++col) {
      const std::size_t idx =
          static_cast<std::size_t>(row * embedding_dim + col);
      values[static_cast<std::size_t>(col)] =
          current[idx] - (kRdmaUpdateLearningRate * grads[idx]);
    }
    updated.push_back(std::move(values));
  }

  return PutParameter(keys, updated);
}

int RDMAPSClientAdapter::InitEmbeddingTable(
    const std::string& table_name, const EmbeddingTableConfig& config) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto [it, inserted] = tables_.emplace(table_name, TableState{config});
  if (!inserted) {
    if (it->second.config.embedding_dim != config.embedding_dim ||
        it->second.config.num_embeddings != config.num_embeddings) {
      return -1;
    }
  }
  return 0;
}

int RDMAPSClientAdapter::AsyncGetParameter(const base::ConstArray<uint64_t>&,
                                           float*) {
  throw std::runtime_error(
      "RDMA adapter AsyncGetParameter not implemented yet");
}

void RDMAPSClientAdapter::Command(PSCommand) {
  throw std::runtime_error("RDMA adapter Command not implemented yet");
}

uint64_t
RDMAPSClientAdapter::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
  EnsureThreadInitialized();
  if (client_ == nullptr) {
    throw std::runtime_error("RDMA adapter has no initialized client");
  }
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA prefetch requires at least one key");
  }

  const int64_t embedding_dim = DefaultEmbeddingDimOrThrow();
  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  auto* buffer = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
  std::memset(buffer, 0, response_bytes);

  const int rpc_id = client_->GetParameter(keys, buffer, true, 0);

  std::lock_guard<std::mutex> guard(state_mu_);
  const uint64_t prefetch_id = next_prefetch_id_++;
  prefetches_.emplace(
      prefetch_id,
      PrefetchState{
          buffer,
          rpc_id,
          static_cast<int64_t>(keys.Size()),
          embedding_dim,
      });
  return prefetch_id;
}

bool RDMAPSClientAdapter::IsPrefetchDone(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  if (client_ == nullptr) {
    return false;
  }
  const PrefetchState state = GetPrefetchState(prefetch_id);
  return client_->QueryRPCFinished(state.rpc_id);
}

void RDMAPSClientAdapter::WaitForPrefetch(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  if (client_ == nullptr) {
    throw std::runtime_error("RDMA adapter has no initialized client");
  }
  const PrefetchState state = GetPrefetchState(prefetch_id);
  try {
    client_->WaitRPCFinish(state.rpc_id);
  } catch (...) {
    client_->RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    throw;
  }
}

bool RDMAPSClientAdapter::GetPrefetchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  if (values == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  std::vector<float> flat;
  int64_t num_rows = 0;
  if (!GetPrefetchResultFlat(
          prefetch_id, &flat, &num_rows, state.embedding_dim)) {
    return false;
  }

  values->clear();
  values->reserve(static_cast<std::size_t>(num_rows));
  for (int64_t row = 0; row < num_rows; ++row) {
    const auto begin =
        flat.begin() + row * static_cast<int64_t>(state.embedding_dim);
    values->emplace_back(
        begin, begin + static_cast<int64_t>(state.embedding_dim));
  }
  return true;
}

bool RDMAPSClientAdapter::GetPrefetchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  if (values == nullptr || num_rows == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  if (embedding_dim != state.embedding_dim) {
    return false;
  }

  WaitForPrefetch(prefetch_id);
  const auto* status_word = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(state.buffer) +
      static_cast<std::size_t>(state.key_count) *
          static_cast<std::size_t>(FLAGS_value_size));
  if (*status_word != 0) {
    client_->RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    return false;
  }

  const std::size_t value_count =
      static_cast<std::size_t>(state.key_count) *
      static_cast<std::size_t>(state.embedding_dim);
  values->assign(state.buffer, state.buffer + value_count);
  *num_rows = state.key_count;
  client_->RevokeRPCResource(state.rpc_id);
  MarkPrefetchConsumed(prefetch_id);
  return true;
}

} // namespace recstore
