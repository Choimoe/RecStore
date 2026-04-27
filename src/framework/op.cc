#include "framework/op.h"
#include "framework/common/local_shm_op_component.h"
#include "framework/common/op_runtime_support.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "base/factory.h"
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <memory>
#include <numeric>
#include <thread>
#include <cstdlib>
#include <string>
#include <fstream>
#include "base/tensor.h"
#include <glog/logging.h>
#ifdef ENABLE_PERF_REPORT
#  include "base/report/report_client.h"
#endif

namespace recstore {

namespace {
bool IsReadWriteSuccess(BasePSClient* client, int ret) {
  if (dynamic_cast<RDMAPSClientAdapter*>(client) != nullptr ||
      dynamic_cast<LocalShmPSClient*>(client) != nullptr) {
    return ret == 0;
  }
  // Legacy GRPC/BRPC read/write methods return bool-like int values.
  return ret != 0;
}

} // namespace

void validate_keys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " +
                                base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional, but has " +
                                std::to_string(keys.dim()) + " dimensions.");
  }
}

void validate_embeddings(const base::RecTensor& embeddings,
                         const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " +
        base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " +
        std::to_string(embeddings.dim()) + " dimensions.");
  }
  // No fixed embedding dimension check for mock.
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const base::RecTensor& init_values) {
  EmbWrite(keys, init_values);
}

void KVClientOp::EmbDelete(const base::RecTensor& keys) {
  throw std::runtime_error("Not impl");
}
bool KVClientOp::EmbExists(const base::RecTensor& keys) {
  throw std::runtime_error("Not impl");
}

void KVClientOp::WaitForWrite(uint64_t write_id) {
  throw std::runtime_error("Not impl");
}
void KVClientOp::SaveToFile(const std::string& path) {
  throw std::runtime_error("Not impl");
}
void KVClientOp::LoadFromFile(const std::string& path) {
  throw std::runtime_error("Not impl");
}

uint64_t KVClientOp::EmbWriteAsync(const base::RecTensor& keys,
                                   const base::RecTensor& values) {
  throw std::runtime_error("Not impl");
}

std::shared_ptr<CommonOp> GetKVClientOp() {
  static std::shared_ptr<CommonOp> instance;
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    instance = std::make_shared<KVClientOp>();
  });
  return instance;
}

} // namespace recstore

#ifndef USE_FAKE_KVCLIENT

namespace recstore {

KVClientOp::KVClientOp() {
  if (!ps_client_) {
    try {
      json config   = GetGlobalConfig();
      bool use_rdma = false;
      try {
        use_rdma = ResolveFrameworkPSClientType(config) == PSClientType::kRdma;
      } catch (...) {
        use_rdma = false;
      }
      std::cerr << "[RDMA-DBG] KVClientOp ctor use_rdma="
                << (use_rdma ? "true" : "false") << std::endl;
      ps_backend_name_ = BackendNameFromConfig(config);

      if (use_rdma) {
        std::cerr << "[RDMA-DBG] InitializeRdmaProcessRuntime before "
                     "ConfigureLogging"
                  << std::endl;
        InitializeRdmaProcessRuntime();
        ConfigureLogging(false);
      } else {
        ConfigureLogging();
      }
      ps_client_holder_ = create_ps_client_from_config(config);
      ps_client_        = ps_client_holder_.get();

      LOG(INFO) << "PS client initialized successfully.";
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to initialize PS client: " << std::string(e.what());
      throw;
    }
  }
}

BasePSClient* KVClientOp::ps_client_ = nullptr;
std::unique_ptr<BasePSClient> KVClientOp::ps_client_holder_;

void KVClientOp::SetPSConfig(const std::string& host, int port) {
  ps_client_holder_.reset();
  ps_client_ = nullptr;

  json file_config = GetGlobalConfig();
  int final_port   = port;
  if (final_port <= 0) {
    if (file_config.contains("client") &&
        file_config["client"].contains("port")) {
      final_port = file_config["client"]["port"].get<int>();
    } else if (file_config.contains("cache_ps") &&
               file_config["cache_ps"].contains("servers") &&
               file_config["cache_ps"]["servers"].is_array() &&
               !file_config["cache_ps"]["servers"].empty()) {
      final_port = file_config["cache_ps"]["servers"][0]["port"].get<int>();
    } else {
      final_port = 15000;
    }
  }

  std::string final_host = host;
  if (final_host.empty()) {
    final_host = "127.0.0.1";
  }

  json config = file_config;
  if (!config.contains("client")) {
    config["client"] = json::object();
  }
  config["client"]["host"]  = final_host;
  config["client"]["port"]  = final_port;
  config["client"]["shard"] = 0;

  ps_client_holder_ = create_ps_client_from_config(config);
  ps_client_        = ps_client_holder_.get();
  ps_backend_name_  = BackendNameFromConfig(config);
  LOG(INFO) << "Re-initialized PS client with host=" << final_host
            << " port=" << final_port;
}

void KVClientOp::SetPSBackend(const std::string& backend) {
  if (backend.empty()) {
    throw std::invalid_argument("backend must be non-empty");
  }

  json config = GetGlobalConfig();
  if (!config.contains("cache_ps")) {
    config["cache_ps"] = json::object();
  }
  config["cache_ps"]["ps_type"] = NormalizePSType(backend);

  ps_client_holder_.reset();
  ps_client_        = nullptr;
  ps_client_holder_ = create_ps_client_from_config(config);
  ps_client_        = ps_client_holder_.get();
  ps_backend_name_  = BackendNameFromConfig(config);
  LOG(INFO) << "Re-initialized PS client with backend=" << ps_backend_name_;
}

std::string KVClientOp::CurrentPSBackend() const { return ps_backend_name_; }

void KVClientOp::LocalLookupFlat(const base::RecTensor& keys,
                                 base::RecTensor& values) {
  LocalShmLookupFlat(ps_client_, ps_backend_name_, keys, values);
}

int KVClientOp::SubmitLocalLookupFlat(const base::RecTensor& keys,
                                      int64_t embedding_dim,
                                      LocalShmFlatGetHandle* handle) {
  return SubmitLocalShmLookupFlat(
      ps_client_, ps_backend_name_, keys, embedding_dim, handle);
}

int KVClientOp::WaitLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  return WaitLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

void KVClientOp::ReleaseLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  ReleaseLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

bool KVClientOp::GetLocalLookupFlatPayloadRegion(const void** base,
                                                 std::size_t* bytes) {
  auto* local_client = GetLocalShmClientOrThrow(
      ps_client_, ps_backend_name_, "warmup_local_lookup_flat_cuda_region");
  return local_client->GetSlotPayloadRegion(base, bytes);
}

void KVClientOp::LocalUpdateFlat(const std::string& table_name,
                                 const base::RecTensor& keys,
                                 const base::RecTensor& grads) {
  LocalShmUpdateFlat(ps_client_, ps_backend_name_, table_name, keys, grads);
}

void KVClientOp::EmbRead(const RecTensor& keys, RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id =
      "op::EmbRead|" + std::to_string(static_cast<uint64_t>(start_us));
  std::string unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
#  endif

  LOG(INFO) << "EmbRead: keys.shape=" << keys.shape(0) << ", values.shape=["
            << values.shape(0) << ", " << values.shape(1) << "]";
  LOG(INFO) << "EmbRead: keys.data=" << keys.data_as<uint64_t>()
            << ", values.data=" << values.data_as<float>();
  if (keys.shape(0) > 0) {
    std::ostringstream oss;
    oss << "EmbRead: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.shape(0)); ++i)
      oss << keys.data_as<uint64_t>()[i] << ", ";
    LOG(INFO) << oss.str();
  }
  if (values.shape(0) > 0) {
    std::ostringstream oss;
    oss << "EmbRead: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.shape(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.shape(1)); ++j) {
        oss << values.data_as<float>()[i * values.shape(1) + j] << ", ";
      }
      oss << "] ";
    }
    LOG(INFO) << oss.str();
  }
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  float* values_data = values.data_as<float>();

  const int64_t D    = values.shape(1);
  const size_t total = static_cast<size_t>(L) * static_cast<size_t>(D);
  std::fill_n(values_data, total, 0.0f);

  int ret = ps_client_->GetParameter(keys_array, values_data);
  if (!IsReadWriteSuccess(ps_client_, ret)) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  std::string op_latency_key =
      "EmbRead|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "request_size",
         static_cast<double>(keys.shape(0)));

  FlameGraphData op_data = {
      "op::EmbRead",
      start_us,
      0, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), op_data);
#  endif
}

void KVClientOp::EmbUpdate(const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  EmbUpdate("default", keys, grads);
}

void KVClientOp::EmbUpdate(const std::string& table_name,
                           const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time         = std::chrono::high_resolution_clock::now();
  const uint64_t trace_id = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count());
  struct TraceGuard {
    explicit TraceGuard(uint64_t new_trace_id)
        : previous_trace_id_(recstore::g_trace_id) {
      recstore::g_trace_id = new_trace_id;
    }
    ~TraceGuard() { recstore::g_trace_id = previous_trace_id_; }
    uint64_t previous_trace_id_;
  } trace_guard(trace_id);
#  endif

  int64_t validate_done_us = 0;
  validate_keys(keys);
  validate_embeddings(grads, "Grads");

  const int64_t L = keys.shape(0);
  if (grads.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but grads has length " + std::to_string(grads.shape(0)));
  }

  const int64_t D = grads.shape(1);
  if (D <= 0) {
    throw std::invalid_argument(
        "Invalid grad dimension D: " + std::to_string(D));
  }

#  ifdef ENABLE_PERF_REPORT
  auto validate_done_time = std::chrono::high_resolution_clock::now();
  validate_done_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          validate_done_time - start_time)
          .count();
#  endif

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);

  const float* grads_data = grads.data_as<float>();
  int ret =
      ps_client_->UpdateParameterFlat(table_name, keys_array, grads_data, L, D);
  if (ret != 0) {
    throw std::runtime_error("Failed to update embeddings via PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbUpdate|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));

  std::string update_stage_id =
      "op_client::EmbUpdate|" + std::to_string(trace_id);
  report("embupdate_stages",
         update_stage_id.c_str(),
         "op_total_us",
         static_cast<double>(duration));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "op_validate_us",
         static_cast<double>(validate_done_us));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "request_size",
         static_cast<double>(L));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "embedding_dim",
         static_cast<double>(D));
#  endif
}

bool KVClientOp::InitEmbeddingTable(const std::string& table_name,
                                    const EmbeddingTableConfig& config) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#  endif
  int ret = ps_client_->InitEmbeddingTable(table_name, config);
#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "InitEmbeddingTable|" + std::to_string(static_cast<uint64_t>(start_us));
  // report(table_name, key, metric_name, value)
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));
#  endif
  return ret == 0;
}

void KVClientOp::EmbWrite(const RecTensor& keys, const RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#  endif

  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L          = keys.shape(0);
  const auto& values_shape = values.shape();
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const int64_t D = values.shape(1);

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  const float* values_data = values.data_as<float>();

  const int64_t total_values = L * D;
  if (values_shape[0] * values_shape[1] != total_values) {
    throw std::invalid_argument(
        "Values total elements mismatch: expected " +
        std::to_string(total_values) + ", but got " +
        std::to_string(values_shape[0] * values_shape[1]));
  }

  if (D <= 0) {
    throw std::invalid_argument(
        "Invalid embedding dimension D: " + std::to_string(D));
  }

  std::vector<std::vector<float>> values_vector;
  values_vector.reserve(L);
  for (int64_t i = 0; i < L; ++i) {
    std::vector<float> row(D);
    std::memcpy(row.data(), values_data + i * D, D * sizeof(float));
    asm volatile("" ::: "memory");
    _mm_mfence();
    values_vector.push_back(std::move(row));
  }

  LOG(INFO) << "=== Keys Array Info ===";
  LOG(INFO) << "Keys size: " << L;
  if (L > 0) {
    std::ostringstream keys_stream;
    keys_stream << "First 3 keys: ";
    for (int64_t i = 0; i < std::min(L, static_cast<int64_t>(3)); ++i) {
      keys_stream << keys_array[i] << " ";
    }
    LOG(INFO) << keys_stream.str();
  }

  LOG(INFO) << "=== Values Vector Info ===";
  LOG(INFO) << "Values total elements: " << total_values;
  LOG(INFO) << "Embedding dimension D: " << D;
  if (L > 0 && D > 0) {
    std::ostringstream values_stream;
    values_stream << "First 3 embeddings (each first 3 items): ";
    for (int64_t i = 0; i < std::min(L, static_cast<int64_t>(3)); ++i) {
      values_stream << "[";
      for (int64_t j = 0; j < std::min(D, static_cast<int64_t>(3)); ++j) {
        values_stream << values_vector[i][j] << " ";
      }
      values_stream << "] ";
    }
    LOG(INFO) << values_stream.str();
  }

  int ret = ps_client_->PutParameter(keys_array, values_vector);
  if (!IsReadWriteSuccess(ps_client_, ret)) {
    throw std::runtime_error("Failed to write embeddings to PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbWrite|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));
#  endif
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const InitStrategy& strategy) {
  validate_keys(keys);
}

uint64_t
KVClientOp::EmbPrefetch(const base::RecTensor& keys, const RecTensor& values) {
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  int64_t L                 = keys.shape(0);
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  return ps_client_->PrefetchParameter(keys_array);
}

bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
  return ps_client_->IsPrefetchDone(prefetch_id);
}

void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
  ps_client_->WaitForPrefetch(prefetch_id);
}

void KVClientOp::GetPretchResult(uint64_t prefetch_id,
                                 std::vector<std::vector<float>>* values) {
  ps_client_->GetPrefetchResult(prefetch_id, values);
}

void KVClientOp::GetPretchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  ps_client_->GetPrefetchResultFlat(
      prefetch_id, values, num_rows, embedding_dim);
}

bool KVClientOp::IsWriteDone(uint64_t write_id) {
  // return ps_client_->IsWriteDone(write_id);
  throw std::runtime_error("Not impl");
}

namespace testing {} // namespace testing

} // namespace recstore

#else

#  include "common/op_mock.cc"

#endif // USE_FAKE_KVCLIENT
