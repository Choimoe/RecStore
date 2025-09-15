#include "framework/op.h"
#include "grpc_ps/grpc_ps_client.h"
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

// Assuming InitStrategyType is defined in base/tensor.h
#include "base/tensor.h"
#include "op.h"

namespace recstore {

// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
static int get_log_level() {
  static int level = []() {
    const char* env = std::getenv("RECSTORE_LOG_LEVEL");
    if (!env)
      return 2; // Default INFO
    return std::atoi(env);
  }();
  return level;
}
#define RECSTORE_LOG(level, msg)                                               \
  do {                                                                         \
    if (get_log_level() >= level) {                                            \
      std::cout << msg << std::endl;                                           \
    }                                                                          \
  } while (0)

json load_config_from_file(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + config_path);
  }

  std::string content(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  try {
    return json::parse(content);
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "Failed to parse config file: " + std::string(e.what()));
  }
}

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

#  include "grpc_ps/grpc_ps_client.h"
namespace recstore {

BasePSClient* create_ps_client_from_config(const json& config) {
  json client_config;
  if (config.contains("client")) {
    client_config = config["client"];
  } else {
    client_config = json{{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}};
  }

  return new GRPCParameterClient(client_config);
}

KVClientOp::KVClientOp() {
  if (!ps_client_) {
    try {
      auto current_path = std::filesystem::current_path();
      RECSTORE_LOG(
          2, "[INFO] Current working directory: " + current_path.string());

      std::filesystem::path config_path;
      bool config_found = false;

      for (auto p = current_path; p.has_parent_path(); p = p.parent_path()) {
        if (std::filesystem::exists(p / "recstore_config.json")) {
          config_path  = p / "recstore_config.json";
          config_found = true;
          RECSTORE_LOG(2,
                       "[INFO] Found config file at: " + config_path.string());
          break;
        }
      }

      if (!config_found) {
        throw std::runtime_error(
            "Could not find 'recstore_config.json' in current or any parent "
            "directory starting from: " +
            current_path.string());
      }

      std::ifstream test_file(config_path);
      if (!test_file.good()) {
        throw std::runtime_error(
            "Config file not found: " + config_path.string() +
            ". Please ensure recstore_config.json exists "
            "in the project root directory.");
      }
      test_file.close();

      json config = load_config_from_file(config_path);

      ps_client_ = create_ps_client_from_config(config);

      RECSTORE_LOG(2,
                   "[INFO] PS client initialized successfully from config: " +
                       config_path.string());
    } catch (const std::exception& e) {
      RECSTORE_LOG(
          0,
          "[ERROR] Failed to initialize PS client: " + std::string(e.what()));
      throw;
    }
  }
}

BasePSClient* KVClientOp::ps_client_ = nullptr;

void KVClientOp::EmbRead(const RecTensor& keys, RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

  RECSTORE_LOG(0,
               "[DEBUG][op.cc] EmbRead: keys.shape="
                   << keys.shape(0) << ", values.shape=[" << values.shape(0)
                   << ", " << values.shape(1) << "]");
  RECSTORE_LOG(0,
               "[DEBUG][op.cc] EmbRead: keys.data="
                   << keys.data_as<uint64_t>()
                   << ", values.data=" << values.data_as<float>());
  if (keys.shape(0) > 0) {
    std::ostringstream oss;
    oss << "[DEBUG][op.cc] EmbRead: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.shape(0)); ++i)
      oss << keys.data_as<uint64_t>()[i] << ", ";
    RECSTORE_LOG(0, oss.str());
  }
  if (values.shape(0) > 0) {
    std::ostringstream oss;
    oss << "[DEBUG][op.cc] EmbRead: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.shape(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.shape(1)); ++j) {
        oss << values.data_as<float>()[i * values.shape(1) + j] << ", ";
      }
      oss << "] ";
    }
    RECSTORE_LOG(0, oss.str());
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

  const int64_t D = values.shape(1);
  const size_t total = static_cast<size_t>(L) * static_cast<size_t>(D);
  std::fill_n(values_data, total, 0.0f);

  
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  bool success = ps_client_->GetParameter(keys_array, values_data);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

void KVClientOp::EmbUpdate(const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  throw std::runtime_error("Not impl");
}

void KVClientOp::EmbWrite(const RecTensor& keys, const RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

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
  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

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

  RECSTORE_LOG(2, "=== Keys Array Info ===");
  RECSTORE_LOG(2, "Keys size: " << L);
  if (L > 0) {
    std::ostringstream keys_stream;
    keys_stream << "First 3 keys: ";
    for (int64_t i = 0; i < std::min(L, static_cast<int64_t>(3)); ++i) {
      keys_stream << keys_array[i] << " ";
    }
    RECSTORE_LOG(2, keys_stream.str());
  }

  RECSTORE_LOG(2, "=== Values Vector Info ===");
  RECSTORE_LOG(2, "Values total elements: " << total_values);
  RECSTORE_LOG(2, "Embedding dimension D: " << D);
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
    RECSTORE_LOG(2, values_stream.str());
  }

  bool success = ps_client_->PutParameter(keys_array, values_vector);
  if (!success) {
    throw std::runtime_error("Failed to write embeddings to PS client.");
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
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

bool KVClientOp::IsWriteDone(uint64_t write_id) {
  // return ps_client_->IsWriteDone(write_id);
  throw std::runtime_error("Not impl");
}

namespace testing {

// void ClearEmbeddingTableForTesting() {
//     bool success = GetGRPCClientInstance().ClearPS();
//     if (!success) {
//         throw std::runtime_error("Failed to clear remote Parameter Server state during testing.");
//     }
// }

} // namespace testing

} // namespace recstore
#else
namespace recstore {

KVClientOp::KVClientOp() : embedding_dim_(-1), learning_rate_(0.01f) {
  std::cout << "KVClientOp: Initialized MOCK (in-memory) backend." << std::endl;
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const InitStrategy& strategy) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (embedding_dim_ == -1) {
    throw std::runtime_error("KVClientOp Error: Must call EmbWrite or set "
                             "dimension before using InitStrategy.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const int64_t num_keys   = keys.shape(0);
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    store_[key]  = std::vector<float>(embedding_dim_, 0.0f);
  }
}

void KVClientOp::EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);
  const int64_t emb_dim  = values.shape(1);
  const int64_t num_keys = keys.shape(0);
  RECSTORE_LOG(2,
               "[INFO] [MOCK] EmbRead called: num_keys="
                   << num_keys << ", emb_dim=" << emb_dim
                   << ", keys ptr=" << keys.data_as<uint64_t>()
                   << ", values ptr=" << values.data_as<float>());
  if (num_keys == 0) {
    RECSTORE_LOG(3, "[DEBUG] [MOCK] EmbRead: num_keys==0, early return");
    return;
  }
  if (embedding_dim_ != -1 && embedding_dim_ != emb_dim) {
    RECSTORE_LOG(0,
                 "[ERROR] [MOCK] EmbRead: embedding_dim mismatch: "
                     << embedding_dim_ << " vs " << emb_dim);
    throw std::runtime_error(
        "KVClientOp Error: Inconsistent embedding dimension for read.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  float* value_data        = values.data_as<float>();
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    auto it      = store_.find(key);
    if (it == store_.end()) {
      std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
      RECSTORE_LOG(3,
                   "[DEBUG] [MOCK] EmbRead: key="
                       << key << " not found, filled with zeros");
    } else {
      if (it->second.size() != emb_dim) {
        RECSTORE_LOG(0,
                     "[ERROR] [MOCK] EmbRead: stored dim mismatch for key="
                         << key << ", stored=" << it->second.size()
                         << ", requested=" << emb_dim);
        throw std::runtime_error(
            "KVClientOp FATAL: Dimension mismatch for key " +
            std::to_string(key) +
            ". Stored dim: " + std::to_string(it->second.size()) +
            ", Requested dim: " + std::to_string(emb_dim));
      }
      std::copy(it->second.begin(), it->second.end(), value_data + i * emb_dim);
      RECSTORE_LOG(
          3, "[DEBUG] [MOCK] EmbRead: key=" << key << " read OK, values=[...]");
    }
  }
}

void KVClientOp::EmbWrite(const base::RecTensor& keys,
                          const base::RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);
  const int64_t emb_dim  = values.shape(1);
  const int64_t num_keys = keys.shape(0);
  RECSTORE_LOG(2,
               "[INFO] [MOCK] EmbWrite called: num_keys="
                   << num_keys << ", emb_dim=" << emb_dim
                   << ", keys ptr=" << keys.data_as<uint64_t>()
                   << ", values ptr=" << values.data_as<float>());
  if (embedding_dim_ == -1) {
    embedding_dim_ = emb_dim;
    RECSTORE_LOG(1,
                 "[WARNING] [MOCK] EmbWrite: embedding_dim_ inferred as "
                     << embedding_dim_);
    std::cout << "KVClientOp: Inferred and set embedding dimension to "
              << embedding_dim_ << std::endl;
  } else if (embedding_dim_ != emb_dim) {
    RECSTORE_LOG(0,
                 "[ERROR] [MOCK] EmbWrite: embedding_dim mismatch: "
                     << embedding_dim_ << " vs " << emb_dim);
    throw std::runtime_error(
        "KVClientOp Error: Inconsistent embedding dimension for write. "
        "Expected " +
        std::to_string(embedding_dim_) + ", but got " +
        std::to_string(emb_dim));
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* value_data  = values.data_as<float>();
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key       = key_data[i];
    const float* start = value_data + i * emb_dim;
    const float* end   = start + emb_dim;
    store_[key].assign(start, end);
    RECSTORE_LOG(
        3, "[DEBUG] [MOCK] EmbWrite: key=" << key << " written, values=[...]");
  }
}

void KVClientOp::EmbUpdate(const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  // std::lock_guard<std::mutex> lock(mtx_);
  // const int64_t emb_dim  = grads.shape(1);
  // const int64_t num_keys = keys.shape(0);
  // RECSTORE_LOG(2,
  //              "[INFO] [MOCK] EmbUpdate called: num_keys="
  //                  << num_keys << ", emb_dim=" << emb_dim
  //                  << ", keys ptr=" << keys.data_as<uint64_t>()
  //                  << ", grads ptr=" << grads.data_as<float>());
  // if (embedding_dim_ == -1) {
  //   embedding_dim_ = emb_dim;
  //   RECSTORE_LOG(1,
  //                "[WARNING] [MOCK] EmbUpdate: embedding_dim_ inferred as "
  //                    << embedding_dim_);
  // } else if (embedding_dim_ != emb_dim) {
  //   RECSTORE_LOG(0,
  //                "[ERROR] [MOCK] EmbUpdate: embedding_dim mismatch: "
  //                    << embedding_dim_ << " vs " << emb_dim);
  //   throw std::runtime_error(
  //       "KVClientOp Error: Inconsistent embedding dimension for update.");
  // }
  // const uint64_t* key_data = keys.data_as<uint64_t>();
  // const float* grad_data   = grads.data_as<float>();
  // for (int64_t i = 0; i < num_keys; ++i) {
  //   uint64_t key = key_data[i];
  //   auto it      = store_.find(key);
  //   if (it != store_.end()) {
  //     for (int64_t j = 0; j < emb_dim; ++j) {
  //       it->second[j] -= learning_rate_ * grad_data[i * emb_dim + j];
  //     }
  //     RECSTORE_LOG(
  //         3,
  //         "[DEBUG] [MOCK] EmbUpdate: key=" << key << " updated, grads=[...]");
  //   }
  // }
}

void KVClientOp::GetPretchResult(uint64_t prefetch_id,
                                 std::vector<std::vector<float>>* values) {
  throw std::runtime_error("Not impl");
}
uint64_t KVClientOp::EmbPrefetch(const base::RecTensor& keys,
                                 const base::RecTensor& values) {
  throw std::runtime_error("Not impl");
}
void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
  throw std::runtime_error("Not impl");
}
bool KVClientOp::IsWriteDone(uint64_t write_id) {
  throw std::runtime_error("Not impl");
}
bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
  throw std::runtime_error("Not impl");
}

} // namespace recstore
#endif