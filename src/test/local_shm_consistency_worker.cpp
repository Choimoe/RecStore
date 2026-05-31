#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/json.h"
#include "ps/local_shm/local_shm_client.h"

DEFINE_string(config_path, "", "Path to local_shm config JSON");
DEFINE_string(mode,
              "update",
              "init|update|verify|init_uniform|update_uniform|read_stress");
DEFINE_string(table_name, "consistency", "Embedding table name");
DEFINE_int32(worker_id, 0, "Worker id used to derive gradient values");
DEFINE_int32(worker_count, 1, "Number of update workers");
DEFINE_int32(iterations, 50, "Update iterations per worker");
DEFINE_int32(rows, 3, "Number of shared rows to update");
DEFINE_int32(embedding_dim, 4, "Embedding dimension");
DEFINE_int32(read_iterations, 1000, "Read iterations for read_stress mode");
DEFINE_double(tolerance, 1e-4, "Absolute verification tolerance");

namespace {

std::vector<uint64_t> MakeKeys() {
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(FLAGS_rows));
  for (int row = 0; row < FLAGS_rows; ++row) {
    keys.push_back(static_cast<uint64_t>(1001 + row));
  }
  return keys;
}

std::vector<std::vector<float>> MakeInitialValues() {
  std::vector<std::vector<float>> values;
  values.reserve(static_cast<std::size_t>(FLAGS_rows));
  for (int row = 0; row < FLAGS_rows; ++row) {
    std::vector<float> row_values;
    row_values.reserve(static_cast<std::size_t>(FLAGS_embedding_dim));
    for (int col = 0; col < FLAGS_embedding_dim; ++col) {
      row_values.push_back(static_cast<float>((row + 1) * 10 + col));
    }
    values.push_back(std::move(row_values));
  }
  return values;
}

std::vector<std::vector<float>> MakeUniformInitialValues() {
  std::vector<std::vector<float>> values;
  values.reserve(static_cast<std::size_t>(FLAGS_rows));
  for (int row = 0; row < FLAGS_rows; ++row) {
    values.emplace_back(static_cast<std::size_t>(FLAGS_embedding_dim),
                        static_cast<float>(1000 + row));
  }
  return values;
}

std::vector<float> MakeGradients(int worker_id) {
  std::vector<float> grads(
      static_cast<std::size_t>(FLAGS_rows) *
          static_cast<std::size_t>(FLAGS_embedding_dim),
      0.0f);
  for (int row = 0; row < FLAGS_rows; ++row) {
    for (int col = 0; col < FLAGS_embedding_dim; ++col) {
      grads[static_cast<std::size_t>(row) * FLAGS_embedding_dim + col] =
          static_cast<float>((worker_id + 1) * (row + 1) * (col + 1));
    }
  }
  return grads;
}

std::vector<float> MakeUniformGradients() {
  return std::vector<float>(
      static_cast<std::size_t>(FLAGS_rows) *
          static_cast<std::size_t>(FLAGS_embedding_dim),
      1.0f);
}

recstore::LocalShmPSClient MakeClient() {
  std::ifstream config_file(FLAGS_config_path);
  if (!config_file.good()) {
    throw std::runtime_error(
        "failed to open config_path: " + FLAGS_config_path);
  }
  json config;
  config_file >> config;
  return recstore::LocalShmPSClient(config["local_shm"]);
}

int RunInit() {
  auto client = MakeClient();
  if (client.InitEmbeddingTable(
          FLAGS_table_name,
          recstore::EmbeddingTableConfig{
              static_cast<uint64_t>(FLAGS_rows),
              static_cast<uint64_t>(FLAGS_embedding_dim)}) != 0) {
    std::cerr << "InitEmbeddingTable failed\n";
    return 1;
  }
  const auto keys   = MakeKeys();
  const auto values = MakeInitialValues();
  if (client.PutParameter(base::ConstArray<uint64_t>(keys), values) != 0) {
    std::cerr << "PutParameter failed\n";
    return 1;
  }
  return 0;
}

int RunInitUniform() {
  auto client = MakeClient();
  if (client.InitEmbeddingTable(
          FLAGS_table_name,
          recstore::EmbeddingTableConfig{
              static_cast<uint64_t>(FLAGS_rows),
              static_cast<uint64_t>(FLAGS_embedding_dim)}) != 0) {
    std::cerr << "InitEmbeddingTable failed\n";
    return 1;
  }
  const auto keys   = MakeKeys();
  const auto values = MakeUniformInitialValues();
  if (client.PutParameter(base::ConstArray<uint64_t>(keys), values) != 0) {
    std::cerr << "PutParameter failed\n";
    return 1;
  }
  return 0;
}

int RunUpdate() {
  auto client          = MakeClient();
  const auto keys      = MakeKeys();
  const auto grads     = MakeGradients(FLAGS_worker_id);
  const auto key_array = base::ConstArray<uint64_t>(keys);
  for (int iter = 0; iter < FLAGS_iterations; ++iter) {
    if (client.UpdateParameterFlat(
            FLAGS_table_name,
            key_array,
            grads.data(),
            FLAGS_rows,
            FLAGS_embedding_dim) != 0) {
      std::cerr << "UpdateParameterFlat failed at iter=" << iter << "\n";
      return 1;
    }
  }
  return 0;
}

int RunUpdateUniform() {
  auto client          = MakeClient();
  const auto keys      = MakeKeys();
  const auto grads     = MakeUniformGradients();
  const auto key_array = base::ConstArray<uint64_t>(keys);
  for (int iter = 0; iter < FLAGS_iterations; ++iter) {
    if (client.UpdateParameterFlat(
            FLAGS_table_name,
            key_array,
            grads.data(),
            FLAGS_rows,
            FLAGS_embedding_dim) != 0) {
      std::cerr << "UpdateParameterFlat failed at iter=" << iter << "\n";
      return 1;
    }
  }
  return 0;
}

bool CheckUniformReadback(const std::vector<float>& readback, int iteration) {
  constexpr float kLearningRate = 0.01f;
  for (int row = 0; row < FLAGS_rows; ++row) {
    const float initial = static_cast<float>(1000 + row);
    float min_expected  = initial;
    for (int i = 0; i < FLAGS_worker_count * FLAGS_iterations; ++i) {
      min_expected -= kLearningRate;
    }
    min_expected -= static_cast<float>(FLAGS_tolerance);
    const float first =
        readback[static_cast<std::size_t>(row) * FLAGS_embedding_dim];
    if (!std::isfinite(first) || first > initial + FLAGS_tolerance ||
        first < min_expected) {
      std::cerr << "read_stress value out of range iteration=" << iteration
                << " row=" << row << " actual=" << first << " expected_range=["
                << min_expected << ", " << initial << "]\n";
      return false;
    }
    for (int col = 1; col < FLAGS_embedding_dim; ++col) {
      const float actual =
          readback[static_cast<std::size_t>(row) * FLAGS_embedding_dim + col];
      if (!std::isfinite(actual) ||
          std::fabs(actual - first) > FLAGS_tolerance) {
        std::cerr << "read_stress inconsistent row iteration=" << iteration
                  << " row=" << row << " col=" << col << " first=" << first
                  << " actual=" << actual << "\n";
        return false;
      }
    }
  }
  return true;
}

int RunReadStress() {
  auto client     = MakeClient();
  const auto keys = MakeKeys();
  std::vector<float> readback(
      static_cast<std::size_t>(FLAGS_rows) *
          static_cast<std::size_t>(FLAGS_embedding_dim),
      0.0f);
  for (int iter = 0; iter < FLAGS_read_iterations; ++iter) {
    if (client.GetParameterFlat(
            base::ConstArray<uint64_t>(keys),
            readback.data(),
            FLAGS_rows,
            FLAGS_embedding_dim) != 0) {
      std::cerr << "GetParameterFlat failed at iter=" << iter << "\n";
      return 1;
    }
    if (!CheckUniformReadback(readback, iter)) {
      return 1;
    }
  }
  return 0;
}

int RunVerify() {
  auto client               = MakeClient();
  const auto keys           = MakeKeys();
  const auto initial_values = MakeInitialValues();
  std::vector<float> readback(
      static_cast<std::size_t>(FLAGS_rows) *
          static_cast<std::size_t>(FLAGS_embedding_dim),
      0.0f);
  if (client.GetParameterFlat(
          base::ConstArray<uint64_t>(keys),
          readback.data(),
          FLAGS_rows,
          FLAGS_embedding_dim) != 0) {
    std::cerr << "GetParameterFlat failed\n";
    return 1;
  }

  constexpr float kLearningRate = 0.01f;
  const float worker_factor_sum =
      static_cast<float>(FLAGS_worker_count * (FLAGS_worker_count + 1) / 2);
  for (int row = 0; row < FLAGS_rows; ++row) {
    for (int col = 0; col < FLAGS_embedding_dim; ++col) {
      const float total_grad =
          static_cast<float>(FLAGS_iterations) * worker_factor_sum *
          static_cast<float>((row + 1) * (col + 1));
      const float expected =
          initial_values[row][col] - kLearningRate * total_grad;
      const float actual =
          readback[static_cast<std::size_t>(row) * FLAGS_embedding_dim + col];
      if (std::fabs(actual - expected) > FLAGS_tolerance) {
        std::cerr << "value mismatch row=" << row << " col=" << col
                  << " actual=" << actual << " expected=" << expected << "\n";
        return 1;
      }
    }
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  if (FLAGS_config_path.empty()) {
    std::cerr << "--config_path is required\n";
    return 1;
  }
  if (FLAGS_mode == "init") {
    return RunInit();
  }
  if (FLAGS_mode == "init_uniform") {
    return RunInitUniform();
  }
  if (FLAGS_mode == "update") {
    return RunUpdate();
  }
  if (FLAGS_mode == "update_uniform") {
    return RunUpdateUniform();
  }
  if (FLAGS_mode == "read_stress") {
    return RunReadStress();
  }
  if (FLAGS_mode == "verify") {
    return RunVerify();
  }
  std::cerr << "unsupported --mode: " << FLAGS_mode << "\n";
  return 1;
}
