#include "framework/hugectr/op_hugectr.h"

#include "framework/hugectr/hierkv_backend.h"
#include "framework/op.h"
#include "base/tensor.h"

#include <cuda_runtime_api.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace recstore {
void EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
  auto op = GetKVClientOp();
  if (op == nullptr) {
    throw std::runtime_error("KV client op is not initialized.");
  }
  op->EmbRead(keys, values);
}

void EmbUpdate(const base::RecTensor& keys, const base::RecTensor& grads) {
  auto op = GetKVClientOp();
  if (op == nullptr) {
    throw std::runtime_error("KV client op is not initialized.");
  }
  op->EmbUpdate(keys, grads);
}
} // namespace recstore

static void check_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "CUDA Error in op_hugectr.cc: " + std::string(cudaGetErrorString(err)));
  }
}

namespace recstore {
namespace framework {

namespace {

void ValidateReadTensors(const HugeCTR::Tensor2<long long>& keys,
                         const HugeCTR::Tensor2<float>& values) {
  const auto& keys_dims   = keys.get_dimensions();
  const auto& values_dims = values.get_dimensions();

  if (keys_dims.size() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional.");
  }
  if (values_dims.size() != 2) {
    throw std::invalid_argument("Values tensor must be 2-dimensional.");
  }
  if (keys_dims[0] != values_dims[0]) {
    throw std::invalid_argument(
        "Keys and Values tensors must have the same size in dimension 0.");
  }
  if (values_dims[1] != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument(
        "Values tensor has incorrect embedding dimension.");
  }
}

void ValidateUpdateTensors(const HugeCTR::Tensor2<long long>& keys,
                           const HugeCTR::Tensor2<float>& grads) {
  const auto& keys_dims  = keys.get_dimensions();
  const auto& grads_dims = grads.get_dimensions();

  if (keys_dims.size() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional.");
  }
  if (grads_dims.size() != 2) {
    throw std::invalid_argument("Grads tensor must be 2-dimensional.");
  }
  if (keys_dims[0] != grads_dims[0]) {
    throw std::invalid_argument(
        "Keys and Grads tensors must have the same size in dimension 0.");
  }
  if (grads_dims[1] != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument(
        "Grads tensor has incorrect embedding dimension.");
  }
}

class HugeCTRRecStoreBackend : public HugeCTRBackend {
public:
  void Read(const HugeCTR::Tensor2<long long>& keys,
            HugeCTR::Tensor2<float>& values) override {
    ValidateReadTensors(keys, values);

    const auto& keys_dims   = keys.get_dimensions();
    const auto& values_dims = values.get_dimensions();
    const int64_t L         = static_cast<int64_t>(keys_dims[0]);
    const int64_t D         = static_cast<int64_t>(values_dims[1]);

    std::vector<long long> h_keys(static_cast<size_t>(L));
    std::vector<float> h_values(static_cast<size_t>(L * D));

    check_cuda_error(cudaMemcpy(
        h_keys.data(),
        keys.get_ptr(),
        static_cast<size_t>(L) * sizeof(long long),
        cudaMemcpyDeviceToHost));

    base::RecTensor rec_keys(h_keys.data(), {L}, base::DataType::UINT64);
    base::RecTensor rec_values(
        h_values.data(), {L, D}, base::DataType::FLOAT32);

    try {
      recstore::EmbRead(rec_keys, rec_values);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "Recstore EmbRead failed during HugeCTR operation: " +
          std::string(e.what()));
    }

    check_cuda_error(cudaMemcpy(
        values.get_ptr(),
        h_values.data(),
        static_cast<size_t>(L * D) * sizeof(float),
        cudaMemcpyHostToDevice));
  }

  void Update(const HugeCTR::Tensor2<long long>& keys,
              const HugeCTR::Tensor2<float>& grads) override {
    ValidateUpdateTensors(keys, grads);

    const auto& keys_dims  = keys.get_dimensions();
    const auto& grads_dims = grads.get_dimensions();
    const int64_t L        = static_cast<int64_t>(keys_dims[0]);
    const int64_t D        = static_cast<int64_t>(grads_dims[1]);

    std::vector<long long> h_keys(static_cast<size_t>(L));
    std::vector<float> h_grads(static_cast<size_t>(L * D));

    check_cuda_error(cudaMemcpy(
        h_keys.data(),
        keys.get_ptr(),
        static_cast<size_t>(L) * sizeof(long long),
        cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(
        h_grads.data(),
        grads.get_ptr(),
        static_cast<size_t>(L * D) * sizeof(float),
        cudaMemcpyDeviceToHost));

    base::RecTensor rec_keys(h_keys.data(), {L}, base::DataType::UINT64);
    base::RecTensor rec_grads(h_grads.data(), {L, D}, base::DataType::FLOAT32);

    try {
      recstore::EmbUpdate(rec_keys, rec_grads);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "Recstore EmbUpdate failed during HugeCTR operation: " +
          std::string(e.what()));
    }
  }
};

std::filesystem::path FindRecStoreConfigPath() {
  auto current_path = std::filesystem::current_path();
  for (auto p = current_path; p.has_parent_path(); p = p.parent_path()) {
    const auto candidate = p / "recstore_config.json";
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error(
      "Could not find 'recstore_config.json' in current or any parent "
      "directory.");
}

json LoadHugeCTRConfig() {
  return ParseFile2Json(FindRecStoreConfigPath().string());
}

std::unique_ptr<HugeCTRBackend> CreateHugeCTRBackend(const json& config) {
  switch (ParseHugeCTRBackendKind(config)) {
  case HugeCTRBackendKind::RecStore:
    return std::make_unique<HugeCTRRecStoreBackend>();
  case HugeCTRBackendKind::HierKV:
    return std::make_unique<HugeCTRHierKVBackend>(config);
  }
  throw std::invalid_argument("Unsupported HugeCTR backend kind.");
}

std::unique_ptr<HugeCTRBackend> CreateHugeCTRBackendFromRuntimeConfig() {
  return CreateHugeCTRBackend(LoadHugeCTRConfig());
}

} // namespace

HugeCTRBackendKind ParseHugeCTRBackendKind(const json& config) {
  if (!config.contains("hugectr")) {
    return HugeCTRBackendKind::RecStore;
  }
  const json& hugectr = config["hugectr"];
  if (!hugectr.is_object() || !hugectr.contains("backend")) {
    return HugeCTRBackendKind::RecStore;
  }
  const std::string backend = hugectr["backend"].get<std::string>();
  if (backend == "recstore") {
    return HugeCTRBackendKind::RecStore;
  }
  if (backend == "hierkv") {
    return HugeCTRBackendKind::HierKV;
  }
  throw std::invalid_argument("Unsupported hugectr.backend value: " + backend);
}

void emb_read_hugectr(const HugeCTR::Tensor2<long long>& keys,
                      HugeCTR::Tensor2<float>& values) {
  auto backend = CreateHugeCTRBackendFromRuntimeConfig();
  backend->Read(keys, values);
}

void emb_update_hugectr(const HugeCTR::Tensor2<long long>& keys,
                        const HugeCTR::Tensor2<float>& grads) {
  auto backend = CreateHugeCTRBackendFromRuntimeConfig();
  backend->Update(keys, grads);
}

} // namespace framework
} // namespace recstore
