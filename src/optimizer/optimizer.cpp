#include "optimizer.h"
#include "ps/local_shm/local_shm_stage_report.h"
#include <cstring>

namespace {

std::vector<uint64_t> CollectReaderKeys(const ParameterCompressReader* reader) {
  const int size = reader->item_size();
  std::vector<uint64_t> keys;
  keys.reserve(size);
  for (int i = 0; i < size; ++i) {
    keys.push_back(reader->item(i)->key);
  }
  return keys;
}

void ValidateFlatUpdateArgs(const base::ConstArray<uint64_t>& keys,
                            const float* grads,
                            int64_t num_rows,
                            int64_t embedding_dim) {
  if (grads == nullptr) {
    throw std::runtime_error("UpdateFlat grads pointer is null");
  }
  if (num_rows < 0 || embedding_dim <= 0) {
    throw std::runtime_error("UpdateFlat invalid rows/dim");
  }
  if (keys.Size() != static_cast<size_t>(num_rows)) {
    throw std::runtime_error("UpdateFlat keys size mismatch");
  }
}

} // namespace

void SGD::Init(const std::vector<std::string> table_name,
               const EmbeddingTableConfig& config,
               BaseKV* base_kv) {
  LOG(INFO) << "SGD::Init called with " << table_name.size() << " table(s)";
  for (const auto& name : table_name) {
    LOG(INFO) << "  Initializing table: '" << name << "' with shape ["
              << config.num_embeddings << ", " << config.embedding_dim << "]";
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0; // PARAMETER tag
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;
  }
  LOG(INFO) << "SGD::Init completed. tensor_map_ now has " << tensor_map_.size()
            << " entries";
}

void SGD::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    throw std::runtime_error("Table not found: " + table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);

  std::vector<base::ConstArray<float>> current_values;
  it->second->BatchGet(keys, &current_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0) {
      // If key not found, we fallback to Put to initialize it
      std::vector<float> zero_init(item->dim, 0.0f);
      for (int j = 0; j < item->dim; ++j) {
        zero_init[j] = -learning_rate_ * item->data()[j];
      }
      std::string val_str(
          (char*)zero_init.data(), zero_init.size() * sizeof(float));
      it->second->Put(item->key, val_str, tid);
      continue;
    }

    float* data = const_cast<float*>(current_values[i].Data());
    int dim     = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      data[j] -= learning_rate_ * item->data()[j];
    }
  }
}

void SGD::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    throw std::runtime_error("Table not found: " + table);
  }

  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  const auto batch_get_start = std::chrono::steady_clock::now();
  std::vector<base::ConstArray<float>> current_values;
  it->second->BatchGet(key_vec, &current_values, tid);
  recstore::ReportLocalShmStageMetric(
      "sgd_update_batch_get_us", recstore::LocalShmElapsedUs(batch_get_start));

  const auto apply_start = std::chrono::steady_clock::now();
  int64_t missing_rows   = 0;
  for (int64_t row = 0; row < num_rows; ++row) {
    const float* row_grad = grads + row * embedding_dim;
    const auto& current   = current_values[static_cast<size_t>(row)];
    if (current.Size() == 0) {
      ++missing_rows;
      std::vector<float> zero_init(static_cast<size_t>(embedding_dim), 0.0f);
      for (int64_t col = 0; col < embedding_dim; ++col) {
        zero_init[static_cast<size_t>(col)] = -learning_rate_ * row_grad[col];
      }
      std::string val_str(reinterpret_cast<char*>(zero_init.data()),
                          zero_init.size() * sizeof(float));
      it->second->Put(keys[static_cast<size_t>(row)], val_str, tid);
      continue;
    }
    if (static_cast<int64_t>(current.Size()) != embedding_dim) {
      throw std::runtime_error(
          "SGD::UpdateFlat embedding_dim mismatch for table " + table);
    }

    float* data = const_cast<float*>(current.Data());
#pragma omp simd
    for (int64_t col = 0; col < embedding_dim; ++col) {
      data[col] -= learning_rate_ * row_grad[col];
    }
  }
  recstore::ReportLocalShmStageMetric(
      "sgd_update_apply_us", recstore::LocalShmElapsedUs(apply_start));
  recstore::ReportLocalShmStageMetric(
      "sgd_update_missing_rows", static_cast<double>(missing_rows));
}

void AdaGrad::Init(const std::vector<std::string> table_name,
                   const EmbeddingTableConfig& config,
                   BaseKV* base_kv) {
  for (const auto& name : table_name) {
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0;
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;

    std::string acc_table_name = name + "_accumulated_grad";
    SparseTensor* acc_tensor   = new SparseTensor();
    acc_tensor->init(
        const_cast<std::string&>(acc_table_name),
        MOMENT_1,
        tag,
        shape,
        base_kv);
    tensor_map_[acc_table_name] = acc_tensor;
  }
}

void AdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Accumulated gradient table not found: " + acc_table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0 || acc_values[i].Size() == 0) {
      // Fallback to sequential initialization if not found
      // (This is rare in training but kept for robustness)
      continue;
    }

    float* param_data = const_cast<float*>(current_values[i].Data());
    float* acc_data   = const_cast<float*>(acc_values[i].Data());
    int dim           = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      acc_data[j] += item->data()[j] * item->data()[j];
      float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[j]) + epsilon_);
      param_data[j] -= adaptive_lr * item->data()[j];
    }
  }
}

void AdaGrad::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Accumulated gradient table not found: " + acc_table);
  }

  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(key_vec, &current_values, tid);
  acc_it->second->BatchGet(key_vec, &acc_values, tid);

  for (int64_t row = 0; row < num_rows; ++row) {
    const auto& current = current_values[static_cast<size_t>(row)];
    const auto& acc     = acc_values[static_cast<size_t>(row)];
    if (current.Size() == 0 || acc.Size() == 0) {
      continue;
    }
    if (static_cast<int64_t>(current.Size()) != embedding_dim ||
        static_cast<int64_t>(acc.Size()) != embedding_dim) {
      throw std::runtime_error(
          "AdaGrad::UpdateFlat embedding_dim mismatch for table " + table);
    }

    const float* row_grad = grads + row * embedding_dim;
    float* param_data     = const_cast<float*>(current.Data());
    float* acc_data       = const_cast<float*>(acc.Data());
#pragma omp simd
    for (int64_t col = 0; col < embedding_dim; ++col) {
      acc_data[col] += row_grad[col] * row_grad[col];
      float adaptive_lr =
          learning_rate_ / (std::sqrt(acc_data[col]) + epsilon_);
      param_data[col] -= adaptive_lr * row_grad[col];
    }
  }
}

void RowWiseAdaGrad::Init(const std::vector<std::string> table_name,
                          const EmbeddingTableConfig& config,
                          BaseKV* base_kv) {
  for (const auto& name : table_name) {
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0;
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;

    std::string acc_table_name      = name + "_rowwise_accumulated_grad";
    SparseTensor* acc_tensor        = new SparseTensor();
    std::vector<uint64_t> acc_shape = {
        config.num_embeddings, 1}; // One value per row
    acc_tensor->init(
        const_cast<std::string&>(acc_table_name),
        MOMENT_1,
        tag,
        acc_shape,
        base_kv);
    tensor_map_[acc_table_name] = acc_tensor;
  }
}

void RowWiseAdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_rowwise_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Row-wise accumulated gradient table not found: " + acc_table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0 || acc_values[i].Size() == 0) {
      continue;
    }

    float* param_data = const_cast<float*>(current_values[i].Data());
    float* acc_data   = const_cast<float*>(acc_values[i].Data());
    int dim           = std::min(current_values[i].Size(), item->dim);

    float grad_square_mean = 0.0;
#pragma omp simd reduction(+ : grad_square_mean)
    for (int j = 0; j < dim; ++j) {
      grad_square_mean += item->data()[j] * item->data()[j];
    }
    grad_square_mean /= dim;

    acc_data[0] += grad_square_mean;

    float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[0]) + epsilon_);
#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      param_data[j] -= adaptive_lr * item->data()[j];
    }
  }
}

void RowWiseAdaGrad::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_rowwise_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Row-wise accumulated gradient table not found: " + acc_table);
  }

  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(key_vec, &current_values, tid);
  acc_it->second->BatchGet(key_vec, &acc_values, tid);

  for (int64_t row = 0; row < num_rows; ++row) {
    const auto& current = current_values[static_cast<size_t>(row)];
    const auto& acc     = acc_values[static_cast<size_t>(row)];
    if (current.Size() == 0 || acc.Size() == 0) {
      continue;
    }
    if (static_cast<int64_t>(current.Size()) != embedding_dim ||
        acc.Size() != 1) {
      throw std::runtime_error(
          "RowWiseAdaGrad::UpdateFlat embedding_dim mismatch for table " +
          table);
    }

    const float* row_grad = grads + row * embedding_dim;
    float* param_data     = const_cast<float*>(current.Data());
    float* acc_data       = const_cast<float*>(acc.Data());

    float grad_square_mean = 0.0f;
#pragma omp simd reduction(+ : grad_square_mean)
    for (int64_t col = 0; col < embedding_dim; ++col) {
      grad_square_mean += row_grad[col] * row_grad[col];
    }
    grad_square_mean /= static_cast<float>(embedding_dim);

    acc_data[0] += grad_square_mean;
    float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[0]) + epsilon_);
#pragma omp simd
    for (int64_t col = 0; col < embedding_dim; ++col) {
      param_data[col] -= adaptive_lr * row_grad[col];
    }
  }
}
