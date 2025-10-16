#include "ssd_extendible_hash.h"
#include <iostream>

ExtendibleHashSSD::ExtendibleHashSSD(const IndexConfig& config)
    : Index(config), hash_table_(nullptr) {
  LOG(INFO) << "SSD Index init";
  auto path_it = config.json_config_.find("path");
  if (path_it == config.json_config_.end()) {
    throw std::invalid_argument("IndexConfig missing 'path'");
  }
  std::string base_path = path_it->get<std::string>();
  std::string db_path   = base_path + "/cceh_index.db";
  filename_             = db_path;
  hash_table_           = new CCEH(); // 传入消息队列大小
}

ExtendibleHashSSD::~ExtendibleHashSSD() {
  LOG(INFO) << "delte ExtendibleHashSSD";
  if (hash_table_ != nullptr) {
    delete hash_table_;
    base::file_util::Delete(filename_, false);
    hash_table_ = nullptr;
  }
}

void ExtendibleHashSSD::Util() {
  // double util = hash_table_->Utilization();
  std::cout << "CCEH utilization " << std::endl;
}

void ExtendibleHashSSD::Put(const uint64_t key, uint64_t pointer, unsigned) {
  Key_t hash_key = key;
  Value_t value  = pointer;
  hash_table_->Insert(hash_key, value);
}

void ExtendibleHashSSD::Get(const uint64_t key, uint64_t& pointer, unsigned) {
  Key_t hash_key = key;
  Value_t result = hash_table_->Get(hash_key);
  if (result == NONE) {
    pointer = 0;
  } else {
    pointer = result;
  }
}

void ExtendibleHashSSD::BatchGet(
    base::ConstArray<uint64_t> keys, uint64_t* pointers, unsigned) {
  size_t n = keys.Size();
  for (size_t i = 0; i < n; ++i) {
    Key_t k     = keys[i];
    Value_t v   = hash_table_->Get(k);
    pointers[i] = (v == NONE) ? 0 : v;
  }
}

void ExtendibleHashSSD::BatchGet(
    coroutine<void>::push_type& sink,
    base::ConstArray<uint64_t> keys,
    uint64_t* pointers,
    unsigned tid) {
  if (pointers == nullptr || keys.Size() == 0) {
    LOG(FATAL) << "Invalid pointers array or empty keys";
  }

  const int batch_size = 32;
  size_t i             = 0;

  while (i < keys.Size()) {
    int batched_size = std::min(batch_size, static_cast<int>(keys.Size() - i));
    for (int j = 0; j < batched_size; ++j) {
      uint64_t key = keys[i + j];
      Get(key, pointers[i + j], tid);
    }
    i += batched_size;
    sink();
  }
}

void ExtendibleHashSSD::BatchPut(
    coroutine<void>::push_type& sink,
    base::ConstArray<uint64_t> keys,
    uint64_t* pointers,
    unsigned tid) {
  const int nr_batch_pages = 32; // 仿照 SSD 分批大小
  int i                    = 0;

  while (i < keys.Size()) {
    int batched_size =
        std::min(nr_batch_pages, static_cast<int>(keys.Size() - i));
    // 处理当前批次
    for (int j = 0; j < batched_size; ++j) {
      uint64_t key       = keys[i + j];
      uint64_t ptr_value = pointers[i + j];
      Put(key, ptr_value, tid);
    }
    i += batched_size;
    sink();
  }
}

bool ExtendibleHashSSD::Delete(uint64_t& key) {
  Key_t k = key;
  return hash_table_->Delete(k);
}

void ExtendibleHashSSD::BulkLoad(base::ConstArray<uint64_t> keys,
                                 const void* value) {
  size_t value_size = sizeof(Value_t);
  for (size_t i = 0; i < keys.Size(); ++i) {
    uint64_t ptr_value =
        *reinterpret_cast<const uint64_t*>(value + i * value_size);
    Put(keys[i], ptr_value, 0);
  }
}

void ExtendibleHashSSD::LoadFakeData(int64_t key_capacity, int value_size) {
  Index::LoadFakeData(key_capacity, sizeof(Value_t));
}

void ExtendibleHashSSD::clear() {
  delete hash_table_;
  hash_table_ = nullptr;
  base::file_util::Delete(filename_, false);
  // Recreate a new CCEH at the same path.
  hash_table_ = new CCEH(); // 传入消息队列大小
}

void ExtendibleHashSSD::DebugInfo() const {
  // TODO: implement debug info
}
