#pragma once

#include <torch/torch.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace c10 {
using float16_t = c10::Half;
}

namespace base {

const int64_t EMBEDDING_DIMENSION_D = 128;

enum class DataType {
  UNKNOWN,
  UINT64,
  FLOAT32,
  FLOAT16,
  INT32,
  INT16,
  INT8,
};

inline std::string DataTypeToString(DataType dtype) {
  switch (dtype) {
    case DataType::UINT64:
      return "UINT64";
    case DataType::FLOAT32:
      return "FLOAT32";
    case DataType::FLOAT16:
      return "FLOAT16";
    case DataType::INT32:
      return "INT32";
    case DataType::INT16:
      return "INT16";
    case DataType::INT8:
      return "INT8";
    default:
      return "UNKNOWN";
  }
}

class RecTensor {
 public:
  RecTensor()
      : data_ptr_(nullptr), dtype_(DataType::UNKNOWN), num_elements_(0) {}

  template <typename T>
  RecTensor(T* data, const std::vector<int64_t>& shape)
      : data_ptr_(static_cast<void*>(data)), shape_(shape) {
    if (std::is_same<T, uint64_t>::value) {
      dtype_ = DataType::UINT64;
    } else if (std::is_same<T, float>::value) {
      dtype_ = DataType::FLOAT32;
    } else if (std::is_same<T, c10::float16_t>::value) {
      dtype_ = DataType::FLOAT16;
    } else if (std::is_same<T, int32_t>::value) {
      dtype_ = DataType::INT32;
    } else if (std::is_same<T, int16_t>::value) {
      dtype_ = DataType::INT16;
    } else if (std::is_same<T, int8_t>::value) {
      dtype_ = DataType::INT8;
    } else {
      dtype_ = DataType::UNKNOWN;
      throw std::runtime_error("Unsupported type for RecTensor constructor.");
    }
    recalculate_num_elements();
  }

  RecTensor(void* data, const std::vector<int64_t>& shape, DataType dtype)
      : data_ptr_(data), shape_(shape), dtype_(dtype) {
    recalculate_num_elements();
  }

  void* data() const { return data_ptr_; }

  template <typename T>
  T* data_as() const {
    if (dtype_ == DataType::UINT64 && !std::is_same<T, uint64_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is UINT64, accessed as different type.");
    }
    if (dtype_ == DataType::FLOAT32 && !std::is_same<T, float>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is FLOAT32, accessed as different type.");
    }
    if (dtype_ == DataType::FLOAT16 &&
        !std::is_same<T, c10::float16_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is FLOAT16, accessed as different type.");
    }
    if (dtype_ == DataType::INT32 && !std::is_same<T, int32_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT32, accessed as different type.");
    }
    if (dtype_ == DataType::INT16 && !std::is_same<T, int16_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT16, accessed as different type.");
    }
    if (dtype_ == DataType::INT8 && !std::is_same<T, int8_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT8, accessed as different type.");
    }
    return static_cast<T*>(data_ptr_);
  }

  const std::vector<int64_t>& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  size_t num_elements() const { return num_elements_; }
  size_t dim() const { return shape_.size(); }
  int64_t shape(size_t i) const {
    if (i >= shape_.size())
      throw std::out_of_range("Shape index out of range.");
    return shape_[i];
  }

  void set_data(void* data) { data_ptr_ = data; }
  void set_shape(const std::vector<int64_t>& new_shape) {
    shape_ = new_shape;
    recalculate_num_elements();
  }
  void set_dtype(DataType new_dtype) { dtype_ = new_dtype; }

  const std::vector<int64_t>& shape_as_vector() const { return shape_; }

 private:
  void recalculate_num_elements() {
    if (shape_.empty()) {
      num_elements_ = 0;
    } else {
      for (long long dim_size : shape_) {
        if (dim_size < 0)
          throw std::runtime_error("Tensor dimension size cannot be negative.");
      }
      num_elements_ = std::accumulate(shape_.begin(), shape_.end(), 1LL,
                                      std::multiplies<int64_t>());
    }
  }

  void* data_ptr_;
  std::vector<int64_t> shape_;
  DataType dtype_;
  size_t num_elements_;
};

inline DataType FromTorchDType(const at::ScalarType& dtype) {
  switch (dtype) {
    case at::kFloat:
      return DataType::FLOAT32;
    case at::kHalf:
      return DataType::FLOAT16;
    case at::kLong:
      return DataType::UINT64;
    case at::kInt:
      return DataType::INT32;
    case at::kShort:
      return DataType::INT16;
    case at::kChar:
      return DataType::INT8;
    default:
      return DataType::UNKNOWN;
  }
}

inline at::ScalarType ToTorchDType(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return at::kFloat;
    case DataType::FLOAT16:
      return at::kHalf;
    case DataType::UINT64:
      return at::kLong;
    case DataType::INT32:
      return at::kInt;
    case DataType::INT16:
      return at::kShort;
    case DataType::INT8:
      return at::kChar;
    default:
      throw std::runtime_error("Unsupported DataType for ToTorchDType");
  }
}

}  // namespace base