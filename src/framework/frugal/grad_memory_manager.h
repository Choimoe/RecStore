#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <gperftools/profiler.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "base/log.h"

namespace recstore {

// class GradTensor {
//  public:
//   GradTensor() {}

//   GradTensor(const torch::Tensor &tensor) : tensor_(tensor) {}

//   torch::Tensor tensor() { return tensor_; }

//   void set_tensor(const torch::Tensor &tensor) { tensor_ = tensor; }

//  private:
//   torch::Tensor tensor_;
// };

class SubGradTensor {
  torch::Tensor father_;
  int64_t index_;

public:
  SubGradTensor() {}
  SubGradTensor(torch::Tensor father, int64_t index) {
    father_ = father;
    index_  = index;
    CHECK_EQ(father_.dim(), 2);
    CHECK(father_.is_cpu());
  }

  torch::Tensor to_tensor() const {
    int dim_            = father_.size(1);
    float* father_data_ = father_.data_ptr<float>();
    at::Device device   = father_.device();
    return torch::from_blob(
        father_data_ + index_ * dim_,
        {dim_},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // at::Tensor::select(father_, 0, index_);
  }
};

// class GradMemoryManager {
//   std::unordered_map<int, std::vector<torch::Tensor>> grad_tensors_by_step_;

//  public:
//   GradMemoryManager() {}

//   void PutGradTensor(const std::vector<torch::Tensor> &input_grads, int step)
//   {
//     int rank = 0;
//     for (auto &grad : input_grads) {
//       auto key = base::SFormat("{}_{}", step, rank);
//       grad_tensors_[key] = grad;
//       rank++;
//     }
//   }

//   void RecycleGradTensor(int step) {
//     auto key = base::SFormat("{}", step);
//     auto it = grad_tensors_.find(key);
//     if (it != grad_tensors_.end()) {
//       grad_tensors_.erase(it);
//     }
//   }
// };
} // namespace recstore
