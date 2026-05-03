#include "IPC_barrier.h"
#include "kg_controller.h"

using namespace recstore;

DEFINE_string(backMode, "CppSync", "CppSync or CppAsync");

namespace recstore {

class CircleBuffer {
private:
  int L_;
  int start_ = 0;
  int end_   = 0;

  int rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> sliced_id_tensor_;
  torch::Tensor circle_buffer_end_;
  torch::Tensor circle_buffer_old_end_;
  torch::Tensor step_tensor_;

  std::string backmode_;

  const bool is_async_process_;

public:
  CircleBuffer(int L, int rank, std::string backmode)
      : L_(L),
        rank_(rank),
        backmode_(backmode),
        is_async_process_(
            backmode_ == "CppAsync" || backmode_ == "CppAsyncV2") {
    for (int i = 0; i < L_; i++) {
      sliced_id_tensor_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          base::SFormat("cached_sampler_r{}_{}", rank, i),
          {int(1e6)},
          torch::kInt64));
    }

    circle_buffer_end_ =
        IPCTensorFactory::NewIPCTensor(
            base::SFormat("circle_buffer_end_r{}", rank),
            {int(1)},
            torch::kInt64)
            .value();

    circle_buffer_old_end_ =
        IPCTensorFactory::NewIPCTensor(
            base::SFormat("circle_buffer_end_cppseen_r{}", rank),
            {int(1)},
            torch::kInt64)
            .value();

    step_tensor_ =
        IPCTensorFactory::NewIPCTensor(
            base::SFormat("step_r{}", rank), {int(L_)}, torch::kInt64)
            .value();

    circle_buffer_end_.fill_(0);
    circle_buffer_old_end_.fill_(0);
  }

  void Push(int step, torch::Tensor item, bool sync = false) {
    sliced_id_tensor_[end_]->Copy_(item, false);
    step_tensor_[end_] = step;

    end_ = (end_ + 1) % L_;

    asm volatile("mfence" ::: "memory");
    circle_buffer_end_[0] = end_;

    if (is_async_process_ && sync) {
      while (circle_buffer_end_[0].item<int64_t>() !=
             circle_buffer_old_end_[0].item<int64_t>()) {
        RECSTORE_LOG_EVERY_MS(INFO, 5000) << base::SFormat(
            "Waiting for CppAsync to finish processing the item {}",
            circle_buffer_old_end_[0].item<int64_t>());
      }
    }

    if (end_ == start_) {
      // 这里防止吃样本
      // TODO:只有当start_在cppend~end_之间的时候，才能不等待，往后走
      // 不这么写是有BUG的，但没时间查了 while (1) {
      //   int64_t cppend = circle_buffer_old_end_[0].item<int64_t>();
      //   for (int i = cppend; i < end_ + L_; i++) {
      //     int j = i % L_;
      //     if (j == start_) {
      //       break;
      //     }
      //   }
      // }
      start_ = (start_ + 1) % L_;
    }
  }

  std::pair<int, c10::intrusive_ptr<SlicedTensor>> Pop() {
    if (start_ == end_) {
      LOG(FATAL) << "empty sample buffer";
    }

    auto item = sliced_id_tensor_[start_];

    int step = step_tensor_[start_].item<int64_t>();

    start_ = (start_ + 1) % L_;
    return std::make_pair(step, item);
  }
};

class BasePerfSampler {
public:
  BasePerfSampler(int rank,
                  int L,
                  int num_ids_per_step,
                  int64_t full_emb_capacity,
                  std::string backmode)
      : rank_(rank),
        L_(L),
        ids_circle_buffer_(L, rank, backmode),
        sampler_iter_num_(0),
        num_ids_per_step_(num_ids_per_step),
        full_emb_capacity_(full_emb_capacity),
        backmode_(backmode) {}

  void Prefill() {
    // NOTE: must be L_ - 1
    for (int i = 0; i < L_ - 1; ++i) {
      torch::Tensor entity_id = gen_next_sample();
      ids_circle_buffer_.Push(sampler_iter_num_, entity_id, true);
      ++sampler_iter_num_;
    }
  }

  std::pair<int, torch::Tensor> __next__() {
    auto entity_id      = gen_next_sample();
    auto [step, sample] = ids_circle_buffer_.Pop();

    // TODO: 继上面的问题，这里只能先用同步
    ids_circle_buffer_.Push(sampler_iter_num_, entity_id, false);
    // ids_circle_buffer_.Push(sampler_iter_num_, entity_id, true);
    ++sampler_iter_num_;
    return std::make_pair(step, sample->GetSlicedTensor());
  }

protected:
  virtual torch::Tensor gen_next_sample() = 0;

  int rank_;
  int L_;
  CircleBuffer ids_circle_buffer_;
  int sampler_iter_num_;
  int num_ids_per_step_;
  int64_t full_emb_capacity_;
  std::string backmode_;
};

class TestPerfSampler : public BasePerfSampler {
public:
  TestPerfSampler(int rank,
                  int L,
                  int num_ids_per_step,
                  int full_emb_capacity,
                  std::string backmode)
      : BasePerfSampler(
            rank, L, num_ids_per_step, full_emb_capacity, backmode) {}

protected:
  torch::Tensor gen_next_sample() override {
    // return torch::randint(0, full_emb_capacity_, {num_ids_per_step_}).cuda();
    return UniformRandom();
  }

  torch::Tensor UniformRandom() {
    return torch::randint(0, full_emb_capacity_, {num_ids_per_step_}).cuda();
  }

  torch::Tensor ZipfianRandom() {
    return torch::randint(0, full_emb_capacity_, {num_ids_per_step_}).cuda();
  }
};

class VirtualEnvironment {
private:
  int num_gpus_;
  int emb_dim_;
  int64_t cached_capacity_;
  int64_t full_emb_capacity_;
  int L_;
  std::string backmode_;
  int num_ids_per_step_;

  std::vector<torch::Tensor> embedding_cache_;
  torch::Tensor full_emb_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_;

  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_;

  // for both
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_cpu_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_cpu_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_gpu_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_gpu_;

  std::vector<TestPerfSampler> test_perf_sampler_;

  std::vector<std::thread> threads_;

  base::Barrier* barrier_;
  std::string backgrad_init_;

public:
  VirtualEnvironment(const std::string& json_str, int64_t cached_capcacity) {
    auto json_config   = json::parse(json_str);
    num_gpus_          = json_config.at("num_gpus");
    emb_dim_           = json_config.at("emb_dim");
    cached_capacity_   = cached_capcacity;
    full_emb_capacity_ = json_config.at("full_emb_capacity");
    num_ids_per_step_  = json_config.at("num_ids_per_step");
    L_                 = json_config.at("L");
    backmode_          = json_config.at("backwardMode");
    backgrad_init_     = json_config.at("backgrad_init");

    barrier_ = new base::Barrier(num_gpus_);

    CreateVirutalEnvironment();
  }

  void CreateVirutalEnvironment() {
    full_emb_ = IPCTensorFactory::NewIPCTensor(
                    "full_emb", {full_emb_capacity_, emb_dim_}, torch::kFloat32)
                    .value();

    for (int rank = 0; rank < num_gpus_; rank++) {
      embedding_cache_.push_back(IPCTensorFactory::NewIPCGPUTensor(
          base::SFormat("embedding_cache_{}", rank),
          {cached_capacity_, emb_dim_},
          torch::kFloat32,
          rank));

      input_keys_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          base::SFormat("input_keys_{}", rank), {int(1e6)}, torch::kInt64));
      input_keys_neg_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          base::SFormat("input_keys_neg_{}", rank), {int(1e6)}, torch::kInt64));

      if (backgrad_init_ == "cpu") {
        backward_grads_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
            base::SFormat("backward_grads_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32));
        backward_grads_neg_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
            base::SFormat("backward_grads_neg_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32));
      } else if (backgrad_init_ == "gpu") {
        backward_grads_.push_back(IPCTensorFactory::NewSlicedIPCGPUTensor(
            base::SFormat("backward_grads_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32,
            rank));
        backward_grads_neg_.push_back(IPCTensorFactory::NewSlicedIPCGPUTensor(
            base::SFormat("backward_grads_neg_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32,
            rank));
      } else if (backgrad_init_ == "both") {
        backward_grads_gpu_.push_back(IPCTensorFactory::NewSlicedIPCGPUTensor(
            base::SFormat("backward_grads_{}_gpu", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32,
            rank));
        backward_grads_neg_gpu_.push_back(
            IPCTensorFactory::NewSlicedIPCGPUTensor(
                base::SFormat("backward_grads_neg_{}_gpu", rank),
                {int(1e6), emb_dim_},
                torch::kFloat32,
                rank));

        backward_grads_cpu_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
            base::SFormat("backward_grads_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32));
        backward_grads_neg_cpu_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
            base::SFormat("backward_grads_neg_{}", rank),
            {int(1e6), emb_dim_},
            torch::kFloat32));

      } else {
        LOG(FATAL) << "error";
      }

      test_perf_sampler_.emplace_back(
          rank, L_, num_ids_per_step_, full_emb_capacity_, backmode_);
    }
  }

  void PrefillSampler() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      test_perf_sampler_[rank].Prefill();
    }
  }

  void StartThreads() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      threads_.emplace_back(&VirtualEnvironment::RunThread, this, rank);
    }
  }

  void StopThreads() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      threads_[rank].join();
    }
  }

private:
  void RunThread(int rank) {
    KGCacheController* controller = KGCacheController::GetInstance();
    cudaSetDevice(rank);
    int step_no   = 0;
    auto backgrad = torch::randn({num_ids_per_step_, emb_dim_}).cuda();
    while (true) {
      // if (rank == 0 && step_no == 10) ProfilerStart("/tmp/profile.prof");

      // 1. Get the next step
      auto [sample_step, next_ids] = test_perf_sampler_[rank].__next__();
      CHECK_EQ(sample_step, step_no);

      // LOG(INFO) << base::SFormat("rank{}: {}", rank, toString(next_ids));

      input_keys_[rank]->Copy_(next_ids, false);

      if (backgrad_init_ == "both") {
        backward_grads_cpu_[rank]->Copy_(backgrad, false);
        backward_grads_gpu_[rank]->Copy_(backgrad, false);
      } else if (backgrad_init_ == "cpu") {
        backward_grads_[rank]->Copy_(backgrad, false);
      } else if (backgrad_init_ == "gpu") {
        backward_grads_[rank]->Copy_(backgrad, false);
      } else {
        LOG(FATAL) << "error";
      }

      LOG(INFO) << "Step " << step_no;
      // 2. Forward

      // 3. Backward
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      // 4. Update

      barrier_->Wait();

      if (rank == 0) {
        LOG(INFO) << "rank 0 ProcessOneStep";
        controller->ProcessOneStep(step_no);
      }
      barrier_->Wait();

      step_no++;
      if (rank == 0)
        controller->BlockToStepN(step_no);
      barrier_->Wait();

      // if (rank == 0 && step_no == 100) ProfilerStop();
      if (step_no == 30)
        break;
    }
  }
};

} // namespace recstore

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string json_str = base::SFormat(
      R"({{
            "num_gpus": 4,
            "L": 10,
            "kForwardItersPerStep": 1,
            "clr": 2,
            "backwardMode": "{}",
            "nr_background_threads": 32,
            "full_emb_capacity": 10000000,
            "emb_dim" : 400,
            "num_ids_per_step": 25000,
            "backgrad_init": "both"
        }})",
      FLAGS_backMode);

  auto json_config = json::parse(json_str);

  json_str = json_config.dump();

  int64_t full_emb_capacity        = 100LL * int(1e6);
  json_config["full_emb_capacity"] = full_emb_capacity;

  LOG(INFO) << json_config;

  int num_gpus = json_config.at("num_gpus");

  IPCTensorFactory::ClearIPCMemory();
  MultiProcessBarrierFactory::ClearIPCMemoryStatic();

  xmh::Reporter::StartReportThread();

  float cache_ratio              = 0.02;
  int64_t total_cached_capcacity = full_emb_capacity * cache_ratio * num_gpus;
  int64_t per_shard_cached_capcacity = full_emb_capacity * cache_ratio;

  VirtualEnvironment env(json_str, per_shard_cached_capcacity);

  std::vector<std::vector<int64_t>> cached_range;
  for (int i = 0; i < num_gpus; i++)
    cached_range.push_back({i * per_shard_cached_capcacity,
                            std::min((i + 1) * per_shard_cached_capcacity,
                                     total_cached_capcacity)});

  auto hold_pointer =
      KGCacheController::Init(json_str, cached_range, full_emb_capacity);
  KGCacheController* controller = KGCacheController::GetInstance();
  controller->RegTensorsPerProcess();
  env.PrefillSampler();
  env.StartThreads();
  env.StopThreads();
  controller->StopThreads();
  return 0;
}
