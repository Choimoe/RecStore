#pragma once
#include "base/queue.h"
#include "frugal_thread_pool.h"
#include "grad_async_v1.h"
#include "grad_base.h"
#include "grad_memory_manager.h"
#include "parallel_pq_v2.h"
// #define XMH_DEBUG_KG

namespace recstore {

class GradAsyncProcessingV2 : public GradProcessingBase {
  typedef std::pair<int64_t, torch::Tensor> GradWorkTask;
  static constexpr int kInf             = std::numeric_limits<int>::max();
  static constexpr bool kUseBackThread_ = false;
  // static constexpr bool kUseBackThread_ = true;

  // for method 2
  const int kUpdatePqWorkerNum_ = 0;
  std::unique_ptr<FrugalCPUThreadPoolExecutor> update_pq_thread_pool_;

public:
  GradAsyncProcessingV2(const std::string& json_str,
                        const std::vector<std::vector<int64_t>>& cached_range)
      : GradProcessingBase(json_str, cached_range),
        kEmbNumber_(full_emb_.size(0)),
        kGradDim_(full_emb_.size(1)),
        pq_(kEmbNumber_, json_config_.value("kUseParallelClean", 1)) {
    nr_background_threads_ = json_config_.at("nr_background_threads");
    CHECK_GT(nr_background_threads_, 0);
    dict_.assign(full_emb_.size(0), nullptr);
    for (int i = 0; i < full_emb_.size(0); i++)
      dict_[i] = new AsyncGradElement(i);

    if (kUseBackThread_) {
      LOG(INFO) << "Use background thread to update emb. nr_background_threads="
                << nr_background_threads_;
      for (int i = 0; i < nr_background_threads_; ++i) {
        backthread_work_queues_.emplace_back(
            std::make_unique<base::ProducerConsumerQueue<GradWorkTask>>(100));
      }
    } else {
      LOG(INFO) << "Use main thread to update emb.";
    }

    for (int rank = 0; rank < num_gpus_; rank++) {
      auto ret_tensor = IPCTensorFactory::FindIPCTensorFromName(
          base::SFormat("circle_buffer_end_cppseen_r{}", rank));
      circle_buffer_end_cppseen_.push_back(ret_tensor);
    }
    for (int rank = 0; rank < num_gpus_; rank++) {
      sample_step_cpp_seen_[rank].store(-1);
    }

    if (update_pq_use_omp_ == 2) {
      int temp = json_config_.value("kUpdatePqWorkerNum", 8);
      *((int*)&kUpdatePqWorkerNum_) = temp;
      update_pq_thread_pool_.reset(
          new FrugalCPUThreadPoolExecutor(kUpdatePqWorkerNum_));
    }
  }

  void StartThreads() {
    CHECK(isInitialized_);
    dispatch_thread_stop_flag_ = false;
    grad_thread_stop_flag_     = false;

    if (kUseBackThread_) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_.emplace_back(
            std::bind(&GradAsyncProcessingV2::GradWorkThread, this, i));
      }
    }
    dispatch_thread_ =
        std::thread(&GradAsyncProcessingV2::DispatchThread, this);

    detect_thread_ = std::thread(&GradAsyncProcessingV2::DetectThread, this);
  }

  void StopThreads() override {
    CHECK(isInitialized_);
    LOG(WARNING) << "call StopThreads. PID = " << getpid();
    GradProcessingBase::StopThreads();
    LOG(WARNING) << "call GradProcessingBase::StopThreads.";

    bool expected = false;
    if (!detect_thread_stop_flag_.compare_exchange_strong(expected, true)) {
      return;
    }

    LOG(WARNING) << "call detect_thread_.join()";
    detect_thread_stop_flag_ = true;
    detect_thread_.join();

    LOG(WARNING) << "call dispatch_thread_.join()";
    dispatch_thread_stop_flag_ = true;
    dispatch_thread_.join();

    LOG(WARNING) << "call backward_threads_.join()";
    grad_thread_stop_flag_ = true;
    if (kUseBackThread_) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_[i].join();
      }
    }
    LOG(WARNING) << "StopThreads done.";
  }

  void RegTensorsPerProcess() override {
    GradProcessingBase::RegTensorsPerProcess();
    StartThreads();
  }

  void WhenNewSampleComes(
      c10::intrusive_ptr<SlicedTensor> input_keys, int rank, int step_no) {
    // 来了一个新样本step号：把<里面的ID, step号>插堆
    auto* data = input_keys->GetSlicedTensor().data_ptr<int64_t>();
    for (int i = 0; i < input_keys->GetSlicedTensor().size(0); ++i) {
      int64_t id = data[i];
      auto* p    = dict_[id];
      base::LockGuard element_lock_guard(*p);
      p->MarkReadInStepN(step_no);
      p->RecaculatePriority();
      pq_.PushOrUpdate(p);
    }
  }

  void DetectNewSamplesCome() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      volatile int64_t* p_new_end =
          circle_buffer_end_per_rank_[rank].data_ptr<int64_t>();
      int64_t new_end = *p_new_end;
      volatile int64_t* p_old_end =
          circle_buffer_end_cppseen_[rank].data_ptr<int64_t>();
      int64_t old_end = *p_old_end;

      if (new_end != old_end) {
        RECSTORE_LOG_EVERY_MS(WARNING, 1000) << base::SFormat(
            "Detect new sample comes, old_end{}, new_end{}", old_end, new_end);

        // add [circle_buffer_old_end, new_end)
        if (new_end < old_end)
          new_end += L_;
        for (int i = old_end; i < new_end; ++i) {
          int pointer = (i % L_);
          int step    = step_tensor_per_rank_[rank][pointer].item<int64_t>();
          WhenNewSampleComes(
              cached_id_circle_buffer_[rank][pointer], rank, step);

          // update the seen sample step of detect threads
          CHECK_LT(sample_step_cpp_seen_[rank], step);
          sample_step_cpp_seen_[rank] = step;
        }
        new_end    = new_end % L_;
        *p_old_end = new_end;
      }
    }
  }

  void ChunkCleanHelper(AsyncGradElement* p) {
    CHECK_EQ(p->magic_, 0xdeadbeef);
    int64_t id = p->GetID();

    // base::LockGuard element_lock_guard(*p);

    p->Lock();
    auto [not_used, vec_grads] = p->DrainWrites();
    p->Unlock();

#ifdef USE_SUB_GRAD_TENSOR
    std::vector<torch::Tensor> grads;
    for (auto& each : vec_grads) {
      grads.push_back(each.to_tensor());
    }
#else
    auto& grads = vec_grads;
#endif

    static int round_robin = 0;
    for (int i = 0; i < grads.size(); i++) {
      if (kUseBackThread_) {
        while (!backthread_work_queues_[round_robin]->write(
            std::make_pair(id, grads[i]))) {
          continue;
        }
      } else {
        auto grad = grads[i].cpu().unsqueeze_(0);
#ifdef XMH_DEBUG_KG
        LOG(INFO) << "+Grad: " << "| " << p->ToString() << "|"
                  << toString(full_emb_[id]) << " -> "
                  << toString(full_emb_[id] - clr_ * grad.squeeze(0));
#endif
        full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
      }
    }

    p->Lock();
    p->RecaculatePriority();
    pq_.PushOrUpdate(p);
    p->Unlock();
    round_robin = (round_robin + 1) % nr_background_threads_;
  }

  void DetectThread() {
    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!detect_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);
      DetectNewSamplesCome();
    }
  }

  void DispatchThread() {
    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!dispatch_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);

      // DetectNewSamplesCome();

      // 后台线程，不断取堆头，dispatch给worker
      if (pq_.empty()) {
        continue;
      }

      // Chunk clean
      pq_.ChunkClean(std::bind(&GradAsyncProcessingV2::ChunkCleanHelper,
                               this,
                               std::placeholders::_1));
    }
  }

  void GradWorkThread(int thread_id) {
    torch::AutoGradMode guard_false(false);
    auto* queue = backthread_work_queues_[thread_id].get();
    CHECK(queue != nullptr);
    while (!grad_thread_stop_flag_.load()) {
      std::pair<int64_t, torch::Tensor> p;
      while (!queue->read(p)) {
        // spin until we get a value
        if (grad_thread_stop_flag_.load()) {
          return;
        }
        continue;
      }
      int64_t id         = p.first;
      torch::Tensor grad = p.second;
      grad               = grad.cpu().unsqueeze_(0);
      full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
    }
  }

  // 等到可以让step_no开始训练
  void BlockToStepN(int step_no) override {
    // 等待堆头的元素大于step_no号
    while (true) {
      base::LockGuard _(large_lock_);

      if (pq_.empty()) {
        // LOG(WARNING) << "pq is empty";
        break;
      }

      int priority = pq_.MinPriority();
      if (priority > step_no) {
#ifdef XMH_DEBUG_KG
        LOG(INFO) << base::SFormat("top(pq)'s priority={} > step_no{}.",
                                   priority,
                                   step_no)
                  << pq_.ToString();
#endif
        break;
      }
      RECSTORE_LOG_EVERY_MS(WARNING, 1000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", pq.top=" << priority;
    }
  }

  void UpsertPqThreadMethod2ThreadPool(
      const std::vector<torch::Tensor>& input_keys,
      const std::vector<torch::Tensor>& input_grads,
      int step_no,
      int thread_id) {
    for (int rank = 0; rank < input_keys.size(); ++rank) {
      auto* data = input_keys[rank].data_ptr<int64_t>();
      CHECK(input_keys[rank].is_cpu());
      CHECK_EQ(input_grads[rank].dim(), 2);

      auto [thread_start, thread_end] =
          base::WorkParititon::MultiThreadWorkPartititon(
              thread_id, kUpdatePqWorkerNum_, input_keys[rank].size(0));

      for (int i = thread_start; i < thread_end; ++i) {
        int64_t id = data[i];

#ifdef USE_SUB_GRAD_TENSOR
        recstore::SubGradTensor grad_tensor(input_grads[rank], i);
#else
#  error "not use SubGradTensor "
        torch::Tensor grad_tensor = input_grads[rank][i];
#endif

        auto* p = dict_[id];
        // NOTE: 改了优先级
        base::LockGuard element_lock_guard(*p);
        p->RemoveReadStep(step_no);
        p->MarkWriteInStepN(step_no, grad_tensor);
        p->RecaculatePriority();
#ifdef XMH_DEBUG_KG
        LOG(INFO) << base::SFormat(
            "Push pq_ | id={}, step_no={}, grad={}",
            id,
            step_no,
            toString(grad_tensor));
#endif
        pq_.PushOrUpdate(p);
      }
    }
  }

  void UpsertPq(const std::vector<torch::Tensor>& input_keys,
                const std::vector<torch::Tensor>& input_grads,
                int step_no) {
    xmh::Timer timer_ProcessBackwardAsync("ProcessBack:UpsertPq");
    if (update_pq_use_omp_ == 0 || update_pq_use_omp_ == 1) {
#pragma omp parallel for num_threads(num_gpus_) if (update_pq_use_omp_)
      for (int rank = 0; rank < input_keys.size(); ++rank) {
        auto* data = input_keys[rank].data_ptr<int64_t>();
        CHECK(input_keys[rank].is_cpu());
        CHECK_EQ(input_grads[rank].dim(), 2);

        for (int i = 0; i < input_keys[rank].size(0); ++i) {
          int64_t id = data[i];

#ifdef USE_SUB_GRAD_TENSOR
          recstore::SubGradTensor grad_tensor(input_grads[rank], i);
#else
#  error "not use SubGradTensor "
          torch::Tensor grad_tensor = input_grads[rank][i];
#endif

          auto* p = dict_[id];
          // NOTE: 改了优先级
          base::LockGuard element_lock_guard(*p);
          p->RemoveReadStep(step_no);
          p->MarkWriteInStepN(step_no, grad_tensor);
          p->RecaculatePriority();
#ifdef XMH_DEBUG_KG
          LOG(INFO) << base::SFormat(
              "Push pq_ | id={}, step_no={}, grad={}",
              id,
              step_no,
              toString(grad_tensor, false));
#endif
          pq_.PushOrUpdate(p);
        }
      }
    } else if (update_pq_use_omp_ == 2) {
      for (int i = 0; i < kUpdatePqWorkerNum_; i++) {
        update_pq_thread_pool_->add(std::bind(
            &GradAsyncProcessingV2::UpsertPqThreadMethod2ThreadPool,
            this,
            input_keys,
            input_grads,
            step_no,
            i));
      }
      update_pq_thread_pool_->join();
    } else {
      LOG(FATAL) << "invalid update_pq_use_omp_";
      //       for (int rank = 0; rank < input_keys.size(); ++rank) {
      //         auto *data = input_keys[rank].data_ptr<int64_t>();
      //         CHECK(input_keys[rank].is_cpu());
      //         CHECK_EQ(input_grads[rank].dim(), 2);

      //         for (int i = 0; i < input_keys[rank].size(0); ++i) {
      //           int64_t id = data[i];

      // #ifdef USE_SUB_GRAD_TENSOR
      //           recstore::SubGradTensor grad_tensor(input_grads[rank], i);
      // #else
      // #error "not use SubGradTensor "
      //           torch::Tensor grad_tensor = input_grads[rank][i];
      // #endif

      //           auto *p = dict_[id];
      //           // NOTE: 改了优先级
      //           base::LockGuard element_lock_guard(*p);
      //           p->RemoveReadStep(step_no);
      //           p->MarkWriteInStepN(step_no, grad_tensor);
      //           p->RecaculatePriority();
      // #ifdef XMH_DEBUG_KG
      //           LOG(INFO) << base::SFormat("Push pq_ | id={}, step_no={},
      //           grad={}",
      //                                       id, step_no,
      //                                       toString(grad_tensor, false));
      // #endif
      //           pq_.PushOrUpdate(p);
      //         }
      //       }
    }
    timer_ProcessBackwardAsync.end();
  }

  void ProcessBackward(const std::vector<torch::Tensor>& input_keys,
                       const std::vector<torch::Tensor>& input_grads,
                       int step_no) override {
    base::LockGuard _(large_lock_);
    // record the update
    // 把 <ID>查一下堆，拿一下step号
    // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
    // 如果在堆，建立映射，把grad指针填进去

    for (int rank = 0; rank < num_gpus_; rank++) {
      while (sample_step_cpp_seen_[rank].load() < step_no)
        RECSTORE_LOG_EVERY_MS(ERROR, 5000)
            << "Stalled in ProcessBackward: "
            << base::SFormat("rank={}, step_no={}, "
                             "sample_step_cpp_seen_[rank]={}",
                             rank,
                             step_no,
                             sample_step_cpp_seen_[rank].load());
    }

    UpsertPq(input_keys, input_grads, step_no);
  }

  void PrintPq() {
    base::LockGuard _(large_lock_);
    //   LOG(ERROR) << pq_.ToString();
  }

private:
  int nr_background_threads_;
  std::vector<AsyncGradElement*> dict_;

  const int64_t kEmbNumber_;
  const int kGradDim_;

  recstore::ParallelPqV2<AsyncGradElement> pq_;

  std::thread detect_thread_;
  std::thread dispatch_thread_;
  std::vector<std::thread> backward_threads_;
  std::vector<std::unique_ptr<base::ProducerConsumerQueue<GradWorkTask>>>
      backthread_work_queues_;

  std::vector<torch::Tensor> circle_buffer_end_cppseen_;

  // base::SpinLock large_lock_;
  base::PlaceboLock large_lock_;

  // shape = [rank]
  std::array<std::atomic_int, 8> sample_step_cpp_seen_;

  std::atomic_bool detect_thread_stop_flag_{false};
  std::atomic_bool dispatch_thread_stop_flag_{false};
  std::atomic_bool grad_thread_stop_flag_{false};
};

} // namespace recstore
