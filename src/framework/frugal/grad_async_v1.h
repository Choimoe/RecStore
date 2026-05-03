#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "IPCTensor.h"
#include "base/base.h"
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/json.h"
#include "base/lock.h"
#include "base/pq.h"
#include "base/queue.h"
#include "base/string.h"
#include "base/timer.h"
#include "frugal_thread_pool.h"
#include "grad_base.h"
#include "grad_memory_manager.h"
#include "parallel_pq.h"
#include "torch_utils.h"

namespace recstore {
class AsyncGradElement {
  static constexpr int kInf = std::numeric_limits<int>::max();

public:
  AsyncGradElement(int64_t id) : id_(id) { RecaculatePriority(); }

  void MarkReadInStepN(int stepN) {
    // LOG(INFO) << "MarkReadInStepN: id=" << id_ << ", step=" << stepN;

    // read_step_.push_back(stepN);

#ifdef GRAD_ASYNC_V1_DEBUG
    lock_.AssertLockHold();
#endif
    int old_size = read_step_.size();
    auto it = std::lower_bound(read_step_.begin(), read_step_.end(), stepN);
    read_step_.insert(it, stepN);
    int new_size = read_step_.size();
    CHECK_EQ(old_size + 1, new_size);
  }

#ifdef USE_SUB_GRAD_TENSOR
  void MarkWriteInStepN(int stepN, const SubGradTensor& grad) {
    write_step_.push_back(stepN);
    write_grad_.push_back(grad);
  }
#else
  void MarkWriteInStepN(int stepN, torch::Tensor grad) {
    write_step_.push_back(stepN);
    write_grad_.push_back(grad);
  }

#endif

  int64_t Priority() const {
    CHECK_EQ(magic_, 0xdeadbeef);
    return priority_;
  }

  void RecaculatePriority() {
    int old_priority = priority_;
    if (read_step_.size() == 0) {
      priority_ = kInf;
      CHECK((old_priority <= priority_) || (old_priority == kInf));
      return;
    }
    int min_read_step = MinReadStep();

    if (write_step_.size() == 0) {
      priority_ = kInf;
    } else {
      priority_ = min_read_step;
    }
#ifdef GRAD_ASYNC_V1_DEBUG
    CHECK((old_priority <= priority_) || (old_priority == kInf));
#endif
  }

  int MinReadStep() const {
#ifdef GRAD_ASYNC_V1_DEBUG
    int min_step = *std::min_element(read_step_.begin(), read_step_.end());
    CHECK_EQ(min_step, read_step_[0]);
#else
    int min_step = *read_step_.begin();
#endif
    return min_step;
  }

  std::string ToString(bool withLock = true) const {
    if (withLock)
      lock_.Lock();
    std::stringstream ss;
    ss << base::SFormat("id={}, read_step=[", id_);
    for (auto each : read_step_) {
      ss << each << ",";
    }
    ss << "], write_step=[";
    for (auto each : write_step_) {
      ss << each << ",";
    }
    ss << "], write_grad=[";
    for (auto each : write_grad_) {
      ss << toString(each, false) << ",";
    }
    ss << "], priority=" << Priority();
    if (withLock)
      lock_.Unlock();
    return ss.str();
  }

  friend struct CompareAsyncGradElement;

  int64_t GetID() const { return id_; }

// NOTE: dont use get grad to pass vector<grad> to the controller
#ifdef USE_SUB_GRAD_TENSOR
  std::pair<std::vector<int>, std::vector<SubGradTensor>> DrainWrites() {
    auto ret_write_step = std::move(write_step_);
    auto ret_write_grad = std::move(write_grad_);
    write_step_.clear();
    write_grad_.clear();
    return std::make_pair(ret_write_step, ret_write_grad);
  }

#else
  std::pair<std::vector<int>, std::vector<torch::Tensor>> DrainWrites() {
    auto ret_write_step = std::move(write_step_);
    auto ret_write_grad = std::move(write_grad_);
    write_step_.clear();
    write_grad_.clear();
    return std::make_pair(ret_write_step, ret_write_grad);
  }

#endif

  void RemoveReadStep(int step_no) {
    // LOG(INFO) << "RemoveReadStep: id=" << id_ << ", step=" << step_no;

    // auto newEnd =
    //     std::remove_if(read_step_.begin(), read_step_.end(),
    //                    [step_no](int value) { return value == step_no; });
    // read_step_.erase(newEnd, read_step_.end());

#ifdef GRAD_ASYNC_V1_DEBUG
    lock_.AssertLockHold();
    CHECK_NE(read_step_.size(), 0) << ToString(false);
    CHECK_EQ(*read_step_.begin(), step_no) << "id is " << id_;
#endif
    if (read_step_.size() != 0)
      read_step_.erase(read_step_.begin());
  }

  void Lock() { return lock_.Lock(); }

  void Unlock() { return lock_.Unlock(); }

private:
  int64_t id_;
  std::vector<int> read_step_;
  std::vector<int> write_step_;
#ifdef USE_SUB_GRAD_TENSOR
  std::vector<SubGradTensor> write_grad_;
#else
  std::vector<torch::Tensor> write_grad_;
#endif

  int64_t priority_;
  mutable base::SpinLock lock_;

public:
  const int magic_ = 0xdeadbeef;
};

struct CompareAsyncGradElement {
  bool operator()(const AsyncGradElement* a, const AsyncGradElement* b) const {
    return a->Priority() > b->Priority();
  }
};

class GradAsyncProcessing : public GradProcessingBase {
  static constexpr bool kUseBackThread_ = false;

protected:
  typedef std::pair<int64_t, torch::Tensor> GradWorkTask;
  static constexpr int kInf = std::numeric_limits<int>::max();

  // for method 2
  const int kUpdatePqWorkerNum_ = 0;
  std::unique_ptr<FrugalCPUThreadPoolExecutor> update_pq_thread_pool_;

public:
  GradAsyncProcessing(const std::string& json_str,
                      const std::vector<std::vector<int64_t>>& cached_range)
      : GradProcessingBase(json_str, cached_range),
        kEmbNumber_(full_emb_.size(0)),
        kGradDim_(full_emb_.size(1)),
        pq_(new recstore::ParallelPq<AsyncGradElement>(kEmbNumber_)) {
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

  void StartThreads() override {
    CHECK(isInitialized_);
    dispatch_thread_stop_flag_ = false;
    grad_thread_stop_flag_     = false;

    if (kUseBackThread_) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_.emplace_back(
            std::bind(&GradAsyncProcessing::GradWorkThread, this, i));
      }
    }
    dispatch_thread_ = std::thread(&GradAsyncProcessing::DispatchThread, this);

    detect_thread_ = std::thread(&GradAsyncProcessing::DetectThread, this);
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

    detect_thread_stop_flag_ = true;
    detect_thread_.join();

    dispatch_thread_stop_flag_ = true;
    dispatch_thread_.join();

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
      pq_->PushOrUpdate(p);
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

      CHECK_EQ(*p_old_end, old_end);
      if (new_end != old_end) {
        // RECSTORE_LOG_EVERY_MS(WARNING, 1000) << base::SFormat(
        // LOG(WARNING) << base::SFormat(
        //     "Rank{}: Detect new sample comes, old_end{}, new_end{}", rank,
        //     old_end, new_end);

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

  void DetectThread() {
    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!detect_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);
      DetectNewSamplesCome();
    }
  }

  void DispatchThread() {
    LOG(INFO) << "dispatch thread: PID=" << getpid();

    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!dispatch_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);
#ifdef XMH_DEBUG_KG
      RECSTORE_LOG_EVERY_MS(INFO, 1000) << "pq is empty";
#endif
      // 后台线程，不断取堆头，dispatch给worker
      if (pq_->empty()) {
#ifdef XMH_DEBUG_KG
        RECSTORE_LOG_EVERY_MS(INFO, 1000) << "pq is empty";
#endif
        continue;
      }
      auto* p = pq_->top();

      // re-read
      if (!p) {
#ifdef XMH_DEBUG_KG
        RECSTORE_LOG_EVERY_MS(INFO, 1000) << "pq.top is nullptr";
#endif
        continue;
      }
      CHECK_EQ(p->magic_, 0xdeadbeef);
      int64_t id = p->GetID();

      // NOTE: 改了优先级
      base::LockGuard element_lock_guard(*p);
      auto [not_used, vec_grads] = p->DrainWrites();

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
          LOG(INFO)
              << "+Grad: "
              // << "| " << p->ToString() << "|" << toString(full_emb_[id])
              << " -> " << toString(full_emb_[id] - clr_ * grad.squeeze(0));
#endif
          full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
        }
      }
      // TODO:
      // 其实这里并不能把他删掉，因为如果用后台线程GradWorkThread，后台线程还没做完
      // pq_->pop_x(p);
      // p->RecaculatePriority();

      p->RecaculatePriority();
      pq_->PushOrUpdate(p);

      round_robin = (round_robin + 1) % nr_background_threads_;
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
        if (grad_thread_stop_flag_.load())
          return;
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
      if (pq_->empty()) {
#ifdef XMH_DEBUG_KG
        LOG(WARNING) << "pq is empty";
#endif
        break;
      }

      int priority = pq_->MinPriority();
      if (priority > step_no) {
#ifdef XMH_DEBUG_KG
        LOG(INFO) << base::SFormat(
            "top(pq)'s priority={} > step_no{}.", priority, step_no);
#endif
        break;
      }
      RECSTORE_LOG_EVERY_MS(WARNING, 1000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", pq.top=" << priority;
    }

    pq_->UpdatePossibleMIN(step_no);
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
            toString(grad_tensor, false));
#endif
        pq_->PushOrUpdate(p);
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
          pq_->PushOrUpdate(p);
        }
      }
    } else if (update_pq_use_omp_ == 2) {
      for (int i = 0; i < kUpdatePqWorkerNum_; i++) {
        update_pq_thread_pool_->add(std::bind(
            &GradAsyncProcessing::UpsertPqThreadMethod2ThreadPool,
            this,
            input_keys,
            input_grads,
            step_no,
            i));
      }
      update_pq_thread_pool_->join();
    } else {
      LOG(FATAL) << "invalid update_pq_use_omp";
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

  void PrintPq() { base::LockGuard _(large_lock_); }

protected:
  int nr_background_threads_;
  std::vector<AsyncGradElement*> dict_;

  const int64_t kEmbNumber_;
  const int kGradDim_;

  recstore::AbstractParallelPq<AsyncGradElement>* pq_;

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
