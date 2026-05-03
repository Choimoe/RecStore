#pragma once

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace base {

class CPUThreadPoolExecutor {
public:
  class Options {
  public:
    enum class Blocking { block, prohibit };

    Options& setBlocking(Blocking blocking) {
      blocking_ = blocking;
      return *this;
    }

    Options& setMaxQueueSize(size_t max_queue_size) {
      max_queue_size_ = max_queue_size;
      return *this;
    }

  private:
    friend class CPUThreadPoolExecutor;

    Blocking blocking_     = Blocking::block;
    size_t max_queue_size_ = 0;
  };

  explicit CPUThreadPoolExecutor(size_t num_threads)
      : CPUThreadPoolExecutor(num_threads, Options()) {}

  CPUThreadPoolExecutor(size_t num_threads, Options options)
      : options_(options) {
    if (num_threads == 0) {
      throw std::invalid_argument("CPUThreadPoolExecutor requires threads");
    }
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  ~CPUThreadPoolExecutor() {
    join();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    queue_not_full_cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  CPUThreadPoolExecutor(const CPUThreadPoolExecutor&)            = delete;
  CPUThreadPoolExecutor& operator=(const CPUThreadPoolExecutor&) = delete;

  template <typename Func>
  void add(Func&& func) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("cannot add task after shutdown");
    }
    if (options_.max_queue_size_ != 0) {
      if (options_.blocking_ == Options::Blocking::prohibit &&
          tasks_.size() >= options_.max_queue_size_) {
        throw std::runtime_error("thread pool queue is full");
      }
      queue_not_full_cv_.wait(lock, [this] {
        return stopping_ || tasks_.size() < options_.max_queue_size_;
      });
      if (stopping_) {
        throw std::runtime_error("cannot add task after shutdown");
      }
    }
    ++pending_tasks_;
    tasks_.emplace_back(std::forward<Func>(func));
    cv_.notify_one();
  }

  void join() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this] {
      return tasks_.empty() && pending_tasks_ == 0;
    });
  }

private:
  void WorkerLoop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
        queue_not_full_cv_.notify_one();
      }

      try {
        task();
      } catch (...) {
        // Match executor-style fire-and-forget behavior: failed tasks do not
        // stop worker threads or escape through join().
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        --pending_tasks_;
        if (tasks_.empty() && pending_tasks_ == 0) {
          idle_cv_.notify_all();
        }
      }
    }
  }

  Options options_;
  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::condition_variable queue_not_full_cv_;
  size_t pending_tasks_ = 0;
  bool stopping_        = false;
};

} // namespace base
