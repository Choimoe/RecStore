#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace recstore {

class FrugalCPUThreadPoolExecutor {
public:
  explicit FrugalCPUThreadPoolExecutor(size_t num_threads) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back(&FrugalCPUThreadPoolExecutor::WorkerLoop, this);
    }
  }

  ~FrugalCPUThreadPoolExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  template <typename Func>
  void add(Func&& func) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace(std::forward<Func>(func));
    }
    cv_.notify_one();
  }

  void join() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this] {
      return tasks_.empty() && active_tasks_ == 0;
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
        tasks_.pop();
        ++active_tasks_;
      }

      task();

      {
        std::lock_guard<std::mutex> lock(mutex_);
        --active_tasks_;
        if (tasks_.empty() && active_tasks_ == 0) {
          idle_cv_.notify_all();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  size_t active_tasks_{0};
  bool stopping_{false};
};

} // namespace recstore
