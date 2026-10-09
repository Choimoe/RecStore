#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "base/base.h"
#include "base/bind_core.h"
#include "base/factory.h"
#include "base/init.h"
#include "memory/allocators/concurrent_slab_memory_pool.h"
#include "memory/allocators/persist_loop_slab_allocator.h"
#include "memory/allocators/r2_slab_allocator.h"
#include "memory/malloc.h"

DEFINE_string(
    allocators,
    "CONCURRENT_SLAB_MEMORY_POOL,R2_SLAB,PERSIST_LOOP_SLAB,PERSIST_MEMORY_POOL",
    "Comma-separated MallocApi factory names to benchmark");
DEFINE_int32(thread_num, 16, "Number of concurrent worker threads");
DEFINE_int32(alloc_size,
             128,
             "Allocation size in bytes for each New/Free pair");
DEFINE_int32(warmup_seconds, 1, "Warmup duration in seconds");
DEFINE_int32(running_seconds, 5, "Measured benchmark duration in seconds");
DEFINE_int64(pool_bytes,
             256LL * 1024 * 1024,
             "Backing memory pool size in bytes");
DEFINE_string(medium, "DRAM", "Shm medium passed to MallocApi factory");

namespace {

using MallocFactory = base::
    Factory<base::MallocApi, const std::string&, int64, const std::string&>;

struct BenchResult {
  std::string allocator;
  int threads             = 0;
  int alloc_size          = 0;
  double elapsed_sec      = 0.0;
  uint64_t total_ops      = 0;
  uint64_t failures       = 0;
  double ops_per_sec      = 0.0;
  double alloc_mb_per_sec = 0.0;
  bool ok                 = false;
  std::string error;
};

std::vector<std::string> SplitCommaList(const std::string& value) {
  std::vector<std::string> items;
  std::string current;
  for (char c : value) {
    if (c == ',') {
      if (!current.empty()) {
        items.push_back(current);
        current.clear();
      }
      continue;
    }
    if (c != ' ') {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    items.push_back(current);
  }
  return items;
}

std::string BenchPath(const std::string& allocator) {
  return "/tmp/recstore_malloc_bench_" + allocator + "_" +
         std::to_string(getpid());
}

void RunTimedWorkers(base::MallocApi* allocator,
                     int alloc_size,
                     int duration_seconds,
                     std::vector<uint64_t>* per_thread_ops,
                     std::vector<uint64_t>* per_thread_failures) {
  std::atomic<bool> stop{false};
  const int thread_num = static_cast<int>(per_thread_ops->size());
  std::fill(per_thread_ops->begin(), per_thread_ops->end(), 0);
  std::fill(per_thread_failures->begin(), per_thread_failures->end(), 0);

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(thread_num));
  for (int tid = 0; tid < thread_num; ++tid) {
    threads.emplace_back(
        [allocator,
         alloc_size,
         &stop,
         tid,
         per_thread_ops,
         per_thread_failures]() {
          base::auto_bind_core();
          uint64_t ops      = 0;
          uint64_t failures = 0;
          while (!stop.load(std::memory_order_acquire)) {
            char* ptr = allocator->New(alloc_size);
            if (ptr == nullptr) {
              ++failures;
              continue;
            }
            if (!allocator->Free(ptr)) {
              ++failures;
            }
            ++ops;
          }
          (*per_thread_ops)[static_cast<size_t>(tid)]      = ops;
          (*per_thread_failures)[static_cast<size_t>(tid)] = failures;
        });
  }

  std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
  stop.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
}

BenchResult BenchmarkAllocator(const std::string& allocator_name) {
  BenchResult result;
  result.allocator  = allocator_name;
  result.threads    = FLAGS_thread_num;
  result.alloc_size = FLAGS_alloc_size;

  const std::string path = BenchPath(allocator_name);
  base::file_util::Delete(path, false);

  std::unique_ptr<base::MallocApi> allocator(MallocFactory::NewInstance(
      allocator_name, path, FLAGS_pool_bytes, FLAGS_medium));
  if (!allocator) {
    result.error = "factory returned nullptr";
    return result;
  }

  std::vector<uint64_t> per_thread_ops(
      static_cast<size_t>(FLAGS_thread_num), 0);
  std::vector<uint64_t> per_thread_failures(
      static_cast<size_t>(FLAGS_thread_num), 0);

  if (FLAGS_warmup_seconds > 0) {
    RunTimedWorkers(
        allocator.get(),
        FLAGS_alloc_size,
        FLAGS_warmup_seconds,
        &per_thread_ops,
        &per_thread_failures);
  }

  const auto start = std::chrono::steady_clock::now();
  RunTimedWorkers(
      allocator.get(),
      FLAGS_alloc_size,
      FLAGS_running_seconds,
      &per_thread_ops,
      &per_thread_failures);
  const auto end = std::chrono::steady_clock::now();

  result.elapsed_sec = std::chrono::duration<double>(end - start).count();
  for (size_t tid = 0; tid < per_thread_ops.size(); ++tid) {
    result.total_ops += per_thread_ops[tid];
    result.failures += per_thread_failures[tid];
  }

  if (result.elapsed_sec > 0.0) {
    result.ops_per_sec =
        static_cast<double>(result.total_ops) / result.elapsed_sec;
    result.alloc_mb_per_sec =
        (static_cast<double>(result.total_ops) * FLAGS_alloc_size) /
        result.elapsed_sec / (1024.0 * 1024.0);
  }

  result.ok = true;
  allocator.reset();
  base::file_util::Delete(path, false);
  return result;
}

void PrintSummaryTable(const std::vector<BenchResult>& results) {
  printf("\nMallocApi concurrent New/Free benchmark summary\n");
  printf("threads=%d alloc_size=%d warmup=%ds running=%ds pool_bytes=%ld\n\n",
         FLAGS_thread_num,
         FLAGS_alloc_size,
         FLAGS_warmup_seconds,
         FLAGS_running_seconds,
         static_cast<long>(FLAGS_pool_bytes));

  printf("%-28s %7s %7s %9s %12s %9s %14s %12s\n",
         "Allocator",
         "Threads",
         "Size(B)",
         "Seconds",
         "TotalOps",
         "Failures",
         "Ops/s",
         "AllocMB/s");
  printf("%-28s %7s %7s %9s %12s %9s %14s %12s\n",
         "----------------------------",
         "-------",
         "-------",
         "---------",
         "------------",
         "---------",
         "--------------",
         "------------");

  for (const BenchResult& result : results) {
    if (!result.ok) {
      printf("%-28s %7d %7d %9s %12s %9s %14s %12s  (%s)\n",
             result.allocator.c_str(),
             result.threads,
             result.alloc_size,
             "-",
             "-",
             "-",
             "-",
             "-",
             result.error.c_str());
      continue;
    }

    printf("%-28s %7d %7d %9.3f %12lu %9lu %14.0f %12.2f\n",
           result.allocator.c_str(),
           result.threads,
           result.alloc_size,
           result.elapsed_sec,
           static_cast<unsigned long>(result.total_ops),
           static_cast<unsigned long>(result.failures),
           result.ops_per_sec,
           result.alloc_mb_per_sec);
  }
  printf("\n");
}

} // namespace

int main(int argc, char** argv) {
  base::Init(&argc, &argv);

  const std::vector<std::string> allocator_names =
      SplitCommaList(FLAGS_allocators);
  if (allocator_names.empty()) {
    fprintf(stderr, "No allocators specified via --allocators\n");
    return 1;
  }

  std::vector<BenchResult> results;
  results.reserve(allocator_names.size());
  for (const std::string& allocator_name : allocator_names) {
    results.push_back(BenchmarkAllocator(allocator_name));
  }

  PrintSummaryTable(results);
  return 0;
}
