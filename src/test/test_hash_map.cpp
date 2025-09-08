#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/container/F14Map.h>
#include <omp.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <unordered_map>

// using dict_type = folly::F14FastMap<
//       uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
//       folly::f14::DefaultAlloc<std::pair<uint64_t const, uint64_t>>>;

using dict_type = folly::ConcurrentHashMap<uint64_t, uint64_t>;

// using dict_type = std::unordered_map<uint64_t, uint64_t>;

int main() {
  dict_type myMap;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; ++i) {
    myMap.insert(i, i);
    // myMap[i] = i;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  omp_set_num_threads(36);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    
    for (int i = 0; i < 10; ++i) {
      if (i % num_threads != thread_id) continue;
      
      auto it = myMap.find(i);
      if (it != myMap.end()) {
        if (thread_id == 0) {
          printf("T%d %d %lu\n", thread_id, i, it->second);
        }
      }
    }
  }

  return 0;
}