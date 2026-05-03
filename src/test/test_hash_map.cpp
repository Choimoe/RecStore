#include <omp.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

using dict_type = std::unordered_map<uint64_t, uint64_t>;

int main() {
  dict_type myMap;
  std::mutex myMapMutex;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; ++i) {
    myMap.emplace(i, i);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  omp_set_num_threads(36);

#pragma omp parallel
  {
    int thread_id   = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    for (int i = 0; i < 10; ++i) {
      if (i % num_threads != thread_id)
        continue;

      std::lock_guard<std::mutex> guard(myMapMutex);
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
