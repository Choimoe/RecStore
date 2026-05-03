#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

using dict_type = std::unordered_map<uint64_t, uint64_t>;

dict_type hash_table_;
std::mutex hash_table_mu;

uint64_t KEY_SIZE   = 1000000;
uint64_t THREAD_NUM = 2;
std::atomic<int> now_thread(0);

void test_map(int thread_id) {
  now_thread.fetch_add(1);
  while (now_thread.load() != THREAD_NUM)
    ;
  for (int i = 0; i < KEY_SIZE; i++) {
    std::lock_guard<std::mutex> guard(hash_table_mu);
    hash_table_.emplace(thread_id * KEY_SIZE + i, thread_id * KEY_SIZE + i);
  }
  for (int i = 0; i < KEY_SIZE; i++) {
    std::lock_guard<std::mutex> guard(hash_table_mu);
    if (hash_table_.at(thread_id * KEY_SIZE + i) != thread_id * KEY_SIZE + i) {
      throw std::runtime_error("error");
    }
  }
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < THREAD_NUM; i++) {
    threads.emplace_back(test_map, i);
  }
  for (auto& t : threads) {
    t.join();
  }
  std::cout << "finish" << std::endl;
  return 0;
}
