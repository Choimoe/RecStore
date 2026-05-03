#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

std::unordered_map<int64_t, int64_t> hashTable;
std::mutex hashTableMu;

void insertIntoHashTable(int tid) {
  {
    std::lock_guard<std::mutex> guard(hashTableMu);
    hashTable[tid] = tid;
  }
  for (int _ = 0; _ < 10; _++) {
    std::lock_guard<std::mutex> guard(hashTableMu);
    auto iter = hashTable.find(tid);
    if (iter != hashTable.end() && iter->second == tid) {
      iter->second = tid + 1;
    }
    std::cout << "tid" << tid << " " << (iter != hashTable.end()) << "|"
              << hashTable[tid] << std::endl;
  }
}

int main() {
  int nr_thread = 32;

  std::vector<std::thread> threads;

  for (int i = 0; i < nr_thread; i++) {
    threads.emplace_back(&insertIntoHashTable, i);
  }
  for (int i = 0; i < nr_thread; i++) {
    threads[i].join();
  }

  //   for (const auto& pair : hashTable) {
  //     std::cout << "Key: " << pair.first << ", Value: " << pair.second
  //               << std::endl;
  //   }

  return 0;
}
