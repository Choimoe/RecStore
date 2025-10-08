#include "CCEH.h"
#include "hash.h"
#include "util.h"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdio.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

thread_local int pending = 0;
thread_local std::vector<std::unique_ptr<coroutine<void>::pull_type>> coros;

#define f_seed 0xc70697UL
#define s_seed 0xc70697UL
// #define f_seed 0xc70f6907UL
// #define s_seed 0xc70f6907UL

using namespace std;

class ExponentialBackoff {
public:
  ExponentialBackoff(int initial_us = 10, int max_us = 10000)
      : current_sleep_us(initial_us),
        max_sleep_us(max_us),
        eng(rd()),
        dist(0.5, 1.5) {}

  void wait() {
    std::this_thread::sleep_for(std::chrono::microseconds(
        static_cast<int>(current_sleep_us * dist(eng))));
    current_sleep_us = std::min(current_sleep_us * 2, max_sleep_us);
  }

private:
  int current_sleep_us;
  int max_sleep_us;
  std::random_device rd;
  std::mt19937 eng;
  std::uniform_real_distribution<> dist;
};

void Segment::execute_path(FileManager* file_manager,
                           vector<pair<size_t, size_t>>& path,
                           Key_t& key,
                           Value_t value) {
  for (int i = path.size() - 1; i > 0; --i)
    bucket[path[i].first] = bucket[path[i - 1].first];
  bucket[path[0].first].value = value;
  mfence();
  bucket[path[0].first].key = key;
}

void Segment::execute_path(vector<pair<size_t, size_t>>& path, Pair _bucket) {
  int i = 0;
  int j = (i + 1) % 2;
  Pair temp[2];
  temp[0] = _bucket;
  for (auto p : path) {
    temp[j]         = bucket[p.first];
    bucket[p.first] = temp[i];
    i               = (i + 1) % 2;
    j               = (i + 1) % 2;
  }
}

vector<pair<size_t, size_t>> Segment::find_path(size_t target, size_t pattern) {
  vector<pair<size_t, size_t>> path;
  path.reserve(kCuckooThreshold);
  path.emplace_back(target, bucket[target].key);
  auto cur = target;
  int i    = 0;
  do {
    Key_t* key = &bucket[cur].key;
    if (*key == INVALID)
      break;
    auto f_hash = hash_funcs[0](key, sizeof(Key_t), f_seed);
    auto s_hash = hash_funcs[2](key, sizeof(Key_t), s_seed);
    if ((f_hash >> (8 * sizeof(f_hash) - local_depth)) != pattern)
      break;
    for (int j = 0; j < kNumPairPerCacheLine * kNumCacheLine; ++j) {
      auto f_idx =
          ((f_hash % kNumGroups) * kNumPairPerCacheLine + j) % kNumSlot;
      auto s_idx =
          ((s_hash % kNumGroups) * kNumPairPerCacheLine + j) % kNumSlot;
      if (f_idx == cur) {
        path.emplace_back(s_idx, bucket[s_idx].key);
        cur = s_idx;
        break;
      } else if (s_idx == cur) {
        path.emplace_back(f_idx, bucket[f_idx].key);
        cur = f_idx;
        break;
      }
    }
    ++i;
  } while (i < kCuckooThreshold);

  if (i == kCuckooThreshold)
    path.resize(0);
  return path;
}

bool Segment::Insert4split(Key_t& key, Value_t value, size_t loc) {
  for (int i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
    auto slot = (loc + i) % kNumSlot;
    if (bucket[slot].key == INVALID) {
      bucket[slot].key   = key;
      bucket[slot].value = value;
      return true;
    }
  }
  return false;
}

PageID_t*
Segment::Split(coroutine<void>::push_type& sink, int index, FileManager* fm) {
  PageID_t* split = new PageID_t[2];
  split[0]        = fm->AllocatePage(sink, index);
  split[1]        = fm->AllocatePage(sink, index);
  auto new_seg1   = fm->GetPage<Segment>(sink, index, split[0]);
  auto new_seg2   = fm->GetPage<Segment>(sink, index, split[1]);
  new_seg1->initSegment(local_depth + 1);
  new_seg2->initSegment(local_depth + 1);
  auto pattern = ((size_t)1 << (sizeof(Key_t) * 8 - local_depth - 1));
  for (int i = 0; i < kNumSlot; ++i) {
    if (bucket[i].key == INVALID)
      continue;
    auto f_hash = hash_funcs[0](&bucket[i].key, sizeof(Key_t), f_seed);
    if (f_hash & pattern) {
      if (!new_seg2->Insert4split(
              bucket[i].key,
              bucket[i].value,
              (f_hash % kNumGroups) * kNumPairPerCacheLine)) {
        auto s_hash = hash_funcs[2](&bucket[i].key, sizeof(Key_t), s_seed);
        if (!new_seg2->Insert4split(
                bucket[i].key,
                bucket[i].value,
                (s_hash % kNumGroups) * kNumPairPerCacheLine)) {
          auto cuckoo_path = new_seg2->find_path(
              (f_hash % kNumGroups) * kNumPairPerCacheLine,
              (f_hash >> (8 * sizeof(f_hash) - new_seg2->local_depth)));
          if (cuckoo_path.size() == 0) {
            cuckoo_path = new_seg2->find_path(
                (s_hash % kNumGroups) * kNumPairPerCacheLine,
                (f_hash >> (8 * sizeof(f_hash) - new_seg2->local_depth)));
          }
          if (cuckoo_path.size() == 0) {
            cerr << "[" << __func__
                 << "]: Cuckoo failed during split for new_seg2" << endl;
          } else {
            new_seg2->execute_path(cuckoo_path, bucket[i]);
          }
        }
      }
    } else {
      if (!new_seg1->Insert4split(
              bucket[i].key,
              bucket[i].value,
              (f_hash % kNumGroups) * kNumPairPerCacheLine)) {
        auto s_hash = hash_funcs[2](&bucket[i].key, sizeof(Key_t), s_seed);
        if (!new_seg1->Insert4split(
                bucket[i].key,
                bucket[i].value,
                (s_hash % kNumGroups) * kNumPairPerCacheLine)) {
          auto cuckoo_path = new_seg1->find_path(
              (f_hash % kNumGroups) * kNumPairPerCacheLine,
              (f_hash >> (8 * sizeof(f_hash) - new_seg1->local_depth)));
          if (cuckoo_path.size() == 0) {
            cuckoo_path = new_seg1->find_path(
                (s_hash % kNumGroups) * kNumPairPerCacheLine,
                (f_hash >> (8 * sizeof(f_hash) - new_seg1->local_depth)));
          }
          if (cuckoo_path.size() == 0) {
            cerr << "[" << __func__
                 << "]: Cuckoo failed during split for new_seg1" << endl;
          } else {
            new_seg1->execute_path(cuckoo_path, bucket[i]);
          }
        }
      }
    }
  }
  fm->Unpin(sink, index, split[0], new_seg1, true);
  fm->Unpin(sink, index, split[1], new_seg2, true);
  return split;
}

CCEH::CCEH(coroutine<void>::push_type& sink,
           int index,
           int queue_size,
           FileManager* _fm) {
  if (!_fm)
    fm = new FileManager(queue_size);
  else
    fm = _fm;
  struct stat st;
  initCCEH(sink, index, 2);
  crashed = false;
}

CCEH::~CCEH() { delete fm; }

void CCEH::initCCEH(
    coroutine<void>::push_type& sink, int index, size_t initCap) {
  crashed            = true;
  dir_header_page_id = fm->AllocatePage(sink, index);
  auto dir_header_ptr =
      fm->GetPage<DirectoryHeader>(sink, index, dir_header_page_id);
  size_t init_depth = static_cast<size_t>(log2(initCap));
  dir_header_ptr->initDirectory(init_depth);
  size_t num_dir_pages =
      (dir_header_ptr->capacity + DirectoryPage::kNumPointers - 1) /
      DirectoryPage::kNumPointers;
  for (size_t i = 0; i < num_dir_pages; ++i)
    dir_header_ptr->dir_pages[i] = fm->AllocatePage(sink, index);
  for (unsigned i = 0; i < dir_header_ptr->capacity; ++i) {
    size_t dir_page_idx = i / DirectoryPage::kNumPointers;
    size_t offset       = i % DirectoryPage::kNumPointers;
    auto dir_page_ptr   = fm->GetPage<DirectoryPage>(
        sink, index, dir_header_ptr->dir_pages[dir_page_idx]);

    PageID_t new_seg_id            = fm->AllocatePage(sink, index);
    dir_page_ptr->segments[offset] = new_seg_id;
    auto new_seg_ptr = fm->GetPage<Segment>(sink, index, new_seg_id);
    new_seg_ptr->initSegment(init_depth);
    fm->Unpin(sink, index, new_seg_id, new_seg_ptr, true);
    fm->Unpin(sink,
              index,
              dir_header_ptr->dir_pages[dir_page_idx],
              dir_page_ptr,
              true);
  }
  fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, true);
  crashed = false;
}

std::shared_mutex& CCEH::get_segment_lock(PageID_t page_id) const {
  std::lock_guard<std::mutex> guard(segment_locks_mutex);
  auto it = segment_locks.find(page_id);
  if (it == segment_locks.end())
    it = segment_locks.emplace(page_id, std::make_unique<std::shared_mutex>())
             .first;
  return *it->second;
}

void CCEH::Insert(
    coroutine<void>::push_type& sink, int index, Key_t& key, Value_t value) {
  auto f_hash = hash_funcs[0](&key, sizeof(Key_t), f_seed);
  auto f_idx  = (f_hash % Segment::kNumGroups) * kNumPairPerCacheLine;
  ExponentialBackoff backoff;
  while (true) {
    // Directory Read
    PageID_t target_page_id;
    size_t dir_depth;
    {
      std::shared_lock<std::shared_mutex> dir_rd_lock(dir_mutex);
      auto dir_header_ptr =
          fm->GetPage<DirectoryHeader>(sink, index, dir_header_page_id);
      dir_depth = dir_header_ptr->depth;
      auto x    = (f_hash >> (8 * sizeof(f_hash) - dir_depth));
      if (x >= dir_header_ptr->capacity) {
        fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
        backoff.wait();
        continue; // Retry
      }
      size_t dir_page_idx         = x / DirectoryPage::kNumPointers;
      size_t offset_in_dir_page   = x % DirectoryPage::kNumPointers;
      PageID_t target_dir_page_id = dir_header_ptr->dir_pages[dir_page_idx];
      auto dir_page_ptr =
          fm->GetPage<DirectoryPage>(sink, index, target_dir_page_id);
      target_page_id = dir_page_ptr->segments[offset_in_dir_page];
      fm->Unpin(sink, index, target_dir_page_id, dir_page_ptr, false);
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
    }
    // Segment Lock and Insert
    std::unique_lock<std::shared_mutex> seg_lock(
        get_segment_lock(target_page_id));
    auto target_ptr = fm->GetPage<Segment>(sink, index, target_page_id);
    // Validation after acquiring segment lock
    {
      std::shared_lock<std::shared_mutex> dir_rd_lock(dir_mutex);
      auto dir_header_ptr =
          fm->GetPage<DirectoryHeader>(sink, index, dir_header_page_id);
      auto x = (f_hash >> (8 * sizeof(f_hash) - dir_header_ptr->depth));
      size_t dir_page_idx       = x / DirectoryPage::kNumPointers;
      size_t offset_in_dir_page = x % DirectoryPage::kNumPointers;
      auto dir_page_ptr         = fm->GetPage<DirectoryPage>(
          sink, index, dir_header_ptr->dir_pages[dir_page_idx]);
      PageID_t current_target_page_id =
          dir_page_ptr->segments[offset_in_dir_page];
      fm->Unpin(sink,
                index,
                dir_header_ptr->dir_pages[dir_page_idx],
                dir_page_ptr,
                false);
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
      if (current_target_page_id != target_page_id) {
        fm->Unpin(sink, index, target_page_id, target_ptr, false);
        seg_lock.unlock();
        backoff.wait();
        continue; // Stale segment, retry
      }
    }
    // Insert into segment if space is available
    auto pattern = (f_hash >> (8 * sizeof(f_hash) - target_ptr->local_depth));
    for (unsigned i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
      auto loc       = (f_idx + i) % Segment::kNumSlot;
      auto storedKey = target_ptr->bucket[loc].key;
      if (storedKey == key) {
        target_ptr->bucket[loc].value = value;
        mfence();
        fm->Unpin(sink, index, target_page_id, target_ptr, true);
        return;
      }
      if ((((hash_funcs[0](&storedKey, sizeof(Key_t), f_seed) >>
             (8 * sizeof(f_hash) - target_ptr->local_depth)) != pattern) ||
           (storedKey == INVALID)) &&
          (storedKey != SENTINEL)) {
        if (CAS(&target_ptr->bucket[loc].key, &storedKey, SENTINEL)) {
          target_ptr->bucket[loc].value = value;
          mfence();
          target_ptr->bucket[loc].key = key;
          fm->Unpin(sink, index, target_page_id, target_ptr, true);
          return;
        }
      }
    }
    auto s_hash = hash_funcs[2](&key, sizeof(Key_t), s_seed);
    auto s_idx  = (s_hash % Segment::kNumGroups) * kNumPairPerCacheLine;
    for (unsigned i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
      auto loc       = (s_idx + i) % Segment::kNumSlot;
      auto storedKey = target_ptr->bucket[loc].key;
      if (storedKey == key) {
        target_ptr->bucket[loc].value = value;
        mfence();
        fm->Unpin(sink, index, target_page_id, target_ptr, true);
        return;
      }
      if ((((hash_funcs[0](&storedKey, sizeof(Key_t), f_seed) >>
             (8 * sizeof(s_hash) - target_ptr->local_depth)) != pattern) ||
           (storedKey == INVALID)) &&
          (storedKey != SENTINEL)) {
        if (CAS(&target_ptr->bucket[loc].key, &storedKey, SENTINEL)) {
          target_ptr->bucket[loc].value = value;
          mfence();
          target_ptr->bucket[loc].key = key;
          fm->Unpin(sink, index, target_page_id, target_ptr, true);
          return;
        }
      }
    }
    // Segment is full, need to split
    auto target_local_depth = target_ptr->local_depth;
    // Try to acquire directory write lock to perform split
    std::unique_lock<std::shared_mutex> dir_wr_lock(
        dir_mutex, std::try_to_lock);
    if (!dir_wr_lock.owns_lock()) {
      fm->Unpin(sink, index, target_page_id, target_ptr, false);
      seg_lock.unlock();
      backoff.wait();
      continue; // Failed to get dir write lock, retry
    }
    // Re-check segment state after acquiring all locks
    if (target_ptr->local_depth != target_local_depth) {
      fm->Unpin(sink, index, target_page_id, target_ptr, false);
      backoff.wait();
      continue; // Another thread already split this segment
    }
    PageID_t* s = target_ptr->Split(sink, index, fm);
    auto dir_header_ptr =
        fm->GetPage<DirectoryHeader>(sink, index, dir_header_page_id);
    auto x = (f_hash >> (8 * sizeof(f_hash) - dir_header_ptr->depth));
    if (target_ptr->local_depth == dir_header_ptr->depth) {
      // Directory expansion
      size_t new_capacity = dir_header_ptr->capacity * 2;
      if ((new_capacity / DirectoryPage::kNumPointers) >
          DirectoryHeader::kMaxDirectoryPages) {
        fm->Unpin(sink, index, target_page_id, target_ptr, true);
        fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
        delete[] s;
        std::cerr << "Expansion limit reached" << std::endl;
        return; // Expansion limit reached
      }
      auto old_dir_header_page_id     = this->dir_header_page_id;
      auto old_dir_header_ptr         = dir_header_ptr;
      PageID_t new_dir_header_page_id = fm->AllocatePage(sink, index);
      auto new_dir_header_ptr =
          fm->GetPage<DirectoryHeader>(sink, index, new_dir_header_page_id);
      new_dir_header_ptr->initDirectory(old_dir_header_ptr->depth + 1);
      size_t num_new_dir_pages =
          (new_dir_header_ptr->capacity + DirectoryPage::kNumPointers - 1) /
          DirectoryPage::kNumPointers;
      for (size_t i = 0; i < num_new_dir_pages; ++i) {
        new_dir_header_ptr->dir_pages[i] = fm->AllocatePage(sink, index);
      }
      x = (f_hash >> (8 * sizeof(f_hash) - old_dir_header_ptr->depth));
      DirectoryPage* current_old_dir_page_ptr = nullptr;
      size_t current_old_dir_page_idx         = -1;
      DirectoryPage* current_new_dir_page_ptr = nullptr;
      PageID_t current_new_dir_page_id        = INVALID_PAGE;
      size_t current_new_dir_page_idx         = -1;
      for (size_t i = 0; i < new_dir_header_ptr->capacity; ++i) {
        size_t new_page_idx = i / DirectoryPage::kNumPointers;
        if (new_page_idx != current_new_dir_page_idx) {
          if (current_new_dir_page_ptr != nullptr)
            fm->Unpin(sink,
                      index,
                      current_new_dir_page_id,
                      current_new_dir_page_ptr,
                      true);
          current_new_dir_page_idx = new_page_idx;
          current_new_dir_page_id =
              new_dir_header_ptr->dir_pages[current_new_dir_page_idx];
          current_new_dir_page_ptr =
              fm->GetPage<DirectoryPage>(sink, index, current_new_dir_page_id);
        }
        size_t offset_in_new_page = i % DirectoryPage::kNumPointers;
        size_t old_i              = i / 2;
        if (old_i == x) {
          current_new_dir_page_ptr->segments[offset_in_new_page] =
              (i % 2 == 0) ? s[0] : s[1];
        } else {
          size_t old_page_idx = old_i / DirectoryPage::kNumPointers;
          if (old_page_idx != current_old_dir_page_idx) {
            if (current_old_dir_page_ptr != nullptr)
              fm->Unpin(sink,
                        index,
                        old_dir_header_ptr->dir_pages[current_old_dir_page_idx],
                        current_old_dir_page_ptr,
                        false);
            current_old_dir_page_idx = old_page_idx;
            current_old_dir_page_ptr = fm->GetPage<DirectoryPage>(
                sink,
                index,
                old_dir_header_ptr->dir_pages[current_old_dir_page_idx]);
          }
          size_t offset_in_old_page = old_i % DirectoryPage::kNumPointers;
          current_new_dir_page_ptr->segments[offset_in_new_page] =
              current_old_dir_page_ptr->segments[offset_in_old_page];
        }
      }
      if (current_new_dir_page_ptr != nullptr)
        fm->Unpin(sink,
                  index,
                  current_new_dir_page_id,
                  current_new_dir_page_ptr,
                  true);
      if (current_old_dir_page_ptr != nullptr)
        fm->Unpin(sink,
                  index,
                  old_dir_header_ptr->dir_pages[current_old_dir_page_idx],
                  current_old_dir_page_ptr,
                  false);

      fm->Unpin(sink, index, new_dir_header_page_id, new_dir_header_ptr, true);
      this->dir_header_page_id = new_dir_header_page_id;

      fm->Unpin(sink, index, target_page_id, target_ptr, false);
      fm->Unpin(sink, index, old_dir_header_page_id, old_dir_header_ptr, false);
      delete[] s;
    } else {
      // Normal split
      int stride = pow(2, dir_header_ptr->depth - target_local_depth);
      auto loc   = x - (x % stride);
      for (int i = 0; i < stride / 2; ++i) {
        size_t idx          = loc + stride / 2 + i;
        size_t dir_page_idx = idx / DirectoryPage::kNumPointers;
        auto dir_page_ptr   = fm->GetPage<DirectoryPage>(
            sink, index, dir_header_ptr->dir_pages[dir_page_idx]);
        dir_page_ptr->segments[idx % DirectoryPage::kNumPointers] = s[1];
        fm->Unpin(sink,
                  index,
                  dir_header_ptr->dir_pages[dir_page_idx],
                  dir_page_ptr,
                  true);
      }
      mfence(); // ensure writes for s[1] are globally visible before s[0]
      for (int i = 0; i < stride / 2; ++i) {
        size_t idx          = loc + i;
        size_t dir_page_idx = idx / DirectoryPage::kNumPointers;
        auto dir_page_ptr   = fm->GetPage<DirectoryPage>(
            sink, index, dir_header_ptr->dir_pages[dir_page_idx]);
        dir_page_ptr->segments[idx % DirectoryPage::kNumPointers] = s[0];
        fm->Unpin(sink,
                  index,
                  dir_header_ptr->dir_pages[dir_page_idx],
                  dir_page_ptr,
                  true);
      }
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
      fm->Unpin(sink, index, target_page_id, target_ptr, false);
      delete[] s;
    }
  }
}

bool CCEH::Delete(Key_t& key) { return false; }

Value_t
CCEH::Get(coroutine<void>::push_type& sink, int index, const Key_t& key) {
  auto f_hash = hash_funcs[0](&key, sizeof(Key_t), f_seed);
  auto f_idx  = (f_hash % Segment::kNumGroups) * kNumPairPerCacheLine;
  ExponentialBackoff backoff;
  while (true) {
    std::shared_lock<std::shared_mutex> dir_rd_lock(dir_mutex);
    auto dir_header_ptr =
        fm->GetPage<DirectoryHeader>(sink, index, dir_header_page_id);
    size_t x = (f_hash >> (8 * sizeof(f_hash) - dir_header_ptr->depth));
    if (x >= dir_header_ptr->capacity) {
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
      backoff.wait();
      continue;
    }
    size_t dir_page_idx  = x / DirectoryPage::kNumPointers;
    size_t offset        = x % DirectoryPage::kNumPointers;
    PageID_t dir_page_id = dir_header_ptr->dir_pages[dir_page_idx];
    auto dir_page_ptr    = fm->GetPage<DirectoryPage>(sink, index, dir_page_id);
    auto target_page_id  = dir_page_ptr->segments[offset];
    fm->Unpin(sink, index, dir_page_id, dir_page_ptr, false);
    if (target_page_id == INVALID_PAGE) {
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
      backoff.wait();
      continue;
    }
    std::shared_lock<std::shared_mutex> seg_lock(
        get_segment_lock(target_page_id));
    auto target_ptr = fm->GetPage<Segment>(sink, index, target_page_id);
    auto check_x    = (f_hash >> (8 * sizeof(f_hash) - dir_header_ptr->depth));
    if (x != check_x) {
      fm->Unpin(sink, index, target_page_id, target_ptr, false);
      fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
      backoff.wait();
      continue;
    }
    fm->Unpin(sink, index, dir_header_page_id, dir_header_ptr, false);
    dir_rd_lock.unlock();
    for (int i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
      auto loc = (f_idx + i) % Segment::kNumSlot;
      if (target_ptr->bucket[loc].key == key) {
        Value_t v = target_ptr->bucket[loc].value;
        fm->Unpin(sink, index, target_page_id, target_ptr, false);
        return v;
      }
    }
    auto s_hash = hash_funcs[2](&key, sizeof(Key_t), s_seed);
    auto s_idx  = (s_hash % Segment::kNumGroups) * kNumPairPerCacheLine;
    for (int i = 0; i < kNumPairPerCacheLine * kNumCacheLine; ++i) {
      auto loc = (s_idx + i) % Segment::kNumSlot;
      if (target_ptr->bucket[loc].key == key) {
        Value_t v = target_ptr->bucket[loc].value;
        fm->Unpin(sink, index, target_page_id, target_ptr, false);
        return v;
      }
    }
    fm->Unpin(sink, index, target_page_id, target_ptr, false);
    return NONE;
  }
}