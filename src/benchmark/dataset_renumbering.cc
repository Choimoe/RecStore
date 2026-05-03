#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/base.h"
#include "base/glob.h"
#include "parse_dataset.h"

#include "base/string.h"
#include <oneapi/tbb/parallel_sort.h>

// input & output
DEFINE_string(dataset_file_str, "/data/project/kuai/dump.2022.08.17/*", "");
// input
DEFINE_string(count_bin_file,
              "/data/project/kuai/dump.2022.08.17.id_count.bin",
              "");
// output
DEFINE_string(output_dataset_meta_file,
              "/data/project/kuai/dump.2022.08.17.meta.txt",
              "");
DEFINE_int32(thread_count, 32, "");

struct FileMeta {
  std::string dataset_file;
  int nr_request;
  int nr_keys;
};

// ID to counter
FileMeta
RenumberID(const std::string& dataset_file,
           const std::unordered_map<uint64_t, uint64_t>& renumber_map) {
  FileMeta file_meta;
  file_meta.dataset_file = dataset_file;

  std::vector<char> file_content;
  CHECK(ReadBinaryFile(dataset_file, &file_content));

  PetCursor cursor(
      file_content.data(), file_content.data() + file_content.size());

  auto nr_request      = cursor.ReadInt();
  file_meta.nr_request = nr_request;
  file_meta.nr_keys    = 0;
  for (int64_t i = 0; i < nr_request; i++) {
    auto nr_keys_in_one_request = cursor.ReadInt();
    file_meta.nr_keys += nr_keys_in_one_request;

    for (int j = 0; j < nr_keys_in_one_request; j++) {
      uint64& key = cursor.ReadUint64();
      int dim     = cursor.ReadInt();
      CHECK(4 <= dim && dim <= 64) << dim;
      CHECK(dim == 4 || dim == 8 || dim == 16 || dim == 32 || dim == 64) << dim;
      auto it = renumber_map.find(key);
      CHECK(it != renumber_map.end());
      key = it->second;
    }
  }
  std::ofstream output(dataset_file, std::ios::binary | std::ios::trunc);
  CHECK(output.is_open());
  output.write(file_content.data(), file_content.size());
  CHECK(output.good());
  return file_meta;
}

// ID to counter
void Check(const std::string& dataset_file, uint64_t max_key) {
  std::vector<char> file_content;
  CHECK(ReadBinaryFile(dataset_file, &file_content));

  PetCursor cursor(
      file_content.data(), file_content.data() + file_content.size());
  auto nr_request = cursor.ReadInt();
  for (int64_t i = 0; i < nr_request; i++) {
    auto nr_keys_in_one_request = cursor.ReadInt();
    for (int j = 0; j < nr_keys_in_one_request; j++) {
      uint64 key = cursor.ReadUint64();
      int dim    = cursor.ReadInt();
      CHECK(4 <= dim && dim <= 64) << dim;
      CHECK(dim == 4 || dim == 8 || dim == 16 || dim == 32 || dim == 64) << dim;
      CHECK(0 <= key && key < max_key);
    }
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> dataset_files;
  for (auto& p : glob::glob(FLAGS_dataset_file_str)) {
    dataset_files.push_back(p);
  }

  int nr_dataset_files = dataset_files.size();
  CHECK_NE(nr_dataset_files, 0);

  // Read Id->Count Map
  std::ifstream if_count_bin_file(
      FLAGS_count_bin_file, std::ios::binary | std::ios::ate);
  std::streamsize size = if_count_bin_file.tellg();
  if_count_bin_file.seekg(0, std::ios::beg);
  std::vector<char> count_bin_content(static_cast<std::size_t>(size));
  if_count_bin_file.read(count_bin_content.data(), size);
  CHECK(if_count_bin_file.good() || if_count_bin_file.eof());
  auto* data = reinterpret_cast<uint64_t*>(count_bin_content.data());
  CHECK(size % (2 * sizeof(uint64_t)) == 0);
  uint64 key_num = size / sizeof(uint64) / 2;
  std::unordered_map<uint64_t, uint64_t> id_count_map;
  std::vector<uint64_t> ids;
  id_count_map.reserve(key_num);
  ids.reserve(key_num);

  LOG(INFO) << "start parse id 2 count map";

  const int id_count_map_thread = 32;
  uint64_t id_count_map_per_thread_count =
      (key_num + id_count_map_thread - 1) / id_count_map_thread;

  std::vector<std::thread> id_count_map_threads;
  std::vector<std::unordered_map<uint64_t, uint64_t>> per_thread_maps(
      id_count_map_thread);

  for (int tid = 0; tid < id_count_map_thread; tid++) {
    uint64_t thread_start = tid * id_count_map_per_thread_count;
    uint64_t thread_end =
        std::min((tid + 1) * id_count_map_per_thread_count, key_num);
    id_count_map_threads.emplace_back(
        [tid,
         thread_start,
         thread_end,
         id_count_map_per_thread_count,
         data,
         &per_thread_maps]() {
          for (auto i = thread_start; i < thread_end; i++) {
            // key: data[i];
            // count: data[i + 1];
            if (tid == 0) {
              RECSTORE_LOG_EVERY_MS(INFO, 30000)
                  << 100 * (i - thread_start) / id_count_map_per_thread_count
                  << " %";
            }
            per_thread_maps[tid][data[2 * i]] = data[2 * i + 1];
          }
        });
  }
  for (auto& t : id_count_map_threads)
    t.join();

  for (const auto& per_thread_map : per_thread_maps) {
    for (const auto& entry : per_thread_map) {
      id_count_map[entry.first] = entry.second;
    }
  }

  LOG(INFO) << "parse id 2 count map done";
  for (uint64_t i = 0; i < key_num; i++) {
    ids.push_back(data[2 * i]);
  }

  LOG(INFO) << "push_back ids done";

  // Sort by count
  LOG(INFO) << "start sort";
  oneapi::tbb::parallel_sort(ids, [&id_count_map](uint64_t a, uint64_t b) {
    return id_count_map.at(a) > id_count_map.at(b);
  });
  LOG(INFO) << "sort done";

  std::unordered_map<uint64_t, uint64_t> renumber_map;
  renumber_map.reserve(key_num);

  LOG(INFO) << "renumber map start";
  for (std::size_t i = 0; i < ids.size(); i++) {
    renumber_map[ids[i]] = i;
  }
  LOG(INFO) << "renumber map done";

  CHECK_EQ(ids.size(), renumber_map.size());
  CHECK_EQ(ids.size(), key_num);
  LOG(INFO) << base::SFormat("dataset has {} keys", key_num);

  std::vector<std::thread> th(FLAGS_thread_count);

  int file_num_per_thread =
      (nr_dataset_files + FLAGS_thread_count - 1) / FLAGS_thread_count;

  std::mutex mutex;
  std::ofstream of(FLAGS_output_dataset_meta_file);
  LOG(INFO) << "before renumbering";
  for (int i = 0; i < FLAGS_thread_count; i++) {
    th[i] = std::thread(
        [i,
         file_num_per_thread,
         &dataset_files,
         nr_dataset_files,
         &mutex,
         &of,
         &renumber_map]() {
          for (int j = i * file_num_per_thread;
               j < std::min((i + 1) * file_num_per_thread, nr_dataset_files);
               j++) {
            if (i == 0)
              RECSTORE_LOG_EVERY_MS(INFO, 10000)
                  << (j - i * file_num_per_thread) * 100 / file_num_per_thread
                  << " %";
            auto file_meta = RenumberID(dataset_files[j], renumber_map);
            {
              std::lock_guard<std::mutex> _(mutex);
              of << file_meta.dataset_file << " " << file_meta.nr_request << " "
                 << file_meta.nr_keys << std::endl;
            }
          }
        });
  }
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    th[i].join();
  }
  LOG(INFO) << "renumbering done";
  for (int i = 0; i < FLAGS_thread_count; i++) {
    th[i] = std::thread(
        [i, file_num_per_thread, &dataset_files, nr_dataset_files, key_num]() {
          for (int j = i * file_num_per_thread;
               j < std::min((i + 1) * file_num_per_thread, nr_dataset_files);
               j++) {
            if (i == 0)
              RECSTORE_LOG_EVERY_MS(INFO, 10000)
                  << (j - i * file_num_per_thread) * 100 / file_num_per_thread
                  << " %";
            Check(dataset_files[j], key_num);
          }
        });
  }
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    th[i].join();
  }
  LOG(INFO) << "check done";
  return 0;
}
