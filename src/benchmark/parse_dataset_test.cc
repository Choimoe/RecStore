#include "base/base.h"
#include "base/glob.h"

#include "parse_dataset.h"

#include <string>

DEFINE_string(dataset_file_str, "", "");
DEFINE_int32(thread_count, 32, "");

class PetCursor {
public:
  PetCursor(const char* start, const char* end) : start_(start), end_(end) {
    data_ = start_;
  }

  int ReadInt() {
    auto r = *(int*)data_;
    data_ += sizeof(int);
    CHECK_LE(data_, end_);
    return r;
  }
  uint64 ReadUint64() {
    auto r = *(uint64*)data_;
    data_ += sizeof(uint64);
    CHECK_LE(data_, end_);
    return r;
  }
  int64 ReadInt64() {
    auto r = *(int64*)data_;
    data_ += sizeof(int64);
    CHECK_LE(data_, end_);
    return r;
  }

  bool isEnd() const { return data_ == end_; }

private:
  const char* data_;
  const char* start_;
  const char* end_;
};

void ParseDataSet(const std::string& filename) {
  std::vector<char> file_contents;
  CHECK(ReadBinaryFile(filename, &file_contents));
  PetCursor cursor(
      file_contents.data(), file_contents.data() + file_contents.size());

  auto nr_request = cursor.ReadInt();
  for (int64_t i = 0; i < nr_request; i++) {
    auto nr_keys_in_one_request = cursor.ReadInt();
    for (int j = 0; j < nr_keys_in_one_request; j++) {
      uint64 key = cursor.ReadUint64();
      int dim    = cursor.ReadInt();
      CHECK(4 <= dim && dim <= 64) << dim;
      CHECK(dim == 4 || dim == 8 || dim == 16 || dim == 32 || dim == 64) << dim;
    }
  }
  CHECK(cursor.isEnd());
}
int main(int argc, char** argv) {
  std::vector<std::string> dataset_files;
  for (auto& p : glob::glob(FLAGS_dataset_file_str)) {
    dataset_files.push_back(p);
  }

  int nr_dataset_files = dataset_files.size();
  CHECK_NE(nr_dataset_files, 0);

  std::vector<std::thread> th(FLAGS_thread_count);

  int file_num_per_thread =
      (nr_dataset_files + FLAGS_thread_count - 1) / FLAGS_thread_count;
  for (int i = 0; i < FLAGS_thread_count; i++) {
    th[i] = std::thread(
        [i, file_num_per_thread, &dataset_files, nr_dataset_files]() {
          for (int j = i * file_num_per_thread;
               j < std::min((i + 1) * file_num_per_thread, nr_dataset_files);
               j++) {
            ParseDataSet(dataset_files[j]);
          }
        });
  }
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    th[i].join();
  }
  LOG(INFO) << "finished check";

  return 0;
}
