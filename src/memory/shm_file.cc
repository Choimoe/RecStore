#include "shm_file.h"

#include <algorithm>
#include <iostream>
#ifdef __linux__
#  include <sys/stat.h>
#endif

namespace base {
namespace {

constexpr const char* kFsDaxType     = "FS_DAX";
constexpr const char* kAnonyDramType = "ANONY_DRAM";
constexpr const char* kDevDaxType    = "DEV_DAX";

json BuildShmFileConfig(
    const std::string& type, const std::string& filename, int64 size) {
  return json{{"type", type}, {"filename", filename}, {"size", size}};
}

} // namespace

std::unique_ptr<ShmFile> ShmFile::New(const json& config) {
  CHECK(config.contains("type")) << "ShmFile config requires type";
  const std::string type = config.at("type").get<std::string>();
  using SF               = base::Factory<ShmFile, const json&>;
  std::unique_ptr<ShmFile> shm_file(SF::NewInstance(type, config));
  if (!shm_file->Initialize(config)) {
    return nullptr;
  }
  return shm_file;
}

json ShmFile::ConfigForMedium(
    const std::string& medium, const std::string& filename, int64 size) {
  if (medium == "DRAM")
    return BuildShmFileConfig(kAnonyDramType, filename, size);
  // return BuildShmFileConfig(kDevDaxType, filename, size);
  if (medium == "SSD")
    return BuildShmFileConfig(kFsDaxType, filename, size);
  LOG(FATAL) << "Unsupported ShmFile medium: " << medium;
  return json();
}

std::string ShmFile::ConfigFilename(const json& config) {
  CHECK(config.contains("filename")) << "ShmFile config requires filename";
  return config.at("filename").get<std::string>();
}

int64 ShmFile::ConfigSize(const json& config) {
  CHECK(config.contains("size")) << "ShmFile config requires size";
  return config.at("size").get<int64>();
}

bool FsDaxShmFile::Initialize(const json& config) {
  Clear();

  const std::string filename = ConfigFilename(config);
  const int64 size           = ConfigSize(config);

  std::error_code ec;
  bool file_exists = fs::exists(filename, ec);
  if (ec) {
    LOG(ERROR) << "fs::exists failed for " << filename << ": " << ec.message();
    return false;
  }

  if (!file_exists) {
    fs::create_directories(fs::path(filename).parent_path(), ec);
    if (ec) {
      LOG(ERROR) << "fs::create_directories failed: " << ec.message();
      return false;
    }
    LOG(INFO) << "Create ShmFile: " << filename << ", size: " << size;
  }

  fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) {
    LOG(ERROR) << "Failed to open file " << filename << ": " << strerror(errno);
    return false;
  }

  if (!file_exists) {
    int ret = posix_fallocate(fd_, 0, size);
    if (ret != 0) {
      LOG(ERROR) << "posix_fallocate failed: " << strerror(ret);
      close(fd_);
      fd_ = -1;
      return false;
    }
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    LOG(ERROR) << "fstat failed: " << strerror(errno);
    close(fd_);
    fd_ = -1;
    return false;
  }
  size_ = st.st_size;

  if (size_ != size) {
    LOG(ERROR) << "Size Error: " << size_ << " vs " << size;
    close(fd_);
    fd_   = -1;
    size_ = 0;
    return false;
  }

  data_ = reinterpret_cast<char*>(
      mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
  CHECK_NE(data_, MAP_FAILED) << "map failed";

  filename_ = filename;
  LOG(INFO) << "mmap shm file: " << filename << ", size: " << size_
            << ", fd: " << fd_;
  return true;
}

void FsDaxShmFile::Clear() {
  if (fd_ >= 0) {
    LOG(INFO) << "ummap shm file: " << filename_ << ", size: " << size_
              << ", fd: " << fd_;
    munmap(data_, size_);
    close(fd_);
  }
  filename_.clear();
  data_ = NULL;
  size_ = 0;
  fd_   = -1;
}

bool AnonyDramShmFile::Initialize(const json& config) {
  Clear();

  const std::string filename = ConfigFilename(config);
  const int64 size           = ConfigSize(config);

  data_ = reinterpret_cast<char*>(mmap(
      nullptr,
      size,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0));
  CHECK_NE(data_, MAP_FAILED) << "anonymous DRAM mmap failed";
  filename_ = filename;
  size_     = size;
  LOG(INFO) << "ShmFile, anonymous DRAM mmap, filename:" << filename
            << ", size:" << size;
  return true;
}

void AnonyDramShmFile::Clear() {
  if (data_ != NULL) {
    LOG(INFO) << "ummap anonymous DRAM: " << filename_ << ", size: " << size_;
    munmap(data_, size_);
  }
  filename_.clear();
  data_ = NULL;
  size_ = 0;
}

bool DevDaxShmFile::Initialize(const json& config) {
  Clear();

  const std::string filename = ConfigFilename(config);
  const int64 size           = ConfigSize(config);

  {
    static std::mutex m;
    std::lock_guard<std::mutex> _(m);
    data_ =
        (char*)PMMmapRegisterCenter::GetInstance()->Register(filename, size);
    filename_ = filename;
  }

  std::error_code ec;
  if (!fs::exists(filename, ec)) {
    fs::create_directories(fs::path(filename).parent_path(), ec);
    LOG(INFO) << "Create ShmFile: " << filename << ", size: " << size;
    std::ofstream output(filename);
    output.write("a", 1);
    output.close();
  }
  size_ = size;
  return true;
}

void DevDaxShmFile::Clear() {
  filename_.clear();
  data_ = NULL;
  size_ = 0;
}

FACTORY_REGISTER(ShmFile, FS_DAX, FsDaxShmFile, const json&);
FACTORY_REGISTER(ShmFile, ANONY_DRAM, AnonyDramShmFile, const json&);
FACTORY_REGISTER(ShmFile, DEV_DAX, DevDaxShmFile, const json&);

} // namespace base
