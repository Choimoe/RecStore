#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <fcntl.h> // for open()
#include <immintrin.h>
#include <sys/mman.h> // for mmap()
#include <sys/stat.h> // for mode constants
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unistd.h> // for close()

#include <cstring> // for memset
#include <memory>
#include <string>
#include <unordered_map>

#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/lock.h"
#include "base/log.h"
#include "base/math.h"
#include "base/mmap.h"
#include "base/pprint.h"
#include "base/sleep.h"

namespace recstore {

int64_t numel(const at::IntArrayRef shape);

enum IPCTensorType {
  kCPUIPCTensor = 0,
  kGPUIPCTensor = 1,
};

class IPCTensorMemoryHandle {
  static constexpr int kShapeDimMax = 3;

public:
  IPCTensorMemoryHandle(
      const std::string& name, at::IntArrayRef shape, at::ScalarType dtype)
      : dtype_(dtype) {
    type_ = kCPUIPCTensor;
    memset(shape_, -1, kShapeDimMax * sizeof(int));
    for (int i = 0; i < shape.size(); i++) {
      shape_[i] = shape[i];
    }
    shape_dim_ = shape.size();
    assert(shape_dim_ <= kShapeDimMax);
    assert(name.size() < 96);
    strcpy(name_, name.c_str());
    size_       = numel(shape_vec()) * c10::elementSize(dtype_);
    pid_        = getpid();
    sliced_end_ = -1;
  }

  IPCTensorMemoryHandle(
      const std::string& name,
      at::IntArrayRef shape,
      at::ScalarType dtype,
      int64_t dev_id,
      cudaIpcMemHandle_t memHandle,
      void* dev_ptr)
      : dtype_(dtype),
        dev_ptr_(dev_ptr),
        memHandle_(memHandle),
        dev_id_(dev_id) {
    type_ = kGPUIPCTensor;
    memset(shape_, -1, kShapeDimMax * sizeof(int));
    for (int i = 0; i < shape.size(); i++) {
      shape_[i] = shape[i];
    }
    shape_dim_ = shape.size();
    assert(shape_dim_ <= kShapeDimMax);
    assert(name.size() < 96);
    strcpy(name_, name.c_str());
    size_       = numel(shape_vec()) * c10::elementSize(dtype_);
    pid_        = getpid();
    sliced_end_ = -1;
  }

  std::vector<int64_t> shape_vec() const {
    std::vector<int64_t> ret;
    for (int i = 0; i < shape_dim_; i++) {
      ret.push_back(shape_[i]);
    }
    return ret;
  }

  int64_t WholeSizeInCPU() const {
    if (type_ == kCPUIPCTensor)
      return sizeof(IPCTensorMemoryHandle) + size_;
    else
      return sizeof(IPCTensorMemoryHandle);
  }

  static int64_t WholeSizeInCPU(
      IPCTensorType type, at::IntArrayRef shape, at::ScalarType dtype) {
    if (type == kCPUIPCTensor)
      return sizeof(IPCTensorMemoryHandle) +
             numel(shape) * c10::elementSize(dtype);
    else
      return sizeof(IPCTensorMemoryHandle);
  }

  void* GetTensorPtr() const {
    if (type_ == kCPUIPCTensor) {
      return (void*)host_ptr_;
    } else {
      return dev_ptr_;
    }
  }

  std::string __repr__() const {
    std::stringstream ss;
    if (type_ == kCPUIPCTensor)
      ss << "IPCTensor(" << name_ << ", " << shape_vec() << ", " << dtype_
         << ")";
    else if (type_ == kGPUIPCTensor)
      ss << "IPCGPUTensor(" << name_ << ", " << shape_vec() << ", " << dtype_
         << ", " << GetDeviceID() << ")";
    else
      assert(false);
    return ss.str();
  }

  auto CheckMagic() const { assert(magic_ == 0xdeadbeef); }

  auto GetDtype() const { return dtype_; }

  auto GetName() const { return name_; }

  int GetDeviceID() const {
    assert(type_ == kGPUIPCTensor);
    return dev_id_;
  }

  auto GetIPCType() const { return type_; }

  auto GetPID() const { return pid_; }

  cudaIpcMemHandle_t GetCUDAIPCMemHandle() const {
    assert(type_ == kGPUIPCTensor);
    return memHandle_;
  }

  void SetSlicedEnd(int64_t end) {
    // sliced_end_ = end;
    std::atomic<int64_t>* p = (std::atomic<int64_t>*)&sliced_end_;
    p->store(end);
  }

  int64_t GetSlicedEnd() const {
    // return sliced_end_;

    std::atomic<int64_t>* p = (std::atomic<int64_t>*)&sliced_end_;
    return p->load();
  }

private:
  union {
    struct {
      IPCTensorType type_;
      char name_[96];
      int shape_[kShapeDimMax];
      int shape_dim_;
      int64_t size_;
      at::ScalarType dtype_;
      int pid_;
      int64_t sliced_end_;
      const int magic_ = 0xdeadbeef;
    };
    char ____padding[4096];
  };
  union {
    struct {
      void* dev_ptr_;
      cudaIpcMemHandle_t memHandle_;
      int dev_id_;
    };
    struct {
      char host_ptr_[0];
    };
  };
};

class IPCMemory {
  static constexpr int kMaxRegTensorNum = 200;

  // static constexpr int64_t kShmSize = 200 * (1024 * 1024 * 1024LL);
  // static constexpr int64_t kShmSize = 50* (1024 * 1024 * 1024LL);
  static int64_t kShmSize;

  struct IPCShmRegion {
    IPCShmRegion() {}
    std::atomic<int64_t> accumulated_offset_;
    base::SpinLock lock_;
    int64_t malloced_offsets_[kMaxRegTensorNum];
    IPCTensorMemoryHandle handles[0];
  };

  static inline std::string getEnvVar(std::string const& key) {
    char* val = getenv(key.c_str());
    return val == NULL ? std::string("") : std::string(val);
  }

  IPCMemory() {
    // get environment variable from shell
    std::string shm_gb = getEnvVar("SHM_GB");
    if (shm_gb != "")
      kShmSize = std::stoi(shm_gb) * (1024 * 1024 * 1024LL);
    else
      kShmSize = DEFINED_SHM_GB * (1024 * 1024 * 1024LL);

    LOG(WARNING) << "IPC Memory use " << kShmSize / (1024 * 1024 * 1024LL)
                 << " GB";

    // system("touch /dev/shm/recstore_ipc_memory");
    // MemoryMapping::Options options =
    //     MemoryMapping::writable().setPrefault(true).setShared(true);
    // options.address = (void *)(0x100000000000);

    // mapping_ = new MemoryMapping("/dev/shm/recstore_ipc_memory",
    // 0,
    //                                     kShmSize, options);
    // header_ =
    //     new ((IPCShmRegion *)mapping_->writableRange().begin())
    //     IPCShmRegion();

    int fd = shm_open("recstore_ipc_memory", O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
      perror("shm_open");
      CHECK(fd != -1);
    }
    if (ftruncate(fd, kShmSize) == -1) {
      perror("ftruncate");
      close(fd);
      CHECK(0);
    }

    void* mapping = mmap(
        (void*)0x100000000000,
        kShmSize,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0);
    CHECK(mapping != MAP_FAILED);
    header_ = new ((IPCShmRegion*)mapping) IPCShmRegion();
  }

  ~IPCMemory() {
    // delete mapping_;
  }

public:
  static IPCMemory* GetInstance() {
    static IPCMemory instance;
    return &instance;
  }

  IPCTensorMemoryHandle* RegisterCPUMemory(
      std::string name, at::IntArrayRef shape, at::ScalarType dtype) {
    int64_t size            = numel(shape) * c10::elementSize(dtype);
    int64_t obj_size_in_shm = IPCTensorMemoryHandle::WholeSizeInCPU(
        IPCTensorType::kCPUIPCTensor, shape, dtype);

    // round up to 4KB
    obj_size_in_shm = MathUtil::round_up_to_int64(obj_size_in_shm, 4096);
    // FAA the atomic variable in the header

    // critical section start

    base::LockGuard<base::SpinLock> lock_guard(header_->lock_);

    int64_t offset = header_->accumulated_offset_.fetch_add(obj_size_in_shm);

    CHECK_GE(offset, 0);
    CHECK_LT(offset + obj_size_in_shm, kShmSize) << "increase kShmSize";
    auto* p =
        new ((char*)header_ + offset) IPCTensorMemoryHandle(name, shape, dtype);
    SetMallocedOffset(offset);
    p->CheckMagic();

    asm volatile("" ::: "memory");
    _mm_mfence();
    return p;
  }

  IPCTensorMemoryHandle* RegisterGPUMemory(
      std::string name,
      at::IntArrayRef shape,
      at::ScalarType dtype,
      int dev_id) {
    int64_t size = numel(shape) * c10::elementSize(dtype);
    nv::CudaDeviceRestorer _;
    cudaSetDevice(dev_id);

    void* d_ptr;
    cudaMalloc(&d_ptr, size);
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, d_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // printf("Put memhandle = %s\n", memHandle.reserved);

    int64_t obj_size_in_shm = IPCTensorMemoryHandle::WholeSizeInCPU(
        IPCTensorType::kGPUIPCTensor, shape, dtype);
    obj_size_in_shm = MathUtil::round_up_to_int64(obj_size_in_shm, 4096);

    // critical section start
    base::LockGuard<base::SpinLock> lock_guard(header_->lock_);
    // FAA the atomic variable in the header
    int64_t offset = header_->accumulated_offset_.fetch_add(obj_size_in_shm);

    CHECK_GE(offset, 0);
    assert(offset + obj_size_in_shm < kShmSize);
    CHECK_LT(offset + obj_size_in_shm, kShmSize);

    auto* p = new ((char*)header_ + offset)
        IPCTensorMemoryHandle(name, shape, dtype, dev_id, memHandle, d_ptr);
    SetMallocedOffset(offset);
    p->CheckMagic();

    asm volatile("" ::: "memory");
    _mm_mfence();
    return p;
  }

  IPCTensorMemoryHandle* GetHandle(const std::string& name) {
    asm volatile("" ::: "memory");
    _mm_mfence();

    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t offset = header_->malloced_offsets_[i];
      if (offset == 0) {
        break;
      }
      auto* p = (IPCTensorMemoryHandle*)((char*)header_ + offset);
      p->CheckMagic();
      if (strcmp(p->GetName(), name.c_str()) == 0) {
        return p;
      }
    }
    return nullptr;
  }

  void ListIPCTensors() {
    std::cout << "ListIPCTensors:\n";
    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t offset = header_->malloced_offsets_[i];
      if (offset == 0) {
        break;
      }
      auto* p = (IPCTensorMemoryHandle*)((char*)header_ + offset);
      p->CheckMagic();
      std::cout << p->__repr__() << std::endl;
    }
  }

  void ClearIPCMemory() {
    for (int i = 0; i < kMaxRegTensorNum; i++) {
      header_->malloced_offsets_[i] = 0;
    }
    header_->accumulated_offset_ =
        MathUtil::round_up_to_int64(sizeof(IPCShmRegion), 4096);

    CHECK_GE(header_->accumulated_offset_.load(), 0);
  }

private:
  void SetMallocedOffset(int64_t offset) {
    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t& malloced_offset = header_->malloced_offsets_[i];
      if (malloced_offset == 0) {
        malloced_offset = offset;
        return;
      }
    }
    LOG(FATAL)
        << "Too many IPCTensors registered, please increase <kMaxRegTensorNum>";
  }

private:
  // MemoryMapping *mapping_;
  IPCShmRegion* header_;
};

class SlicedTensor;

class IPCTensorFactory : public torch::CustomClassHolder {
public:
  static void ClearIPCMemory() { IPCMemory::GetInstance()->ClearIPCMemory(); }

  static torch::optional<torch::Tensor>
  NewIPCTensor(const std::string& name,
               const at::IntArrayRef shape,
               const at::ScalarType dtype) {
    auto handle = IPCMemory::GetInstance()->GetHandle(name);
    if (handle != nullptr) {
      LOG(WARNING) << "IPCTensor " << name << " already exists";
      return torch::nullopt;
      // CHECK_EQ(handle->shape_vec().size(), shape.size());
      // for (int i = 0; i < shape.size(); i++) {
      //   CHECK(handle->shape_vec()[i] == shape[i]);
      // }
      // CHECK(handle->GetDtype() == dtype);
      // assert(handle->GetIPCType() == kCPUIPCTensor);

      // auto tensor = torch::from_blob(
      //     handle->GetTensorPtr(), handle->shape_vec(),
      //     torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
      // return tensor;
    }

    int64_t size_in_bytes = numel(shape) * c10::elementSize(dtype);
    CHECK_NE(size_in_bytes, 0) << "malloc size = 0";

    handle = IPCMemory::GetInstance()->RegisterCPUMemory(name, shape, dtype);
    assert(handle->GetIPCType() == kCPUIPCTensor);

    LOG(WARNING) << "NewIPCTensor: " << name << " " << shape
                 << handle->GetTensorPtr() << " "
                 << base::PrettyPrintBytes(size_in_bytes);

    _mm_mfence();

    auto tensor = torch::from_blob(
        handle->GetTensorPtr(),
        handle->shape_vec(),
        torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
    return tensor;
  }

  static void ListIPCTensors() { IPCMemory::GetInstance()->ListIPCTensors(); }

  static torch::Tensor NewOrGetIPCTensor(const std::string& name,
                                         const at::IntArrayRef shape,
                                         const at::ScalarType dtype) {
    auto handle = IPCMemory::GetInstance()->GetHandle(name);
    if (handle != nullptr) {
      LOG(WARNING) << "IPCTensor " << name << " already exists";
      CHECK_EQ(handle->shape_vec().size(), shape.size());
      for (int i = 0; i < shape.size(); i++) {
        CHECK(handle->shape_vec()[i] == shape[i]);
      }
      CHECK(handle->GetDtype() == dtype);
      assert(handle->GetIPCType() == kCPUIPCTensor);

      auto tensor = torch::from_blob(
          handle->GetTensorPtr(),
          handle->shape_vec(),
          torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
      return tensor;
    }

    int64_t size_in_bytes = numel(shape) * c10::elementSize(dtype);
    CHECK_NE(size_in_bytes, 0) << "malloc size = 0";

    handle = IPCMemory::GetInstance()->RegisterCPUMemory(name, shape, dtype);
    assert(handle->GetIPCType() == kCPUIPCTensor);

    LOG(WARNING) << "NewIPCTensor: " << name << " " << shape
                 << handle->GetTensorPtr() << " "
                 << base::PrettyPrintBytes(size_in_bytes);

    _mm_mfence();

    auto tensor = torch::from_blob(
        handle->GetTensorPtr(),
        handle->shape_vec(),
        torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
    return tensor;
  }

  static torch::Tensor NewIPCGPUTensor(
      const std::string& name,
      const at::IntArrayRef shape,
      const at::ScalarType dtype,
      const int64_t dev_id) {
    _mm_mfence();
    auto handle = IPCMemory::GetInstance()->GetHandle(name);
    if (handle != nullptr) {
      LOG(WARNING) << "IPCTensor " << name << " already exists";
      CHECK_EQ(handle->shape_vec().size(), shape.size());
      for (int i = 0; i < shape.size(); i++) {
        CHECK(handle->shape_vec()[i] == shape[i]);
      }
      CHECK(handle->GetDtype() == dtype);
      CHECK(handle->GetDeviceID() == dev_id);
      assert(handle->GetIPCType() == kGPUIPCTensor);

      auto tensor = torch::from_blob(
          handle->GetTensorPtr(),
          handle->shape_vec(),
          torch::TensorOptions()
              .dtype(handle->GetDtype())
              .device(torch::kCUDA, dev_id));
      return tensor;
    }

    int64_t size_in_bytes = numel(shape) * c10::elementSize(dtype);

    LOG(WARNING) << "NewIPCGPUTensor: " << name << " " << shape
                 << "; dev=" << dev_id
                 << "; size=" << base::PrettyPrintBytes(size_in_bytes);
    CHECK_NE(size_in_bytes, 0) << "malloc size = 0";

    handle =
        IPCMemory::GetInstance()->RegisterGPUMemory(name, shape, dtype, dev_id);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    assert(handle->GetIPCType() == kGPUIPCTensor);
    auto tensor = torch::from_blob(
        handle->GetTensorPtr(),
        handle->shape_vec(),
        torch::TensorOptions()
            .dtype(handle->GetDtype())
            .device(torch::kCUDA, dev_id));
    return tensor;
  }

  static torch::Tensor GetIPCTensorFromHandle(IPCTensorMemoryHandle* handle) {
    _mm_mfence();
    handle->CheckMagic();
    if (handle->GetIPCType() == kGPUIPCTensor) {
      void* ptr;
      int current_pid = getpid();
      if (handle->GetPID() == current_pid) {
        // for the same process, we can directly use the pointer
        // dont use cudaIPCGetMemHandle, it will cause context error
        ptr = handle->GetTensorPtr();
      } else {
        cudaIpcOpenMemHandle(&ptr,
                             handle->GetCUDAIPCMemHandle(),
                             cudaIpcMemLazyEnablePeerAccess);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return torch::from_blob(
          ptr,
          handle->shape_vec(),
          torch::TensorOptions()
              .dtype(handle->GetDtype())
              .device(torch::kCUDA, handle->GetDeviceID()));

    } else {
      void* ptr = (void*)handle->GetTensorPtr();
      return torch::from_blob(
          ptr,
          handle->shape_vec(),
          torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
    }
  }

  static torch::Tensor FindIPCTensorFromName(const std::string& name) {
    _mm_mfence();
    IPCTensorMemoryHandle* handle = IPCMemory::GetInstance()->GetHandle(name);
    if (nullptr == handle) {
      LOG(FATAL) << "IPCTensor " << name << " not found";
      assert(0);
    }
    return GetIPCTensorFromHandle(handle);
  }

  static c10::intrusive_ptr<SlicedTensor>
  GetSlicedIPCTensorFromName(const std::string& name);

  static c10::intrusive_ptr<SlicedTensor> NewSlicedIPCGPUTensor(
      const std::string& name,
      const at::IntArrayRef shape,
      const at::ScalarType dtype,
      const int64_t dev_id);

  static c10::intrusive_ptr<SlicedTensor>
  NewSlicedIPCTensor(const std::string& name,
                     const at::IntArrayRef shape,
                     const at::ScalarType dtype);

private:
};

class SlicedTensor : public torch::CustomClassHolder {
public:
  SlicedTensor(IPCTensorMemoryHandle* handle) : handle_(handle) {}

  torch::Tensor GetSlicedTensor() const {
    int end     = handle_->GetSlicedEnd();
    auto tensor = IPCTensorFactory::GetIPCTensorFromHandle(handle_);
    return tensor.slice(0, 0, end);
  }

  std::string __repr__() const {
    std::stringstream ss;
    ss << "SlicedTensor(" << GetSlicedTensor() << ")";
    return ss.str();
  }

  void Copy_(torch::Tensor right, bool non_blocking) {
    int64_t end = right.sizes()[0];
    auto tensor = IPCTensorFactory::GetIPCTensorFromHandle(handle_);
    CHECK(tensor.sizes()[0] >= end)
        << "Please increase the space of pre-allocated tensor";
    handle_->SetSlicedEnd(end);
    tensor.slice(0, 0, end).copy_(right, non_blocking);
  }

  void SetSameShape4Debug(torch::Tensor right) {
    int64_t end = right.sizes()[0];
    auto tensor = IPCTensorFactory::GetIPCTensorFromHandle(handle_);
    CHECK(tensor.sizes()[0] >= end);
    handle_->SetSlicedEnd(end);
  }

  static std::vector<torch::Tensor> BatchConvertToTensors(
      const std::vector<c10::intrusive_ptr<SlicedTensor>> tensors) {
    nv::CudaDeviceRestorer _;
    std::vector<torch::Tensor> ret;
    for (auto each : tensors) {
      if (each->handle_->GetIPCType() == kGPUIPCTensor)
        CUDA_CHECK(cudaSetDevice(each->handle_->GetDeviceID()));
      ret.push_back(each->GetSlicedTensor());
    }
    return ret;
  }

private:
  IPCTensorMemoryHandle* handle_;
};

void RegisterIPCTensorFactory(torch::Library& m);

} // namespace recstore
