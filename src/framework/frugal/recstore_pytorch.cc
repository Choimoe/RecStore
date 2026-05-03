#include <ATen/cuda/CUDAContext.h>
#include <stdlib.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "IPCTensor.h"
#include "base/init.h"
#include "base/timer.h"
#include "base/zipf.h"
#include "recstore_gpu_cache.h"
#include "storage/gpu_cache/gpu_cache_nohash.h"
#include "torch_utils.h"

namespace recstore {
void RegisterKGCacheController(torch::Library& m);

void merge_op(at::Tensor merge_dst,
              const at::Tensor retrieved,
              const at::Tensor missing_index);

void uva_cache_query_op(
    at::Tensor merge_dst,
    const at::Tensor id_tensor,
    const at::Tensor hbm_tensor,
    const at::Tensor dram_tensor,
    const long cached_start_key,
    const long cached_end_key);

// renumbering all_rank_hotsets from 0 to sum len(all_rank_hotsets)
void ConstructRenumberingDict(torch::Tensor renumbering_dict,
                              int64_t nr_world,
                              std::vector<torch::Tensor> all_rank_hotsets) {
  int64_t n_entities             = renumbering_dict.size(0);
  auto renumbering_dict_accessor = renumbering_dict.accessor<int64_t, 1>();
  std::vector<int64_t> start_ids(nr_world + 1, 0);
  // 计算每个 rank 的起始 ID
  for (int rank = 0; rank < nr_world; rank++) {
    start_ids[rank + 1] = start_ids[rank] + all_rank_hotsets[rank].numel();
  }

// 并行处理每个 rank 的 hotset
#pragma omp parallel for schedule(dynamic)
  for (int rank = 0; rank < nr_world; rank++) {
    torch::Tensor rank_hotset     = all_rank_hotsets[rank];
    int64_t* rank_hotset_data_ptr = rank_hotset.data_ptr<int64_t>();
    int64_t numel                 = rank_hotset.numel();
    int64_t start_id              = start_ids[rank];

    for (int64_t i = 0; i < numel; i++) {
      int64_t each                    = rank_hotset_data_ptr[i];
      renumbering_dict_accessor[each] = start_id + i;
    }
  }

  // 处理剩余的实体
  int64_t remaining_start_id = start_ids[nr_world];
#pragma omp parallel for schedule(dynamic)
  for (int64_t i = 0; i < n_entities; i++) {
    if (renumbering_dict_accessor[i] == -1) {
      renumbering_dict_accessor[i] =
          __sync_fetch_and_add(&remaining_start_id, 1);
    }
  }

  CHECK_EQ(remaining_start_id, n_entities);
}

void init_recstore_runtime() {
  putenv("GLOG_logtostderr=1");
  xmh::Reporter::StartReportThread();
}

class ZipfianTorchFiller : public torch::CustomClassHolder {
public:
  ZipfianTorchFiller(int64_t capacity, double zipf_theta) {
    int tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100;
    mehcached_zipf_init(
        &state, capacity, zipf_theta, (rdtsc() & 0x0000ffffffffffffull) ^ tid);
  }

  void fillArray(int64_t* buffer, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
      buffer[i] = mehcached_zipf_next(&state) + 1;
      // buffer[i] = base::GetHash(mehcached_zipf_next(&state) + 1) % capacity;
    }
  }

  void fillArrayTorch(torch::Tensor buffer) {
    int64_t count       = buffer.numel();
    int64_t* buffer_ptr = buffer.data_ptr<int64_t>();
    fillArray(buffer_ptr, count);
  }

private:
  static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
  }
  struct zipf_gen_state state;
};

void RegisterIPCBarrier(torch::Library& m);

torch::optional<torch::Tensor>
NarrowShapeTensor(torch::Tensor base,
                  const at::IntArrayRef shape,
                  const at::ScalarType dtype) {
  int64_t total_bytes    = base.numel() * base.element_size();
  int64_t required_bytes = at::elementSize(dtype) * TensorUtil::numel(shape);
  CHECK_GE(total_bytes, required_bytes);
  auto tensor = torch::from_blob(
      base.data_ptr(),
      shape,
      torch::TensorOptions().dtype(dtype).device(base.device()));
  return tensor;
}

TORCH_LIBRARY(librecstore_pytorch, m) {
  m.def("NarrowShapeTensor", &NarrowShapeTensor);

  m.class_<CacheQueryResult>("CacheQueryResult")
      .def("__str__", &CacheQueryResult::__repr__)
      .def_property("values", &CacheQueryResult::values)
      .def_property("missing_index", &CacheQueryResult::missing_index)
      .def_property("missing_keys", &CacheQueryResult::missing_keys);

  m.class_<GpuCache>("GpuCache")
      .def(torch::init<int64_t, int64_t>())
      .def("Query", &GpuCache::Query)
      .def("Replace", &GpuCache::Replace);

  m.class_<ZipfianTorchFiller>("ZipfianTorchFiller")
      .def(torch::init<int64_t, double>())
      .def("fillArrayTorch", &ZipfianTorchFiller::fillArrayTorch);

  m.def("merge_op", &merge_op);
  m.def("uva_cache_query_op", &uva_cache_query_op);
  m.def("init_recstore_runtime", &init_recstore_runtime);

  m.def("construct_renumbering_dict_op", &ConstructRenumberingDict);

  m.class_<GPUCacheWithNoHashTorch>("GpuCacheWithNoHash")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t>());
  RegisterIPCTensorFactory(m);
  RegisterKGCacheController(m);
  RegisterIPCBarrier(m);
}

} // namespace recstore
