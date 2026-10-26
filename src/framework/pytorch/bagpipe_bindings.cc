#include <torch/extension.h>
#include "optimization/bagpipe/cache_metadata.h"

#include <torch/custom_class.h>
#include <algorithm>
#include <tuple>

namespace recstore::framework {
namespace {

// Tensor validation and conversion only; the policy lives in optimization/.
class BagPipeMetadataBinding : public torch::CustomClassHolder {
public:
  explicit BagPipeMetadataBinding(int64_t capacity) : metadata_(capacity) {}

  int64_t size() const { return metadata_.Size(); }

  torch::Tensor previous(
      int64_t step, const torch::Tensor& local, const torch::Tensor& active) {
    return Tensor(metadata_.Previous(
        step, Keys(local), local.numel(), Keys(active), active.numel()));
  }

  void clear_plan() { metadata_.ClearPlan(); }
  torch::Tensor updated_ids() const { return Tensor(metadata_.UpdatedIds()); }

  torch::Tensor prefetch(const torch::Tensor& ids,
                         const torch::Tensor& previous,
                         int64_t committed) const {
    TORCH_CHECK(ids.numel() == previous.numel(),
                "BagPipe version count mismatch");
    return Tensor(
        metadata_.Prefetch(Keys(ids), Keys(previous), ids.numel(), committed));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  lookup(const torch::Tensor& ids) {
    auto result = metadata_.Lookup(Keys(ids), ids.numel());
    return {Tensor(result.hit_positions),
            Tensor(result.hit_slots),
            Tensor(result.missing_positions)};
  }

  std::tuple<torch::Tensor, torch::Tensor, int64_t>
  commit(const torch::Tensor& active,
         const torch::Tensor& local,
         const torch::Tensor& exclusive,
         bool ps) {
    TORCH_CHECK(
        exclusive.device().is_cpu() && exclusive.dim() == 1 &&
            exclusive.scalar_type() == torch::kBool &&
            exclusive.is_contiguous() && exclusive.numel() == local.numel(),
        "BagPipe exclusive mask must match local IDs");
    auto result = metadata_.Commit(
        Keys(active),
        active.numel(),
        Keys(local),
        local.numel(),
        exclusive.data_ptr<bool>(),
        ps);
    return {Tensor(result.slots), Tensor(result.rows), result.invalidated};
  }

  std::tuple<torch::Tensor, torch::Tensor, int64_t>
  admit(const torch::Tensor& ids) {
    auto result = metadata_.Admit(Keys(ids), ids.numel());
    return {Tensor(result.slots), Tensor(result.rows), result.evicted};
  }

private:
  static const int64_t* Keys(const torch::Tensor& ids) {
    TORCH_CHECK(ids.device().is_cpu() && ids.dim() == 1 &&
                    ids.scalar_type() == torch::kInt64 && ids.is_contiguous(),
                "BagPipe metadata requires contiguous CPU int64 IDs");
    return ids.data_ptr<int64_t>();
  }

  static torch::Tensor Tensor(const std::vector<int64_t>& values) {
    auto result =
        torch::empty({static_cast<int64_t>(values.size())}, torch::kInt64);
    std::copy(values.begin(), values.end(), result.data_ptr<int64_t>());
    return result;
  }

  optimization::bagpipe::CacheMetadata metadata_;
};

} // namespace

void RegisterBagpipeMetadataOps(torch::Library& m) {
  m.class_<BagPipeMetadataBinding>("BagPipeCacheMetadata")
      .def(torch::init<int64_t>())
      .def("size", &BagPipeMetadataBinding::size)
      .def("previous", &BagPipeMetadataBinding::previous)
      .def("clear_plan", &BagPipeMetadataBinding::clear_plan)
      .def("updated_ids", &BagPipeMetadataBinding::updated_ids)
      .def("prefetch", &BagPipeMetadataBinding::prefetch)
      .def("lookup", &BagPipeMetadataBinding::lookup)
      .def("commit", &BagPipeMetadataBinding::commit)
      .def("admit", &BagPipeMetadataBinding::admit);
}

} // namespace recstore::framework
