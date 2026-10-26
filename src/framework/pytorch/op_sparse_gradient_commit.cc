#include "framework/pytorch/op_sparse_gradient_commit.h"
#include "framework/pytorch/op_internal.h"

namespace recstore::framework {
namespace {
int64_t sparse_gradient_commit_async(
    const std::string& table,
    const torch::Tensor& ids,
    const torch::Tensor& grads,
    int64_t session,
    int64_t step,
    int64_t rank,
    int64_t world,
    double lr,
    int64_t divisor) {
  TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == torch::kInt64 &&
                  grads.dim() == 2 && grads.scalar_type() == torch::kFloat32 &&
                  ids.numel() == grads.size(0),
              "Invalid sparse gradients");
  auto h = MakeSparseGradientCommitHeader(
      session, step, rank, world, grads.size(1), lr, divisor, 0);
  auto keys       = ids.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(ids)
                                  : ids.contiguous().clone();
  auto keys_ready = RecordCudaCompletionForTensor(ids);
  auto values = grads.is_cuda() ? StageCudaTensorToPinnedCpuAsyncNoCast(grads)
                                : grads.contiguous().clone();
  auto grads_ready = RecordCudaCompletionForTensor(grads);
  auto client      = GetConcreteKVClientOp();
  return client->SparseGradientCommitAsync(
      table,
      h,
      ToRecTensor(keys, base::DataType::UINT64),
      ToRecTensor(values, base::DataType::FLOAT32),
      [keys, values, ids, grads, keys_ready, grads_ready] {
        keys_ready();
        grads_ready();
      });
}

void sparse_gradient_commit_wait(int64_t handle) {
  GetConcreteKVClientOp()->WaitSparseGradientCommit(handle);
}

void sparse_gradient_commit_close(
    const std::string& table,
    int64_t session,
    int64_t step,
    int64_t rank,
    int64_t world,
    int64_t dim,
    double lr,
    int64_t divisor,
    bool abort) {
  auto h = MakeSparseGradientCommitHeader(
      session, step, rank, world, dim, lr, divisor, abort ? 3 : 2);
  GetConcreteKVClientOp()->CloseSparseGradientCommit(table, h);
}
} // namespace
void RegisterSparseGradientCommitOps(torch::Library& m) {
  m.def("sparse_gradient_commit_async", sparse_gradient_commit_async);
  m.def("sparse_gradient_commit_wait", sparse_gradient_commit_wait);
  m.def("sparse_gradient_commit_close", sparse_gradient_commit_close);
}
} // namespace recstore::framework
