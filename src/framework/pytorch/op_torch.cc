// Single registration translation unit for the recstore_ops library.
//
// Each op family exposes a Register*Ops(torch::Library&) helper from its own
// translation unit; this file only wires them into TORCH_LIBRARY so the
// registration block stays in one place without becoming the merge point for
// every new op. Op families:
//   base     - basic read/write, PS configuration, table and checkpoint ops
//   flat     - local_shm flat fast-path lookup/update
//   prefetch - asynchronous prefetch ops
//   bagpipe  - GPU cache / BagPipe ops
//   profile  - profile and warmup queries

#include <torch/extension.h>

#include "framework/pytorch/op_bagpipe.h"
#include "framework/pytorch/op_base.h"
#include "framework/pytorch/op_flat.h"
#include "framework/pytorch/op_prefetch.h"
#include "framework/pytorch/op_profile.h"

namespace recstore {
namespace framework {

TORCH_LIBRARY(recstore_ops, m) {
  RegisterBaseOps(m);
  RegisterFlatOps(m);
  RegisterPrefetchOps(m);
  RegisterBagpipeOps(m);
  RegisterProfileOps(m);
}

} // namespace framework
} // namespace recstore
