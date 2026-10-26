# BagPipe with PS-owned parameters

Both default (online) and planned (prescan) execution use the same bounded
cache controller. The PS storage backend owns embedding values and applies
SGD each step. Its configured DRAM/SSD tiers remain independent of GPU cache
capacity. No full-table or full-run embedding working set is loaded on GPUs.

## Execution modes

`--read-mode bagpipe` enables the runtime. `--optimization-prescan` builds
CPU access plans before training; without it, ranks exchange batch IDs online.
Both choices support either gradient aggregation mode:

| Option | Aggregation | PS update |
| --- | --- | --- |
| `--bagpipe-aggregation collective` | Phase 1: all-reduce the current step's active-row gradients, divide by world size | One designated rank submits each row; PS commits after all rank markers |
| `--bagpipe-aggregation ps` (default) | Phase 2: send local unique-row gradients to their PS shard | PS sums contributions in rank/chunk order, divides by world size, and applies SGD once |

Phase 2 has no trainer-to-trainer sparse-gradient collective. Dense DDP
collectives and ID metadata exchange are still present. Prescan exchanges IDs
once and retains plans in CPU memory; online mode exchanges IDs each batch.

Example additions to an existing RDMA training command:

```sh
--ps-type RDMA --read-mode bagpipe --optimization-prescan \
  --optimization-cache-capacity 8192 --bagpipe-aggregation ps
```

Keep the existing RDMA deployment, shard routing, and launcher configuration.
The public rs_demo CLI selects its local backend for a single-node deployment;
`--ps-type RDMA` alone does not override that single-node launcher policy.

## Memory and cache coherence

`cache_capacity` retains its existing unit of 64 rows. Resident GPU value
storage is exactly `cache_capacity * 64 * embedding_dim * 4` bytes. CPU LRU
metadata maps keys to slots. The current batch and bounded prepared-batch
queue use additional temporary tensors; a batch larger than the cache still
runs by gathering into its own batch tensor and admitting only bounded rows.
Full-run plans consume CPU memory proportional to the access trace.

The framework-independent policy lives in
`src/optimization/bagpipe/cache_metadata.h` as `CacheMetadata`. It batches LRU
queries, admission, invalidation, and previous-update tracking using only the
C++ standard library. Input arrays are borrowed during each call; results are
ordinary vectors. It owns no tensors, embedding values, PS clients, or device
state. `test_bagpipe_metadata` exercises this policy without linking Torch.

`src/framework/pytorch/cache/bagpipe_bindings.cc` validates tensor dtype/device/shape
and adapts these arrays and results to PyTorch. The Python-visible class remains
`BagPipeCacheMetadata`; the controller retains PyTorch planning, tensor gathers,
autograd integration, and communication scheduling. Python exchanges compact
position and slot tensors rather than visiting every row. Rebuild
`recstore_torch_ops` when updating this runtime.

Admission returns only the final source row for each destination slot, so an
oversized batch never issues duplicate GPU scatter destinations. The controller
serializes metadata operations with commit completion.

After a PS commit, a rank updates locally cached rows that only it accessed
using its local gradient divided by world size. Rows modified by another rank
are invalidated and fetched on demand. In phase 1 the aggregated gradients are
already available, so all affected resident rows can be updated locally.
Eviction discards clean cache rows; it never overwrites PS values.

Plans record the previous global update step for each local row. A prefetch
is issued only when every preceding update for that row has committed. Other
misses wait until consumption. The next forward waits for a pending commit
only when its rows depend on that step; independent forward work can overlap
PS submission. The next sparse update always drains its predecessor.

## RDMA submission

Gradients are copied to pinned CPU buffers asynchronously. CUDA completion
events are recorded on the producer stream and waited by a native worker
before reading the buffers. IDs already held in the CPU plan avoid a GPU
round trip. The worker owns the tensors until staging and submission finish.

The RDMA adapter gathers gradients directly into bounded packets, submits
chunks round-robin across shards, and allows multiple RC writes before waiting
for ACKs. The outstanding window respects the configured QP/slot capacity.
All shards receive rank markers, including shards receiving zero rows.

GET windows and training submissions share RC slots. Before reusing slots,
the adapter materializes outstanding prefetch results in their CPU buffers
while keeping their public handles valid. Window ownership is serialized
within one adapter; GPU computation and PS aggregation across ranks remain
asynchronous. This avoids exhausting slots when several future batches are
queued. It is not a general concurrent scheduler for mixing arbitrary raw
RDMA requests and BagPipe training on the same client.

## Framework adapter boundary

`src/framework/pytorch/` adapts framework tensors to the common op API. It
owns dtype/shape checks, contiguous and pinned buffers, CUDA stream events,
Tensor result wrapping, and operator registration. It does not own commit
polling, asynchronous work handles, cache bypass thresholds, or cache
coherence decisions. The former `op_bagpipe.cc` has been removed.

The execution paths are split by responsibility:

- `framework/op.{h,cc}` exposes sparse commit submission/wait/close and cached
  update/write operations using `RecTensor` views. Successful table initialization
  and checkpoint restore also clear cached values through this common API.
- `framework/common/sparse_gradient_commit_runtime.h` owns asynchronous work,
  input readiness, committed-step polling, timeout, and peer abort. Adapters
  supply a readiness callback that retains source/staging buffers until the
  operation finishes; no Torch types enter the runtime.
- `framework/cache/cache_aware_op.h` owns the shared query/read-misses/fill/
  scatter pipeline, low-hit bypass, fallback, and invalidation after transport
  completion. Its device actions are callbacks; the same policy is exercised
  by C++ tests without Torch or CUDA.
- `framework/pytorch/cache/tensor_adapter.cc` supplies Tensor allocation and device
  operations for those callbacks. `emb_read`, `local_lookup_flat`, and
  `gpu_cache_lookup_flat` share the common pipeline. Existing GPU cache
  primitives are adapted in `framework/pytorch/cache/gpu_cache_adapter.cu`.
- `framework/cache/gpu_embedding_cache.{h,cu}` owns the GPU cache lifecycle,
  cross-stream ordering, generation, residency operations, and cache SGD.
  Its public API accepts typed device pointers, sizes, a device ordinal, and
  a CUDA stream. Callers own the buffers and retain them through stream
  completion. Query returns a synchronized miss count; hit-only reads and
  ordinary mutations enqueue work. Clear/disable wait for the last operation
  before destroying cache storage on the host.
- Strict cache SGD groups duplicate keys with CUDA/Thrust and sums their
  gradients before updating each row once. If any requested key is missing,
  it invalidates the requested keys and returns false. Best-effort SGD keeps
  the existing asynchronous kernel and skips absent keys.
- Explicit enable/disable/reset also reset the calling thread's bypass
  state. Internal cache clearing preserves that state so fallback and bypass
  transitions remain owned by `cache_aware_op.h`.
- Local flat reads expose a common transport lease. The Tensor adapter retains
  borrowed host storage until its H2D stream callback completes. Generic host
  memory registration lives in `framework/gpu/host_memory.cu`.

Cache files are grouped under `framework/cache/` (shared implementation) and
`framework/pytorch/cache/` (Tensor adapters and registration, including BagPipe
metadata bindings). Host-memory utilities remain in `framework/gpu/`.
`framework_gpu_cache` has no Torch/Python include or link dependencies.

The Python operator names are unchanged by this extraction.
`test_framework_op_runtime` checks readiness/lifetime, timeout and abort,
cache bypass/fallback, and completion-before-invalidation without Torch.
`test_framework_gpu_cache` links only the CUDA cache and ordinary C++ test
dependencies, and exercises caller-owned buffers, duplicate-key SGD, partial
miss failure, cross-stream updates, and lifecycle reset without Torch.

## Sparse gradient commit protocol and failure behavior

`SparseGradientCommit` names the PS protocol for aggregating sparse gradients
and committing an ordered update. Its header, client, and coordinator live in
`src/ps/base/sparse_gradient_commit.h`; it is shared by PS transports and is
independent of BagPipe's cache policy and Torch. Python clients expose
`sparse_gradient_commit_async`, `sparse_gradient_commit_wait`, and
`sparse_gradient_commit_close`. The asynchronous handle completes after all
shards report the requested committed step, rather than just receipt of a chunk.

This replaces the earlier `TrainingStep` names throughout this unmerged feature,
including RPC methods. Update clients, servers, and the native extension together.
RDMA/SHM opcode values and the binary header layout are unchanged.

A session identifies one job on one fused table. Each packet carries session,
step, rank, world size, chunk index/count, dimension, and SGD learning rate.
The PS acknowledges a chunk immediately instead of blocking a server worker
for another rank. When all chunks from every rank arrive, it reduces the
step's gradients and updates storage once. Clients observe the minimum
committed step across shards. Empty-rank and all-empty-step markers advance
the same protocol. Conflicting pending retries and out-of-order steps fail;
retries of committed steps do not apply SGD again.

This provides ordered training visibility, not a durable distributed
transaction. A backend error may leave a partially applied update. The
session is then failed and must not be retried as a fresh optimizer update.
Restart/restore the job and PS from a consistent checkpoint after a failed
session. Rank loss produces an error/timeout, not partial-rank training.
Successful shutdown drains pending work and closes every rank's session.
Failure cleanup performs no new distributed collective.

## Supported contract and metrics

The runtime currently requires fused tables, FP32 gradients, constant-rate
SGD with matching PS/client learning rates, fixed feature order, one ID per
feature per sample, and one fused sparse update per step per rank. Empty ranks
participate. Uneven rank termination, mixed external writers, adaptive
optimizers, and elastic recovery are unsupported. Prescan requires the exact
same ID sequence during training.

Metrics include cache hits, demand/prefetch IDs, eviction/invalidation counts,
resident bytes, dependency waits skipped, commit-wait time, and gradient
payload bytes. Prescan startup and final shutdown costs are recorded
separately from training steps. Compare modes at the same cache capacity and
include final commit time; the former fully resident planned mode is not an
out-of-core baseline.

The old C++ cache controller remains in internal legacy modules for existing
callers/tests. The public BagPipe plugin and `PlannedBagPipeController` alias
use the bounded PS runtime described here.
