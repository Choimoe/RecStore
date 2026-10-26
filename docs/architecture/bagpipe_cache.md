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

The native `BagPipeCacheMetadata` helper batches CPU LRU queries, admission,
invalidation, and last-update tracking. Python exchanges compact position and
slot tensors with it instead of visiting every row or updating counters once
per row. The helper owns metadata only; values remain in the bounded GPU
tensor and PS storage. Rebuild `recstore_torch_ops` when updating this runtime.
Admission returns only the final source row for each destination slot, so an
oversized batch never issues duplicate GPU scatter destinations. The
controller serializes metadata operations with commit completion.

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

## Commit protocol and failure behavior

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
