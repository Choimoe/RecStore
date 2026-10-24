# BagPipe GPU cache ownership

BagPipe mode treats the C++ GPU cache as owned storage. The Python controller
may plan prefetches, but it never assumes that a replace-style insert succeeded
or that an invalidation removed a row.

## State ownership

The C++ hash table is the residency authority. The Python flat tensors are a
mirror that is updated only from authoritative C++ results:

- `prefill_gpu_cache_no_evict` returns a per-key insert mask. Only true rows
  are marked cached.
- `gpu_cache_lookup_flat_no_evict` returns values and a post-fill residency
  mask. Failed inserts remain uncached and are retried later.
- `invalidate_gpu_cache_with_mask` returns a per-key removal mask. Only rows
  reported as removed are cleared from the Python mirror.
- `contains_gpu_cache` validates a policy all-hit decision immediately before
  the lock-free `gpu_cache_lookup_flat_assuming_hits` path.
- `get_gpu_cache_generation` changes whenever the C++ cache is reset. BagPipe
  fails loudly if another component resets its owned cache.

The mirror therefore has one writer protocol:

```text
C++ no-evict insert success -> cached=true, TTL refresh
C++ lookup fill success    -> cached=true, TTL refresh
C++ removal success        -> cached=false, dirty=false, TTL=0
C++ insert/removal failure -> previous state is retained
```

## Eviction

Only BagPipe evicts rows. C++ no-evict fills never replace a resident key.
Python chooses expired or low-TTL candidates, reads dirty values for PS
writeback, and invalidates them explicitly. Bookkeeping changes only for keys
that the C++ removal mask confirms. If a no-evict fill fails because hash sets
are saturated, the next cleanup retires a broader low-TTL slice and retries.

This mirrors upstream BagPipe's `BagCache` invariant: insertion fills known
empty storage, eviction is explicit, and the mapping and occupancy state change
together. RecStore keeps the physical slots inside the C++ set-associative hash
table, so the Python mirror is keyed by embedding ID rather than by local slot.

## Failure policy

BagPipe requires the authoritative APIs above. Missing APIs are configuration
errors, not reasons to fall back to replace-style fills: a fallback could evict
a key without updating the mirror. An external cache reset is also a hard error
because dirty local updates cannot be reconstructed safely after the reset.

## Opt-in planned resident mode

The DLRM runner also supports `--optimization-plugin bagpipe_planned` with
`--optimization-prescan --optimization-lookahead 4`. This mode stores the
complete planned working set in a compact FP32 tensor on every rank, loaded
from the PS before training. It bypasses the C++ hash cache and retains every
row until final writeback; a rank's first use therefore includes all preceding
updates to that row.

The runner prescans exactly the configured training steps, exchanges their
fused IDs once, and builds each step's global active-row layout and local index
maps. Training communicates only full-precision FP32 gradient values through
the configured sparse collective: all-reduce by default, or the opt-in
all-gather layout described below. SGD applies their sum divided by world size,
matching TorchRec/DDP global mean gradient semantics. The pending aggregate is
applied before the next lookup; the final update is completed before validation
or shutdown. There is no gradient compression or stale-read allowance. After
successful training, all ranks align and rank 0 writes the authoritative values
to the PS. Writeback errors propagate; error cleanup does not enter new
collectives.

Plans require a fixed batch schedule, fused tables, consistent feature order,
and exactly one ID per feature per sample. Enqueue and lookup validate actual
IDs and bag lengths, including changes that preserve unique-ID counts. A
different shuffle or training schedule requires a new prescan. Each rank must
issue one fused sparse optimizer update per step; empty optimizer traces and
uneven rank termination are outside the runner's supported contract.

The full global working set must fit on **each** GPU, together with the stored
batch plans and training tensors. The configured row limit retains the existing
capacity unit: `cache_capacity * 64` rows. Exceeding it fails explicitly.
This is a resident working-set optimization, not an out-of-core replacement
for eviction or dynamic cache coherence. Report prescan/startup cost and GPU
memory use separately from steady-state throughput.

CPU layout construction temporarily uses one PyTorch CPU thread because small
per-batch sorts and searches suffer substantial thread-pool overhead. The
previous thread budget is restored even if planning fails; training retains
its original setting. Startup statistics separate metadata exchange, global
unique construction, layout construction, PS loading, device copies, and
rank-ready waiting.

### Planned training pipeline

Plans store lookup and gradient indices in the dense model's
`[batch, feature, dimension]` order. Lookup produces a contiguous `KeyedTensor`
whose values can be viewed directly by DLRM. `update_pooled_grads` consumes the
corresponding pooled FP32 gradient without reconstructing per-feature autograd
graphs. The generic keyed-tensor reshape keeps its fallback for reordered,
subset, or unequal-width features.

In eager planned DLRM, a sparse-input autograd hook submits the sparse
all-reduce as soon as that gradient is available. Both forward and backward
run inside `DDP.no_sync()`. A fixed contiguous dense-gradient buffer then
performs one dense all-reduce. Distributed planned runs create separate
all-rank NCCL process groups for sparse collectives and for the dense
reduction. The sparse controller and gather-SGD submit on the sparse group;
the contiguous dense buffer reduces on the dense group. Both are submitted in
a fixed sparse-then-dense order after backward, but the independent
communicators allow the transfers to make progress concurrently. Each
collective owns separate input and output buffers, and both reductions are
joined before the next step consumes their results. Dense gradients are
divided before SUM, matching DDP, and sparse SGD divides the summed gradient
by world size. Every trainable dense parameter must participate; missing
gradients or replaced gradient views fail explicitly. This path changes
submission order, not the sequence of optimizer updates.

`--optimization-cuda-graphs` captures resident lookup, DLRM forward/backward,
and FP32 pooled-gradient packing for each batch size in the plan, including
the final short batch. Each replay copies the current plan's lookup and payload
indices into fixed buffers. The graph reads the current resident weights
directly, avoiding an intermediate keyed tensor and sparse-input copy. It packs
gradients into a fixed-capacity buffer; only the step's actual row count is
submitted to the collective. Static storage is reused, while communication and
SGD remain outside capture. The next prefill must consume the pending payload
and apply its update before any graph can reuse that storage.
Capture warmup performs no optimizer steps and verifies that parameters,
buffers, and RNG state remain unchanged. The graph path broadcasts initial
dense parameters and buffers directly and never constructs DDP, whose backward
hooks remain attached even after unwrapping. Before the next graph replay,
the previous dense SGD and sparse update are ordered on the training stream.
All arithmetic remains FP32 with the existing TF32 settings. TorchRec DLRM
also accepts `--optimization-cuda-graphs` in replicated training mode, sharing
the same dense graph and contiguous gradient reduction. Its embedding lookup
and sparse backward remain in TorchRec. Use this stronger baseline when
separating generic dense-execution gains from planned embedding gains.

The planned graph reports `planned_graph_ms` for lookup, dense forward/backward,
and sparse packing together, plus `gradient_communication_ms`. TorchRec's
graph reports the narrower `dense_graph_ms`. Planned eager-only forward,
backward, and dense-compute columns are unavailable, not evidence of zero
compute; the lifecycle record marks this explicitly. Compare complete training-step wall
time including synchronization, barriers, and the final pending sparse update.
Graph capture, prescan, and final PS writeback are separate lifecycle costs.

`--optimization-sparse-collective all_gather` is an experimental alternative.
It sends padded rank-local unique-row gradients, then combines them in fixed
rank order before the same single SGD update. It can reduce network payload
when rank-local row sets overlap little, but adds merge kernels and output
storage; less traffic does not guarantee higher end-to-end throughput.
`all_reduce` remains the default. Both modes communicate full FP32 values and
apply every gradient; neither uses quantization, dropped rows, or stale reads.
On the 100 Gb/s RoCE test fabric, four NCCL channels reduced collective tail
latency; the benchmark applies the same channel count to both lanes. Channel
count can change NCCL's FP32 reduction association at the ULP level, so rerun
loss and final-weight alignment whenever it changes. It does not introduce
quantization, dropped rows, or any intentional numerical approximation.
