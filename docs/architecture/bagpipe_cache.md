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

## Prescanned execution

BagPipe has one public plugin name, `bagpipe`. In the DLRM runner, enable
precomputed collective layouts with:

```text
--read-mode bagpipe --optimization-prescan
```

Without prescan, BagPipe uses the dynamic C++ cache described above. With
prescan, it scans the configured training steps, loads the complete touched
working set into a compact FP32 tensor on each rank, and communicates gradients
using fixed per-step layouts. It retains every row until final writeback, so a
rank's first use includes all preceding updates to that row. The internal
resident controller shares the same plugin, read path, and sparse optimizer.
No separate `bagpipe_planned` plugin is registered.

The runner exchanges the prescanned fused IDs once and builds each step's
global active-row layout and local index maps. Each step all-reduces FP32
gradient values and applies their sum divided by world size, matching
TorchRec/DDP global mean gradient semantics. The pending aggregate is applied
before the next lookup, and the final update is drained before validation.
There is no gradient compression or stale-read allowance. After successful
training, all ranks align and rank 0 writes the authoritative values to the PS.
Writeback errors propagate; error cleanup does not enter new collectives.

Plans require an exactly repeatable batch sequence, fused tables, consistent
feature order, and exactly one ID per feature per sample. Enqueue and lookup
validate actual IDs and bag lengths, including changes that preserve unique-ID
counts. A different shuffle or training schedule requires a new prescan. Each
rank must issue one fused sparse optimizer update per step; empty optimizer
traces and uneven rank termination are outside the runner's supported contract.

The full global working set must fit on **each** GPU, together with plan metadata
and training tensors. The row limit is `cache_capacity * 64`, matching the C++
cache's capacity unit. Capacity or schedule violations fail explicitly; they
do not silently switch controllers. This resident working-set optimization does
not replace eviction or dynamic cache coherence for out-of-core workloads.

CPU layout construction temporarily uses one PyTorch CPU thread because small
per-batch sorts and searches suffer substantial thread-pool overhead. The
previous thread budget is restored even if planning fails; training retains
its original setting. Startup statistics separate metadata exchange, global
unique construction, layout construction, PS loading, device copies, and
rank-ready waiting. Report these startup costs, GPU memory use, and final PS
writeback separately from steady training throughput.
