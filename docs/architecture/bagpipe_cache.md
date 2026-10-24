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
maps. Training communicates only FP32 gradient values with an all-reduce.
SGD applies their sum divided by world size, matching TorchRec/DDP global mean
gradient semantics. The pending aggregate is applied before the next lookup;
the final update is completed before validation or shutdown. There is no
gradient compression or stale-read allowance. After successful training, all
ranks align and rank 0 writes the authoritative values to the PS. Writeback
errors propagate; error cleanup does not enter new collectives.

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
