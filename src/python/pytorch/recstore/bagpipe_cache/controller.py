"""BagPipeCacheController — the orchestrator combining all BagPipe mixins.

Manages GPU cache lifecycle with BagPipe-style TTL eviction and writeback.
This is a drop-in replacement for LookaheadPrefetcher when enable_bagpipe_cache
is set.  All cross-cutting logic (comm, prefetch, eviction, gradient update)
is provided by the corresponding mixins.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set, Tuple

import torch

from .comm import BagPipeCommMixin
from .eviction import BagPipeEvictionMixin
from .grads import BagPipeGradMixin
from .prefetch import BagPipePrefetchMixin
from .types import PrefetchSlot

logger = logging.getLogger(__name__)


class BagPipeCacheController(
    BagPipeCommMixin,
    BagPipePrefetchMixin,
    BagPipeEvictionMixin,
    BagPipeGradMixin,
):
    """Manages GPU cache lifecycle with BagPipe-style TTL eviction and writeback.

    Replaces :class:`LookaheadPrefetcher` when ``enable_bagpipe_cache`` is set.
    Works with the local_shm fast path + GPU cache enabled.
    """

    # nv_gpu_cache.hpp: SET_ASSOCIATIVITY=2, SLAB_SIZE=32 -> 64 slots/set
    _GPU_CACHE_SET_SLOTS = 64

    def __init__(
        self,
        embedding_module: Any,
        kv_client: Any,
        *,
        lookahead_value: int,
        cleanup_batch_proportion: float,
        cache_capacity: int,
        embedding_dim: int,
        fuse_k: int,
        table_offsets: Dict[str, int],
        master_table_name: str,
        device: torch.device,
        lr: float = 0.01,
        id_extractor: Callable[[Any], torch.Tensor],
        table_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self._base_lookahead = max(1, int(lookahead_value))
        self._dynamic_lookahead = self._base_lookahead
        self._max_lookahead = max(self._base_lookahead, 16)
        self._additive_increase = 1
        self._multiplicative_decrease = 0.5
        self._pressure_low = 0.70
        self._pressure_high = 0.90
        self._lookahead_adjust_interval = 10
        self.cleanup_batch_proportion = float(cleanup_batch_proportion)
        self.cleanup_interval = max(1, int(self.cleanup_batch_proportion * self._dynamic_lookahead))
        # The C++ gpu cache's capacity unit is a slab set; each set holds
        # SET_ASSOCIATIVITY(2) x SLAB_SIZE(32) = 64 entry slots
        # (nv_gpu_cache.hpp / nv_gpu_cache.cu num_slot_ computation). The
        # bookkeeping's capacity-pressure test must convert to entry
        # counts, otherwise capacity eviction triggers at 1/64 of the real
        # capacity and causes pointless writeback churn (measured 92.8K ->
        # 76.1K samples/s).
        self.cache_capacity = int(cache_capacity) * self._GPU_CACHE_SET_SLOTS
        self.embedding_dim = int(embedding_dim)
        self.fuse_k = int(fuse_k)
        self.table_offsets = dict(table_offsets)
        self.master_table_name = master_table_name
        self.device = device
        self.lr = float(lr)

        # Model-agnostic ID extractor (caller must provide)
        self._id_extractor = id_extractor

        # ---- Oracle tracking: flat tensors indexed by compact ID ----
        # The original BagPipe oracle makes every per-id decision on flat
        # numpy arrays (latest_tracker[table][ids] vectorized); this uses
        # isomorphic GPU tensors, eliminating the per-fid Python dict loop
        # (measured ~15ms in the enqueue path at 12K ids/step). Four
        # parallel tensors are indexed directly by compact id:
        #   _latest_dev  int32  batch number that last enqueued this id
        #   _ttl_dev     int32  expiry batch of the cache entry (0 = expired)
        #   _cached_dev  bool   resident in the GPU cache
        #   _dirty_dev   bool   local SGD applied, not yet written back to PS
        #
        # fused id = (table_idx << fuse_k) | row (fuse_k=30, 26 tables ->
        # nominal space ~2^35; a flat tensor over raw fused ids would OOM).
        # The actual occupancy is [t<<fuse_k, t<<fuse_k + cap_t) per table,
        # so ids map to a compact per-table offset:
        # compact = (table_idx << _bk_shift) | row, with
        # stride = next_pow2(max cap) and total space = num_tables << shift
        # (~6.8M; the four tensors together ~68MB).
        caps = [int(v) for v in (table_sizes or {}).values()] or [1 << 20]
        max_cap = max(caps)
        self._bk_stride = 1 << max(1, max_cap - 1).bit_length()
        self._bk_shift = self._bk_stride.bit_length() - 1
        num_tables = max(1, len(table_offsets))
        self._latest_dev: Optional[torch.Tensor] = None
        self._ttl_dev: Optional[torch.Tensor] = None
        self._cached_dev: Optional[torch.Tensor] = None
        self._dirty_dev: Optional[torch.Tensor] = None
        self._alloc_bookkeeping(num_tables << self._bk_shift)

        # ---- Lookahead buffer (future batch unique ID sets) ----
        self._lookahead_ids: Deque[Tuple[int, torch.Tensor]] = deque()
        self._next_enqueue_batch = 0
        self._current_batch = 0

        # ---- Prefetch pre-issue (opt 3, 10) ----
        # Per-batch prefetch handles pre-issued at enqueue time so that
        # wait_and_get at consume is near-instant (the PS had `lookahead`
        # steps to respond while the main stream ran dense compute).
        # Cap on concurrent in-flight pre-issued prefetches: each prefetch is
        # chunked by the RDMA client (1600 keys/RPC at dim=128, ×2 shards) and
        # every chunk holds an RC write slot until the batch is consumed —
        # one 12K-id prefetch ≈ 8 slots, so the 64-slot pool (2 shards ×
        # 32 QPs × 1 slot) supports only a handful concurrently.  The cap
        # does NOT shorten the pipeline: in steady state one handle is
        # retired (consumed) and one issued per step, so every batch is still
        # pre-issued at enqueue, lookahead steps ahead of its consumption.
        self._prefetch_handles: Dict[int, Optional[PrefetchSlot]] = {}
        self._prefetch_all_hits: Dict[int, bool] = {}
        self._prepared_ids: Dict[
            int, Optional[Tuple[torch.Tensor, torch.Tensor, int]]
        ] = {}
        self._max_inflight_prefetch = 4
        self._insert_failures = 0
        generation_getter = getattr(kv_client, "get_gpu_cache_generation", None)
        self._cache_generation = (
            int(generation_getter()) if callable(generation_getter) else None
        )
        # TTL margin beyond the last-seen batch: ids in this dataset recur
        # only every ~epoch_length batches, so a TTL of exactly "last use"
        # expires every entry between recurrences and the cache collapses to
        # the current window (measured: mean 1116 entries at capacity 160000,
        # i.e. every id re-fetched from the PS on each recurrence).  The
        # margin keeps entries resident across one recurrence gap; capacity
        # is protected by a high-watermark fallback eviction in cleanup().
        self._ttl_margin = 256
        self._capacity_watermark = 0.9
        # Per-cleanup cap on TTL evictions: prevents tens of thousands of
        # entries being evicted at once after a cold start, causing an
        # eviction spike (and a piling-up writeback queue). Excess entries
        # keep their old ttl and are naturally picked up by the next
        # cleanup.
        self._max_evict_per_cleanup = 4096

        # ---- Stats ----
        self._stats: Dict[str, float] = {}
        # Hot-path counters (accumulated several times per step) are held
        # as GPU scalar tensors, avoiding a mid-update_grads /
        # _preissue_prefetch .item() that would force-drain the device
        # queue; consume_stats drains them after timer.finish() (the
        # device drain point).
        self._hot_stats_dev: Dict[str, torch.Tensor] = {
            k: torch.zeros((), dtype=torch.float64, device=device)
            for k in (
                "bagpipe_no_sync_ids",
                "bagpipe_sync_now_ids",
                "bagpipe_unknown_aggregated_ids",
                "bagpipe_prefetch_skip_cached",
            )
        }
        self.reset_stats()

        # ---- Background cleanup thread ----
        self._cleanup_queue: queue.Queue = queue.Queue()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="bagpipe-cleanup"
        )
        self._cleanup_thread.start()

        # ---- Overlap state: async all_reduce on a separate CUDA stream ----
        # The aggregated-gradient all_reduce is issued on a side stream
        # (overlapping main-stream compute) and waited on at the next
        # prefill's barrier, applying in place into the cache.
        self._sync_later_stream: Optional[torch.cuda.Stream] = None
        self._eviction_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._sync_later_stream = torch.cuda.Stream(device=self.device)
            self._eviction_stream = torch.cuda.Stream(device=self.device)
        self._pending_sync_now_work = None
        self._pending_sync_now_lr: Optional[float] = None

        # ---- no_sync: shared / unknown / local-only IDs ----
        self._shared_ids: Optional[Set[int]] = None
        self._shared_ids_tensor: Optional[torch.Tensor] = None
        # True only when the set came from the oracle prescan: the fallback set
        # is a prefix sample, and an ID missing from it is unknown, not local.
        self._shared_ids_complete: bool = False
        self._global_id_to_index: Optional[Dict[int, int]] = None
        self._global_unique_count: int = 0
        self._init_unique_ids: Set[int] = set()
        self._prescan_done: bool = False
        self._prescan_unique_ids: Set[int] = set()
        # Per-batch prescan metadata (opt 2): each batch's unique-ID tensor is
        # kept until finalize_prescan, which exchanges the per-batch shared-ID
        # counts across ranks into _prescan_sparse_max.  update_grads then
        # sizes the sparse all_gather from it -- no per-step count exchange,
        # no .item().  _prescan_batch_numels feeds the enqueue alignment
        # tripwire (the metadata is only sound while the training order
        # matches the prescan order).
        self._prescan_batch_ids: list = []
        self._prescan_batch_numels: list = []
        self._prescan_sparse_max: Optional[list] = None
        # When the shared-ID set exceeds this row threshold, shared ids are
        # routed to the sparse all_gather (communicating only the ids that
        # actually appear this step) instead of all-reducing the whole
        # _global_unique_count x dim buffer.  Like _global_unique_count,
        # the threshold is a rank-consistent quantity, so routing needs no
        # extra communication.  A dense all_reduce costs bytes proportional
        # to the full shared-set row count regardless of this step's hits,
        # while the sparse all_gather is proportional to the per-rank
        # shared ids actually present — usually a small fraction of the
        # former (measured: a fallback set of 14278 rows with only a few
        # thousand hits per rank per step; after the oracle prescan the
        # shared set grows to whole-dataset scale and the dense path
        # becomes unusable).
        self._dense_shared_max_rows = int(
            os.environ.get("RECSTORE_BAGPIPE_DENSE_SHARED_MAX_ROWS", "4096")
        )
        self._stats["bagpipe_no_sync_ids"] = 0.0
        self._stats["bagpipe_unknown_aggregated_ids"] = 0.0
        self._stats["bagpipe_shared_ids"] = 0.0

    # ------------------------------------------------------------------
    #  Public API (compatible with LookaheadPrefetcher where possible)
    # ------------------------------------------------------------------

    @property
    def lookahead_value(self) -> int:
        """Dynamic lookahead, adjusted based on cache pressure (opt 9)."""
        return self._dynamic_lookahead

    @property
    def base_lookahead(self) -> int:
        """Stable prefetch pipeline depth; dynamic pressure tuning ignores it."""
        return self._base_lookahead

    @property
    def max_inflight_prefetch(self) -> int:
        """Number of pre-issued RDMA prefetches the slot pool can support."""
        return self._max_inflight_prefetch

    @property
    def depth(self) -> int:
        return self.lookahead_value

    def reset_stats(self) -> None:
        self._stats = {
            "bagpipe_lookahead": float(self.lookahead_value),
            "bagpipe_cleanup_interval": float(self.cleanup_interval),
            "bagpipe_cache_entries": 0.0,
            "bagpipe_dirty_entries": 0.0,
            "bagpipe_prefetch_batches": 0.0,
            "bagpipe_prefetch_ids": 0.0,
            "bagpipe_prefetch_skip_cached": 0.0,
            "bagpipe_prefetch_pruned": 0.0,
            "bagpipe_prefetch_local_nosync_kept": 0.0,
            "bagpipe_prefetch_throttled": 0.0,
            "bagpipe_all_hit_revalidations": 0.0,
            "bagpipe_insert_failures": 0.0,
            "bagpipe_evict_stale_mirror_ids": 0.0,
            "bagpipe_sync_now_overlap_ms": 0.0,
            "bagpipe_sync_now_ids": 0.0,
            "bagpipe_sync_later_ids": 0.0,
            # __init__ seeds these two separately; omitting them here made
            # reset_stats() (called every step via consume_stats) drop the
            # keys and the next update_grads raise KeyError.
            "bagpipe_no_sync_ids": 0.0,
            "bagpipe_unknown_aggregated_ids": 0.0,
            "bagpipe_shared_ids": 0.0,
            "bagpipe_evicted_ids": 0.0,
            "bagpipe_writeback_ids": 0.0,
            "bagpipe_sgd_cache_success": 0.0,
            "bagpipe_sgd_cache_fallback": 0.0,
            "bagpipe_all_reduce_calls": 0.0,
            "bagpipe_all_reduce_ids": 0.0,
            "bagpipe_all_reduce_ms": 0.0,
            "bagpipe_prefill_ms": 0.0,
            "bagpipe_update_ms": 0.0,
            "bagpipe_cleanup_ms": 0.0,
            "bagpipe_eviction_stream_ms": 0.0,
            "bagpipe_eviction_stream_overlap_ms": 0.0,
            "bagpipe_prescan_batches": 0.0,
            "bagpipe_prescan_ids": 0.0,
            "bagpipe_prescan_sparse_max_sum": 0.0,
            "bagpipe_dynamic_lookahead": 0.0,
            "bagpipe_cache_pressure": 0.0,
        }

    def consume_stats(self, *, reset: bool = True) -> Dict[str, float]:
        """Merge current stats into a snapshot.

        Hot counters are drained here (the .item() sits after
        timer.finish()'s device drain, adding no synchronization to the
        critical path). The GPU hot accumulators are zeroed regardless of
        ``reset``: the accumulated value has already been merged into
        ``_stats``, and leaving it on the device would make the next
        consume_stats count it a second time.

        ``reset=False`` is a read-only snapshot: it returns the accumulated
        values but keeps ``_stats``; a later ``reset=True`` call returns the
        whole accumulated interval (including the snapshot period) before
        zeroing -- every count is consumed exactly once.
        """
        for key, acc in self._hot_stats_dev.items():
            self._stats[key] = self._stats.get(key, 0.0) + float(acc.item())
            acc.zero_()
        self._stats["bagpipe_cache_entries"] = float(self._cached_count())
        self._stats["bagpipe_dirty_entries"] = float(self._dirty_count())
        stats = dict(self._stats)
        if reset:
            self.reset_stats()
        return stats

    # ------------------------------------------------------------------
    #  Flat-tensor bookkeeping (isomorphic to the original oracle's flat
    #  numpy arrays)
    # ------------------------------------------------------------------

    def _alloc_bookkeeping(self, size: int) -> None:
        """(Re)allocate the four parallel bookkeeping tensors, copying old contents."""
        dev = self.device
        latest = torch.zeros(size, dtype=torch.int32, device=dev)
        ttl = torch.zeros(size, dtype=torch.int32, device=dev)
        cached = torch.zeros(size, dtype=torch.bool, device=dev)
        dirty = torch.zeros(size, dtype=torch.bool, device=dev)
        if self._latest_dev is not None:
            n = self._latest_dev.numel()
            latest[:n] = self._latest_dev
            ttl[:n] = self._ttl_dev
            cached[:n] = self._cached_dev
            dirty[:n] = self._dirty_dev
        self._latest_dev = latest
        self._ttl_dev = ttl
        self._cached_dev = cached
        self._dirty_dev = dirty

    def _to_compact(self, fused: torch.Tensor) -> torch.Tensor:
        """fused id -> compact per-table offset index (vectorized, order-preserving)."""
        row = fused & ((1 << self.fuse_k) - 1)
        return ((fused >> self.fuse_k) << self._bk_shift) + row

    def _to_fused(self, compact: torch.Tensor) -> torch.Tensor:
        """compact index -> fused id (vectorized, order-preserving; inverse of _to_compact)."""
        row = compact & (self._bk_stride - 1)
        return ((compact >> self._bk_shift) << self.fuse_k) + row

    def _compact_in_range(self, compact: torch.Tensor) -> torch.Tensor:
        """Drop compact ids beyond the bookkeeping space (unknown table / out-of-range row; should not occur)."""
        return compact[compact < self._latest_dev.numel()]

    def _hot_add(self, key: str, count: torch.Tensor) -> None:
        """Hot-path counter accumulation: GPU scalar when available (no synchronization), host float otherwise."""
        acc = self._hot_stats_dev.get(key)
        if acc is not None:
            acc += count.to(dtype=acc.dtype)
        else:
            self._stats[key] = self._stats.get(key, 0.0) + float(count)

    def _cached_count(self) -> int:
        if self._cached_dev is None:
            return 0
        return int(self._cached_dev.sum().item())

    def _dirty_count(self) -> int:
        if self._dirty_dev is None:
            return 0
        return int(self._dirty_dev.sum().item())

    def _check_gpu_cache_generation(self) -> None:
        """Fail loudly if another component reset the owned C++ cache."""
        getter = getattr(self.kv_client, "get_gpu_cache_generation", None)
        if not callable(getter):
            return
        current = int(getter())
        if self._cache_generation is None:
            self._cache_generation = current
            return
        if current != self._cache_generation:
            raise RuntimeError(
                "GPU cache generation changed while BagPipe owns it "
                f"({self._cache_generation} -> {current}); refusing to "
                "continue with a stale residency mirror"
            )
