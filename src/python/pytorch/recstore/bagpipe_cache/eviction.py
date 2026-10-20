"""Cache eviction, writeback, and dynamic lookahead for BagPipe controller.

Handles TTL-based eviction, value writeback to PS, async background thread,
and dynamic lookahead adjustment (opt 4, 5, 6, 7, 9).

Eviction decisions are a vectorized full-space mask scan over the flat
tensors (~5.2M elements, ~0.1ms), isomorphic to the original BagPipe's
numpy-array semantics; there is no per-fid Python loop.
"""

from __future__ import annotations

import logging
import queue
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BagPipeEvictionMixin:
    """Mixin providing eviction, writeback, and cleanup-thread methods.

    Expects the host class to provide: ``device``, ``kv_client``,
    ``embedding_dim``, ``master_table_name``, ``_cached_dev``, ``_ttl_dev``,
    ``_dirty_dev``, ``_shared_ids_tensor``, ``cache_capacity``,
    ``_eviction_stream``, ``_cleanup_queue``, ``_cleanup_thread``,
    ``_stats``, ``_base_lookahead``, ``_dynamic_lookahead``,
    ``_max_lookahead``, ``_additive_increase``,
    ``_multiplicative_decrease``, ``_pressure_low``, ``_pressure_high``,
    ``_lookahead_adjust_interval``, ``cleanup_batch_proportion``,
    ``cleanup_interval``, ``_cached_count()``, ``_is_distributed()``,
    ``_get_rank()``, ``_capacity_watermark``, ``_max_evict_per_cleanup``,
    ``_to_compact()``, ``_to_fused()``, ``_compact_in_range()``.
    """

    def schedule_cleanup(self, current_batch: int) -> None:
        """Run cheap step-aligned hooks and request cleanup only when needed."""
        # This can contain a collective and therefore remains step-aligned on
        # every rank even when local cache pressure differs.
        self._maybe_build_shared_id_set(current_batch)
        self._maybe_adjust_lookahead(current_batch)

        ttl_due = (current_batch + 1) % self.cleanup_interval == 0
        high_water = int(self.cache_capacity * self._capacity_watermark)
        pressure = (
            self.cache_capacity > 0
            and self._cached_count_upper_bound() >= high_water
        )
        if ttl_due or pressure or self._insert_failures > 0:
            self._cleanup_pending = True
            self._cleanup_scan_ttl = self._cleanup_scan_ttl or ttl_due
            self._cleanup_batch = max(self._cleanup_batch, current_batch)

    def _run_scheduled_cleanup(self) -> None:
        """Apply scheduled eviction after deferred gradients are visible."""
        if not self._cleanup_pending or self._cached_dev is None:
            return
        current_batch = self._cleanup_batch
        scan_ttl = self._cleanup_scan_ttl
        self._cleanup_pending = False
        self._cleanup_scan_ttl = False
        self._run_cleanup(current_batch, scan_ttl=scan_ttl)

    def _run_cleanup(self, current_batch: int, *, scan_ttl: bool) -> None:
        """Reclaim at most one shared eviction budget from all causes."""
        t_start = time.perf_counter()
        budget = self._max_evict_per_cleanup
        exact_count = self._refresh_count_snapshots()
        ttl_backlog = False

        if scan_ttl and budget > 0:
            expired = (
                self._cached_dev & (self._ttl_dev < current_batch)
            ).nonzero(as_tuple=False).squeeze(1)
            ttl_backlog = int(expired.numel()) > budget
            if ttl_backlog:
                order = self._ttl_dev[expired].argsort()
                expired = expired[order[:budget]]
            expired = expired.contiguous()
            evicted = int(expired.numel())
            if evicted:
                self._evict_entries(expired)
                budget -= evicted
                exact_count -= evicted

        failures = int(self._insert_failures)
        high_water = int(self.cache_capacity * self._capacity_watermark)
        overflow = max(0, exact_count - high_water) if self.cache_capacity > 0 else 0
        failure_reclaim = max(failures * 2, 1024) if failures else 0
        reclaim = min(max(overflow, failure_reclaim), budget, exact_count)
        if reclaim > 0:
            ttl_eff = torch.where(
                self._cached_dev,
                self._ttl_dev,
                torch.iinfo(torch.int32).max,
            )
            pressure_ids = torch.topk(
                ttl_eff, k=reclaim, largest=False
            ).indices.contiguous()
            self._evict_entries(pressure_ids)
            budget -= reclaim
            exact_count -= reclaim

        if failures:
            self._insert_failures = 0
            self._stats["bagpipe_insert_failures"] = float(failures)

        pressure_backlog = self.cache_capacity > 0 and exact_count > high_water
        failure_backlog = failure_reclaim > reclaim and exact_count > 0
        if ttl_backlog or pressure_backlog or failure_backlog:
            # Continue at the next prefill safe point instead of exceeding
            # this step's eviction/writeback budget.
            self._cleanup_pending = True
            self._cleanup_scan_ttl = ttl_backlog
            self._cleanup_batch = current_batch + 1

        self._stats["bagpipe_cleanup_ms"] += (
            time.perf_counter() - t_start
        ) * 1e3

    def _maybe_adjust_lookahead(self, current_batch: int) -> None:
        """Adjust lookahead based on cache pressure (opt 9)."""
        if current_batch % self._lookahead_adjust_interval != 0:
            return
        if self.cache_capacity <= 0:
            return
        cached_count = self._cached_count_upper_bound()
        pressure = cached_count / self.cache_capacity
        old_la = self._dynamic_lookahead
        if pressure > self._pressure_high:
            self._dynamic_lookahead = max(1, int(self._dynamic_lookahead * self._multiplicative_decrease))
        elif pressure < self._pressure_low:
            self._dynamic_lookahead = min(self._max_lookahead, self._dynamic_lookahead + self._additive_increase)
        if self._dynamic_lookahead != old_la:
            logger.info(
                "[BagPipe] dynamic lookahead: %d -> %d (pressure=%.2f, cache=%d/%d)",
                old_la, self._dynamic_lookahead, pressure,
                cached_count, self.cache_capacity,
            )
        self._stats["bagpipe_dynamic_lookahead"] = float(self._dynamic_lookahead)
        self._stats["bagpipe_cache_pressure"] = pressure

    def _read_cache_values(self, ids: torch.Tensor) -> tuple:
        """Read GPU cache values for no_sync IDs on the eviction stream (opt 7)."""
        ids_cuda = ids.to(device=self.device, dtype=torch.int64)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        stream = self._eviction_stream
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    values = self.kv_client.gpu_cache_lookup_flat(
                        ids_cuda, self.embedding_dim
                    )
                event = torch.cuda.Event()
                event.record(stream)
            else:
                values = self.kv_client.gpu_cache_lookup_flat(
                    ids_cuda, self.embedding_dim
                )
                event = None
            if values.device != self.device:
                values = values.to(self.device)
            if not values.is_contiguous():
                values = values.contiguous()
            return ids_cuda, values, event
        except Exception as exc:
            logger.warning("[BagPipe] value read for writeback failed: %s", exc)
            return ids_cuda, None, None

    def _evict_entries(self, expired_ids: torch.Tensor) -> None:
        """Evict expired entries: write back values + invalidate (opt 4, 6).

        ``expired_ids`` are compact indices (from the bookkeeping tensors'
        nonzero); the kv_client / GPU cache key space is fused id, converted
        before the call.  Only dirty entries need a value writeback: dirty is
        the single source of truth for "local value != PS value", set by the
        local-only apply AND by the aggregated apply (there is no per-step PS
        push anymore).  Several ranks may write back the same shared row --
        every replica applied the same aggregate sequence, and the per-step
        prefill barrier bounds the inter-rank skew to ~1 step of increments,
        so last-writer-wins loses at most one increment (the same bound the
        official BagPipe accepts).

        Values are read before invalidation (the cache still holds them), but
        only rows the C++ cache actually reports as removed are written back:
        a row the mirror claims but the cache no longer holds reads back
        undefined bytes, and persisting those would corrupt the PS.
        """
        if expired_ids.numel() == 0:
            return

        requested_evictions = expired_ids

        dirty_m = self._dirty_dev[expired_ids]
        wb_mask = dirty_m
        wb_ids = expired_ids[wb_mask]
        dirty_cleared = int(wb_ids.numel())
        wb_vals = None
        wb_event = None
        if wb_ids.numel() > 0:
            wb_ids, wb_vals, wb_event = self._read_cache_values(
                self._to_fused(wb_ids)
            )

        invalidator = getattr(
            self.kv_client, "invalidate_gpu_cache_with_mask", None
        )
        if not callable(invalidator):
            raise RuntimeError(
                "BagPipe requires invalidate_gpu_cache_with_mask to keep its "
                "residency mirror authoritative"
            )
        removed = invalidator(
            self.master_table_name, self._to_fused(requested_evictions)
        )
        expired_ids = requested_evictions[removed]
        self._stats["bagpipe_evicted_ids"] += float(expired_ids.numel())

        # Enqueue only the rows that were genuinely resident.
        if wb_vals is not None and wb_ids.numel() > 0:
            wb_removed = removed[wb_mask]
            wb_ids = wb_ids[wb_removed]
            wb_vals = wb_vals[wb_removed]
            if wb_ids.numel() > 0:
                self._cleanup_queue.put((wb_ids, wb_vals, wb_event))
                self._stats["bagpipe_writeback_ids"] += float(wb_ids.numel())

        # Reconcile the mirror with the authoritative result.  Rows the mirror
        # listed as resident but the cache does not hold have to be dropped
        # too, otherwise they stay expired and are re-picked every cleanup.
        self._stats["bagpipe_evict_stale_mirror_ids"] += float(
            int(requested_evictions.numel()) - int(expired_ids.numel())
        )
        record_removals = getattr(self, "_record_cache_removals", None)
        if callable(record_removals):
            record_removals(int(requested_evictions.numel()))
        record_dirty_clears = getattr(self, "_record_dirty_clears", None)
        if callable(record_dirty_clears):
            record_dirty_clears(dirty_cleared)
        self._cached_dev[requested_evictions] = False
        self._dirty_dev[requested_evictions] = False
        self._ttl_dev[requested_evictions] = 0

    # ------------------------------------------------------------------
    #  Background cleanup thread (opt 5)
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background thread for asynchronous PS value writeback (opt 5)."""
        while True:
            try:
                task = self._cleanup_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            wb_ids, wb_vals, wb_event = task
            if wb_vals is None:
                continue
            if wb_event is not None:
                wb_event.synchronize()
            try:
                self.kv_client.emb_write_values(
                    self.master_table_name, wb_ids, wb_vals
                )
            except Exception as exc:
                logger.warning("[BagPipe] async value writeback failed: %s", exc)

    def shutdown(self) -> None:
        """Signal the background thread to exit and flush pending work."""
        try:
            # Consumes the last deferred aggregation first: its aggregated
            # apply marks rows dirty, and the flush below is what persists
            # them.
            self._wait_pending_sync_now()
        except Exception:
            pass
        try:
            while True:
                try:
                    task = self._cleanup_queue.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    break
                wb_ids, wb_vals, wb_event = task
                if wb_event is not None:
                    wb_event.synchronize()
                if wb_vals is not None:
                    self.kv_client.emb_write_values(
                        self.master_table_name, wb_ids, wb_vals
                    )
        except Exception:
            pass
        self._flush_dirty_at_shutdown()
        prepared_ids = getattr(self, "_prepared_ids", None)
        if prepared_ids is not None:
            prepared_ids.clear()
        embedding_module = getattr(self, "embedding_module", None)
        if embedding_module is not None:
            embedding_module._bagpipe_current_prepared_ids = None
            embedding_module._bagpipe_last_lookup_resident = None
            embedding_module._bagpipe_cache_generation_mismatch = False
        self._cleanup_queue.put(None)

    def _flush_dirty_at_shutdown(self) -> None:
        """Flush every dirty row still in the cache to the PS at shutdown.

        With no per-step PS push, dirty covers local-only rows AND shared
        rows advanced by the aggregated applies (any such row left unflushed
        would lose its increments at process exit).  Shared rows are sharded
        across ranks by their position in the sorted shared set, so each row
        has exactly one writer: ranks may finish a step apart, and duplicate
        writers would race last-writer-wins on the final values.  Reads and
        writes are chunked to bound peak memory (~32MB per chunk).
        """
        if self._dirty_dev is None:
            return
        try:
            dirty_mask = self._dirty_dev.clone()
        except Exception:
            return
        shared_t = self._shared_ids_tensor
        if (
            shared_t is not None
            and shared_t.numel() > 0
            and bool(dirty_mask.any().item())
        ):
            shared_compact = self._to_compact(shared_t)
            shared_compact = shared_compact[
                shared_compact < self._dirty_dev.numel()
            ]
            dirty_shared = dirty_mask[shared_compact]
            keep = dirty_shared
            if self._is_distributed():
                world = dist.get_world_size()
                rank = self._get_rank()
                owner = (torch.arange(
                    shared_compact.numel(), device=shared_compact.device
                ) % world) == rank
                keep = dirty_shared & owner
            dirty_mask[shared_compact] = False
            dirty_mask[shared_compact[keep]] = True
        chunk = 65536
        total = int(dirty_mask.sum().item())
        if total == 0:
            return
        logger.info("[BagPipe] shutdown: flushing %d dirty entries to PS", total)
        done = 0
        while done < total:
            remaining = dirty_mask.sum().item()
            if int(remaining) == 0:
                break
            ids = dirty_mask.nonzero(as_tuple=False).squeeze(1)[:chunk]
            ids_fused, vals, event = self._read_cache_values(self._to_fused(ids))
            if vals is not None and ids_fused.numel() > 0:
                if event is not None:
                    event.synchronize()
                try:
                    self.kv_client.emb_write_values(
                        self.master_table_name, ids_fused, vals
                    )
                except Exception as exc:
                    logger.warning("[BagPipe] shutdown writeback failed: %s", exc)
                self._stats["bagpipe_writeback_ids"] += float(ids_fused.numel())
                done += int(ids_fused.numel())
            # Clear the flag whether or not the writeback succeeded, so the
            # chunking loop cannot spin forever.
            dirty_mask[ids] = False
            self._dirty_dev[ids] = False
