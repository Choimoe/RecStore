"""BagPipeGradMixin — gradient update logic with shared / no_sync split.

Extracted from controller.py.  The overlap barrier (_wait_pending_sync_now)
is tightly coupled to update_grads because it waits for the async all_reduce
launched in the previous step's update_grads call.

Protocol (aligned with the official BagPipe: the PS receives ZERO per-step
traffic; persistence rides on the dirty flag and the eviction writeback):

- local-only id:  immediate best-effort in-place SGD on the local cache (no
                   Query, no synchronization, missing keys silently skipped);
                   PS persistence goes through the dirty flag + the eviction
                   value writeback.  **This requires the id to be provably
                   untouched by other ranks**: only the oracle prescan
                   (_shared_ids_complete) can prove that, so with the
                   fallback (incomplete) set such ids are NOT local-only.
- shared id:      not applied locally (that would double-count the local
                   gradient against the aggregated one); the gradient goes
                   through the aggregation path (issued on a side stream,
                   overlapping main-stream compute), and **every rank**
                   applies the aggregated gradient best-effort in place at
                   the next prefill barrier and marks the row dirty.  Replica
                   convergence = same PS initial value + same sequence of
                   aggregated increments.  The aggregation path routes on the
                   shared-set size (a rank-consistent decision): a small set
                   (<= _dense_shared_max_rows) uses a dense all_reduce over
                   the whole shared-set buffer; a large one folds into the
                   sparse all_gather, communicating only the ids that
                   actually appear this step.  With oracle prescan metadata
                   (_prescan_sparse_max) the sparse path is sized ahead of
                   time, so it needs neither the count all_gather nor the
                   .item() readback (see _all_gather_sparse_prescan_async).
- unknown id:     while the set is incomplete, ids outside the known shared
                   set have unknown ownership: they go through the sparse
                   all_gather (consumed at the same barrier as shared) and
                   are neither applied locally nor marked dirty per-rank --
                   otherwise each rank would apply its own gradient and the
                   eviction writebacks would overwrite each other
                   (last-writer-wins drops increments, replicas fork
                   permanently).
- persistence:    the dirty flag is the single source of truth for
                   "local value != PS value".  It is set by the local-only
                   apply AND by the aggregated apply (a shared row's local
                   value advanced without anyone telling the PS), and it is
                   cleared exactly when the value reaches the PS: the
                   eviction writeback or the shutdown flush.  Eviction may
                   write the same shared row from several ranks; that is
                   safe because every replica applied the same aggregate
                   sequence and the per-step prefill barrier bounds the
                   inter-rank skew to ~1 step of increments (a rank cannot
                   pass step t+1's barrier until every rank issued step t's
                   collectives).
- invalidation:   shared ids are no longer invalidated per step (the old
                   implementation invalidated ~3.6K hot shared ids per step
                   on rank!=0, the main cause of the 41ms embed_lookup).  The
                   barrier runs before prefill, so fills read rows that the
                   aggregated apply has already refreshed in place; the
                   anti-entropy sampler that used to guard the fill-vs-push
                   RDMA ordering race is gone together with the per-step
                   push -- with a potentially stale PS, invalidate+refetch
                   would REGRESS a replica by the whole unpersisted delta
                   history instead of repairing it.

The old all-or-nothing apply_sgd_update_gpu_cache (internal Query forcing a
cudaStreamSynchronize + an all-rank success vote) was removed along with its
fallback path: the best-effort op never fails, and missing keys are simply
refilled by the next prefill.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BagPipeGradMixin:
    """Provides update_grads and the associated overlap barrier.

    Expects the host class to provide (via other mixins or __init__):
        - self.device, self.kv_client, self.master_table_name, self.lr
        - self._latest_dev, self._ttl_dev, self._cached_dev, self._dirty_dev
        - self._to_compact(), _to_fused(), _compact_in_range()
        - self._sync_later_stream
        - self._pending_sync_now_work, self._pending_sync_now_lr
        - self._shared_ids, self._shared_ids_tensor, self._shared_ids_complete,
          self._global_unique_count, self._dense_shared_max_rows, self._stats
        - self._prescan_sparse_max (list[int] or None), _prescan_batch_numels
        - self._dense_all_reduce_async(), _all_gather_sparse_async(),
          _all_gather_sparse_prescan_async(), _record_shared_id_candidates()
        - self._hot_add(), self._is_distributed(), _get_rank()
    """

    # ------------------------------------------------------------------
    #  Aggregated-gradient apply (all ranks converge on the same deltas)
    # ------------------------------------------------------------------

    def _apply_aggregated(self, agg_ids: torch.Tensor,
                          agg_grads: torch.Tensor, lr: float) -> None:
        """All ranks: apply aggregated shared-ID gradients to the local cache.

        The replicas converge because every rank starts from the same PS value
        and applies the same sequence of aggregated increments.  There is no
        per-step PS push (official BagPipe semantics): the applied rows are
        marked dirty instead, and the PS receives the values through the
        eviction writeback / shutdown flush.  Only rows the residency mirror
        believes resident are marked -- the best-effort apply silently skips
        the rest, and a skipped row holds no unpersisted local delta.
        """
        agg_ids = agg_ids.to(self.device, dtype=torch.int64)
        agg_grads = agg_grads.to(self.device, dtype=torch.float32)
        if agg_grads.dim() == 1:
            agg_grads = agg_grads.unsqueeze(1)
        if not agg_ids.is_contiguous():
            agg_ids = agg_ids.contiguous()
        if not agg_grads.is_contiguous():
            agg_grads = agg_grads.contiguous()
        if agg_ids.numel() == 0:
            return

        try:
            self.kv_client.apply_sgd_update_gpu_cache_best_effort(
                self.master_table_name, agg_ids, agg_grads,
                learning_rate=lr if lr is not None else self.lr,
            )
        except Exception as exc:
            logger.warning("[BagPipe] aggregated best-effort apply failed: %s", exc)

        # The local value just advanced past the PS: dirty is what the
        # eviction writeback and the shutdown flush key on.  Mirror-resident
        # rows only -- see the docstring.
        agg_compact = self._compact_in_range(self._to_compact(agg_ids))
        resident = agg_compact[self._cached_dev[agg_compact]]
        self._dirty_dev[resident] = True
        record_dirty = getattr(self, "_record_dirty_marks", None)
        if callable(record_dirty):
            record_dirty(int(resident.numel()))

    # ------------------------------------------------------------------
    #  Overlap barrier (aggregated all_reduce of the previous step)
    # ------------------------------------------------------------------

    def _wait_pending_sync_now(self) -> None:
        """Wait for the previous step's aggregated-gradient all_reduce.

        Consumed at the top of prefill_cache, before the forward lookup: the
        aggregated shared-ID gradients are applied in place on every rank, so
        the entries stay cached and valid — no invalidation, no refetch.
        The all_reduce is issued on the side stream (_sync_later_stream),
        overlapping main-stream compute; the apply is a main-stream kernel,
        so the side stream is synchronized before reading the aggregate.
        """
        works = self._pending_sync_now_work
        if works is None:
            self._pending_sync_now_lr = None
            return
        self._pending_sync_now_work = None
        lr = self._pending_sync_now_lr
        self._pending_sync_now_lr = None

        t_start = time.perf_counter()
        if not isinstance(works, (list, tuple)):
            works = (works,)
        ids_parts = []
        grads_parts = []
        for work in works:
            work.wait()
            agg_ids, agg_grads = work.result
            if agg_ids.numel() > 0:
                ids_parts.append(agg_ids)
                grads_parts.append(agg_grads)
        if self._sync_later_stream is not None:
            self._sync_later_stream.synchronize()
        if ids_parts:
            # The dense (known-shared) and sparse (unknown-ownership) results
            # merge into a single _apply_aggregated call.
            agg_ids = ids_parts[0] if len(ids_parts) == 1 else torch.cat(ids_parts)
            agg_grads = (
                grads_parts[0] if len(grads_parts) == 1 else torch.cat(grads_parts)
            )
            self._apply_aggregated(agg_ids, agg_grads, lr)
        self._stats["bagpipe_sync_now_overlap_ms"] += (time.perf_counter() - t_start) * 1e3

    # ------------------------------------------------------------------
    #  Gradient update (sync_now / sync_later / no_sync split)
    # ------------------------------------------------------------------

    def update_grads(
        self,
        table_name: str,
        unique_ids: torch.Tensor,
        summed_grads: torch.Tensor,
        lr: float,
        batch_num: int,
    ) -> None:
        """Split gradients into local-only (immediate best-effort SGD),
        shared (dense all_reduce) and unknown (sparse all_gather); aggregated
        paths are applied at the next prefill barrier.

        Note: the deferred barrier is consumed in prefill_cache (before the
        forward lookup), NOT here.

        Sync budget on the oracle path (prescan metadata present): the sparse
        all_gather is sized by the prescan counts (rank-consistent, exchanged
        once at finalize_prescan), so the per-step count all_gather and its
        ``.item()`` readback are gone, and the bucket compaction happens
        GPU-side (no boolean-index host sync).  Control flow depends only on
        numel metadata; counters go through the GPU hot accumulators
        (``_hot_add``) and are drained at the consume_stats device-drain
        point.  Known remaining host syncs, deliberately out of scope here:
        the local-only bucket's boolean-index compaction and the prefetch
        pre-issue ``.cpu()`` (both are per-step serialization points for a
        follow-up, not correctness issues).
        """
        t_start = time.perf_counter()
        embedding_module = getattr(self, "embedding_module", None)
        if bool(
            getattr(embedding_module, "_bagpipe_cache_generation_mismatch", False)
        ):
            raise RuntimeError(
                "GPU cache generation changed during BagPipe lookup; "
                "local dirty state cannot be recovered safely"
            )
        check_generation = getattr(self, "_check_gpu_cache_generation", None)
        if callable(check_generation):
            check_generation()
        last_lookup = getattr(
            embedding_module, "_bagpipe_last_lookup_resident", None
        )
        if embedding_module is not None:
            embedding_module._bagpipe_current_prepared_ids = None
            embedding_module._bagpipe_last_lookup_resident = None
            embedding_module._bagpipe_cache_generation_mismatch = False

        if unique_ids.numel() == 0:
            self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
            return

        ids_cuda = unique_ids.to(self.device, dtype=torch.int64)
        grads_cuda = summed_grads.to(self.device, dtype=torch.float32)
        if grads_cuda.dim() == 1:
            grads_cuda = grads_cuda.unsqueeze(1)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        if not grads_cuda.is_contiguous():
            grads_cuda = grads_cuda.contiguous()

        self._stats["bagpipe_sgd_cache_success"] += 1

        self._record_shared_id_candidates(ids_cuda)

        # ---- shared / local-only vectorized split (sorted tensor +
        #      searchsorted, isomorphic to _dense_all_reduce_async,
        #      no .item()) ----
        shared_t = self._shared_ids_tensor
        if shared_t is not None and shared_t.numel() > 0:
            pos = torch.searchsorted(shared_t, ids_cuda)
            pos_clamped = pos.clamp(max=shared_t.numel() - 1)
            shared_mask = shared_t[pos_clamped] == ids_cuda
        else:
            shared_mask = torch.zeros_like(ids_cuda, dtype=torch.bool)
        # The fallback set only covers the first few steps, so ids outside it
        # may still be shared: while the set is incomplete (no oracle
        # prescan) they are unknown-ownership and go through the aggregation
        # path, never local-only.
        aggregate_unknown = self._is_distributed() and not self._shared_ids_complete
        local_mask = ~shared_mask
        unknown_mask = None
        if aggregate_unknown:
            unknown_mask = local_mask
            local_mask = torch.zeros_like(local_mask)
        residency_mask = None
        all_resident = False
        if last_lookup is not None and last_lookup[0] is ids_cuda:
            if last_lookup[1] is None:
                all_resident = True
            else:
                residency_mask = last_lookup[1]
                local_mask = local_mask & residency_mask

        # The no-evict lookup returned the authoritative post-fill residency
        # mask.  Failed inserts stay false and are retried on the next batch.
        all_compact = self._to_compact(ids_cuda)
        in_range = all_compact < self._latest_dev.numel()
        current_compact = all_compact[in_range]
        if all_resident:
            renew = (
                ~self._cached_dev[current_compact]
                | (self._ttl_dev[current_compact] >= batch_num)
            )
            self._cached_dev[current_compact] = True
            self._ttl_dev[current_compact[renew]] = (
                batch_num + self._ttl_margin
            )
        elif residency_mask is not None:
            residency_mask = residency_mask[in_range]
            failed_mask = ~residency_mask
            self._cached_dev[current_compact[failed_mask]] = False
            resident_compact = current_compact[residency_mask]
            renew = (
                ~self._cached_dev[resident_compact]
                | (self._ttl_dev[resident_compact] >= batch_num)
            )
            self._cached_dev[resident_compact] = True
            self._ttl_dev[resident_compact[renew]] = (
                batch_num + self._ttl_margin
            )
            # Backfill no-evict insert failures must feed the eviction
            # backstop too; otherwise a full cache is only noticed via the
            # prefill path.
            self._insert_failures += int(failed_mask.sum().item())
            self._stats["bagpipe_insert_failures"] = float(self._insert_failures)
        else:
            # A fallback lookup supplied no authoritative residency info for
            # this batch.  Leave the mirror untouched.  Clearing it here can
            # drop shared rows before the deferred aggregated apply marks them
            # dirty, losing their increments because there is no per-step PS
            # push.
            # Stale bits are tolerated downstream: the eviction writeback
            # reconciles against authoritative cache state, and the
            # all-hits lookup revalidates residency via contains_gpu_cache.
            pass

        # ---- no_sync (local-only): immediate best-effort in-place SGD ----
        self._hot_add("bagpipe_no_sync_ids", local_mask.sum())
        local_ids = ids_cuda[local_mask]
        if local_ids.numel() > 0:
            local_grads = grads_cuda[local_mask]
            try:
                self.kv_client.apply_sgd_update_gpu_cache_best_effort(
                    table_name, local_ids, local_grads, learning_rate=lr
                )
            except Exception as exc:
                logger.warning("[BagPipe] local best-effort apply failed: %s", exc)
            # PS persistence rides on the dirty tensor + the eviction value
            # writeback (scatter, no tolist).  local_mask already excludes
            # rows whose no-evict insert failed, so the best-effort apply and
            # the dirty mark agree.
            local_compact = self._compact_in_range(self._to_compact(local_ids))
            self._dirty_dev[local_compact] = True
            record_dirty = getattr(self, "_record_dirty_marks", None)
            if callable(record_dirty):
                record_dirty(int(local_compact.numel()))
            self._cached_dev[local_compact] = True
            self._ttl_dev[local_compact] = batch_num + self._ttl_margin

        # ---- Aggregation path: the collectives are issued on the side
        #      stream (overlapping main-stream compute); every rank applies
        #      the aggregated result to its cache at the next prefill
        #      barrier ----
        self._hot_add("bagpipe_sync_now_ids", shared_mask.sum())
        # Routing (rank-consistent, no communication needed): a small
        # shared set -> one dense all_reduce over the whole shared-set
        # buffer (negligible at low row counts, and it avoids the per-id
        # gather); a large shared set (fallback sample sets are ~14K rows,
        # and the oracle prescan can reach whole-dataset scale) -> shared
        # ids fold into the sparse all_gather, communicating only the rows
        # that actually appear this step.  A dense all_reduce costs bytes
        # proportional to the total shared-set row count regardless of this
        # step's hits — pure waste when the set is large and the hit rate
        # low (measured as the bulk of 2-rank per-step communication).
        shared_t = self._shared_ids_tensor
        use_dense = (
            shared_t is not None
            and shared_t.numel() > 0
            and getattr(self, "_global_unique_count", 0) > 0
            and getattr(self, "_global_unique_count", 0)
            <= getattr(self, "_dense_shared_max_rows", 0)
        )
        works = []
        if self._is_distributed():
            # Collectives are issued unconditionally: an empty local bucket
            # must not skip a collective that other ranks will enter (an
            # early exit on numel would hang the peer forever).  The routing
            # condition depends only on rank-consistent quantities
            # (_shared_ids_tensor / _global_unique_count), so one rank can
            # never pick dense while another picks sparse.
            # Sparse is called before dense: the side stream is serial, and
            # dense's wait_stream would otherwise push the count exchange
            # queued after it behind the main-stream backlog.
            # Routing truth table (sparse column = mask folded into the
            # sparse all_gather):
            #   use_dense & unknown    -> sparse: unknown,   dense: shared
            #   use_dense & no unknown -> sparse: none,      dense: shared
            #   large set & unknown    -> sparse: shared|unknown (single aggregate)
            #   large set & no unknown -> sparse: shared,    dense: none
            if unknown_mask is not None:
                self._hot_add(
                    "bagpipe_unknown_aggregated_ids", unknown_mask.sum()
                )
                sparse_mask = (
                    unknown_mask if use_dense else (shared_mask | unknown_mask)
                )
            else:
                sparse_mask = None if use_dense else shared_mask
            if sparse_mask is not None:
                # Oracle path: prescan counts size the all_gather ahead of
                # time (no count exchange, no .item(), no boolean-index
                # compaction -- the bucket is padded GPU-side).  The prescan
                # metadata is globally uniform (finalize_prescan runs on all
                # ranks or none) and batch_num indexes it identically on
                # every rank, so the routing cannot diverge across ranks.
                prescan_max = getattr(self, "_prescan_sparse_max", None)
                if prescan_max and self._shared_ids_complete:
                    max_n = prescan_max[batch_num % len(prescan_max)]
                    _, _, sparse_work = self._all_gather_sparse_prescan_async(
                        ids_cuda, grads_cuda, sparse_mask, max_n,
                        stream=self._sync_later_stream,
                    )
                else:
                    _, _, sparse_work = self._all_gather_sparse_async(
                        ids_cuda[sparse_mask],
                        grads_cuda[sparse_mask],
                        stream=self._sync_later_stream,
                    )
                if sparse_work is not None:
                    works.append(sparse_work)
            if use_dense:
                _, _, dense_work = self._dense_all_reduce_async(
                    ids_cuda[shared_mask],
                    grads_cuda[shared_mask],
                    stream=self._sync_later_stream,
                )
                if dense_work is not None:
                    works.append(dense_work)
        self._pending_sync_now_work = works or None
        self._pending_sync_now_lr = lr if works else None

        self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
