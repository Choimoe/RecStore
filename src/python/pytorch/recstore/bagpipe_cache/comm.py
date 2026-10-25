"""Cross-GPU gradient communication for BagPipe cache controller.

Handles shared-ID classification, dense/sparse all_reduce, and the
one-time all_gather that builds the shared-ID set (opt 1, 2, 8).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .types import _CompositeWork, _DenseWork

logger = logging.getLogger(__name__)


class BagPipeCommMixin:
    """Mixin providing cross-GPU gradient synchronization primitives.

    Expects the host class to provide: ``device``, ``_shared_ids``,
    ``_shared_ids_tensor`` (sorted int64 tensor), ``_shared_ids_complete``
    (True only for an oracle set), ``_global_id_to_index``,
    ``_global_unique_count``, ``_init_unique_ids``, ``_prescan_done``,
    ``_prescan_unique_ids``, ``_prescan_batch_ids`` (per-batch CPU id
    tensors collected during prescan), ``_prescan_batch_numels``,
    ``_prescan_sparse_max``, ``_stats``, ``lookahead_value``.
    """

    def _is_distributed(self) -> bool:
        try:
            return dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    def _get_rank(self) -> int:
        try:
            return dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    #  Shared-ID set construction (opt 1, opt 8 prescan)
    # ------------------------------------------------------------------

    def _record_shared_id_candidates(self, unique_ids: torch.Tensor) -> None:
        """Accumulate a batch's unique IDs for the fallback shared-ID set.

        Pure bookkeeping: the all_gather happens in
        :meth:`_maybe_build_shared_id_set`, which is driven from the
        step-aligned ``cleanup`` hook.  Triggering the collective from the
        hot path instead would let ranks that skip a batch (empty table,
        uneven shard) enter the all_gather at different steps and hang.
        """
        if self._shared_ids is not None or unique_ids.numel() == 0:
            return
        self._init_unique_ids.update(unique_ids.tolist())

    def _maybe_build_shared_id_set(self, current_step: int) -> None:
        """Build the fallback shared-ID set once, at a step boundary.

        The fallback set only covers the first ``max(lookahead, 2)`` steps, so
        it is a *fast-path hint*, never a proof of locality:
        ``_shared_ids_complete`` stays False and ``update_grads`` aggregates
        every ID outside the set through the sparse path.  Only the oracle
        prescan (:meth:`prescan_batch` + :meth:`finalize_prescan`) yields a set
        complete enough to skip synchronization for its local-only IDs.
        """
        if self._shared_ids is not None:
            return
        if not self._is_distributed():
            self._shared_ids = set()
            self._shared_ids_tensor = torch.empty(0, dtype=torch.int64)
            self._shared_ids_complete = True
            return

        if current_step < max(self.lookahead_value, 2):
            return

        self._build_shared_id_set_from(self._init_unique_ids, complete=False)
        logger.warning(
            "[BagPipe] shared-ID hint set built from the first %d steps: "
            "%d IDs known shared, %d IDs still unknown (aggregated via the "
            "sparse path).  A prefix sample cannot prove an ID is local-only; "
            "run the oracle prescan to enable the local-only fast path.",
            int(current_step),
            len(self._shared_ids),
            len(self._init_unique_ids) - len(self._shared_ids),
        )
        self._init_unique_ids = set()

    def _build_shared_id_set_from(
        self, local_id_set: set, *, complete: bool
    ) -> None:
        """Build the shared-ID set + global index mapping via all_gather.

        ``complete`` records whether ``local_id_set`` covers every ID this rank
        can ever touch (oracle prescan) or is only a prefix sample (fallback);
        it is what licenses the local-only fast path in ``update_grads``.
        """
        if not self._is_distributed():
            self._shared_ids = set()
            self._shared_ids_tensor = torch.empty(0, dtype=torch.int64)
            self._global_id_to_index = {}
            self._global_unique_count = 0
            self._shared_ids_complete = True
            return

        local_ids = torch.tensor(sorted(local_id_set),
                                  dtype=torch.int64, device=self.device)
        local_n = torch.tensor([local_ids.numel()], dtype=torch.int64,
                                device=self.device)
        world_size = dist.get_world_size()
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, local_n)
        max_n = max(int(ni.item()) for ni in n_list)

        padded = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        padded[:local_ids.numel()] = local_ids
        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        dist.all_gather(ids_list, padded)

        id_rank_count: Dict[int, int] = {}
        for r in range(world_size):
            nr = int(n_list[r].item())
            for fid in ids_list[r][:nr].tolist():
                id_rank_count[fid] = id_rank_count.get(fid, 0) + 1

        self._shared_ids = {fid for fid, cnt in id_rank_count.items() if cnt > 1}
        all_shared = sorted(self._shared_ids)
        # The sorted tensor serves two purposes at once: the shared/local
        # split (searchsorted membership), and the fid -> dense-buffer row
        # mapping (the sorted position IS the _global_id_to_index value,
        # avoiding a per-fid dict.get loop).
        self._shared_ids_tensor = torch.tensor(
            all_shared, dtype=torch.int64, device=self.device
        )
        self._global_id_to_index = {fid: i for i, fid in enumerate(all_shared)}
        self._global_unique_count = len(all_shared)
        self._shared_ids_complete = bool(complete)

    def finalize_prescan(self) -> None:
        """After all batches pre-scanned, build the complete shared-ID set.

        Also exchanges the per-batch shared-ID counts across ranks
        (:meth:`_build_prescan_sparse_max`) so the sparse all_gather in
        ``update_grads`` can size its buffers without the per-step count
        exchange.  Collective: every rank must call it after its own prescan
        pass, and all ranks must agree on whether prescan ran at all (the
        training loop gates the whole prescan on the same config flag).
        """
        if self._prescan_done:
            return
        self._build_shared_id_set_from(self._prescan_unique_ids, complete=True)
        self._prescan_done = True
        total_unique = len(self._prescan_unique_ids)
        logger.info(
            "[BagPipe] oracle prescan: %d shared IDs, %d local-only, "
            "%d total unique across %d batches",
            len(self._shared_ids),
            total_unique - len(self._shared_ids),
            total_unique,
            int(self._stats["bagpipe_prescan_batches"]),
        )
        self._prescan_unique_ids = set()
        self._build_prescan_sparse_max()

    def _build_prescan_sparse_max(self) -> None:
        """Exchange per-batch shared-ID counts for sync-free sparse sizing.

        For every prescan batch t, computes ``|shared set ∩ batch_t|`` locally
        (searchsorted against the just-built sorted shared tensor), then
        all_gathers the count vectors across ranks and keeps the per-batch
        maximum.  ``update_grads`` uses that maximum as the padded all_gather
        width for step t, eliminating the per-step count all_gather and its
        ``.item()`` readback.

        Correctness contract: the maximum is an upper bound on every rank's
        sparse bucket at step t *iff* the training pass consumes the batches
        in the prescan order.  The runner guarantees that (fixed-seed
        DistributedSampler without set_epoch; the wrap-around repeats the
        epoch-0 order, hence the modulo indexing), and ``enqueue`` verifies
        it cheaply via the per-batch unique-ID counts (a mismatch raises --
        failing loudly beats a silent collective misalignment).
        """
        batches = getattr(self, "_prescan_batch_ids", None) or []
        self._prescan_batch_ids = None
        self._prescan_batch_numels = [int(b.numel()) for b in batches]
        if not self._is_distributed():
            # A single rank never aggregates: no sizing metadata needed.
            self._prescan_sparse_max = None
            return

        n = len(batches)
        shared_t = self._shared_ids_tensor
        local_counts = torch.zeros(max(n, 1), dtype=torch.int64,
                                   device=self.device)
        if n > 0 and shared_t is not None and shared_t.numel() > 0:
            for t, batch in enumerate(batches):
                batch_dev = batch.to(self.device, dtype=torch.int64)
                pos = torch.searchsorted(shared_t, batch_dev)
                pos_clamped = pos.clamp(max=shared_t.numel() - 1)
                hits = (shared_t[pos_clamped] == batch_dev)
                local_counts[t] = hits.sum()

        # Exchange: ranks may scan different batch counts (shard boundary);
        # pad the shorter vectors with 0 so the per-index maximum still comes
        # from whichever rank actually has that batch.
        world_size = dist.get_world_size()
        n_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                  for _ in range(world_size)]
        dist.all_gather(n_list, n_tensor)
        max_batches = max(int(ni.item()) for ni in n_list)
        if max_batches == 0:
            self._prescan_sparse_max = None
            return
        padded_counts = torch.zeros(max_batches, dtype=torch.int64,
                                    device=self.device)
        padded_counts[:n] = local_counts[:n]
        counts_list = [torch.zeros(max_batches, dtype=torch.int64,
                                   device=self.device)
                       for _ in range(world_size)]
        dist.all_gather(counts_list, padded_counts)
        max_counts = torch.stack(counts_list).amax(dim=0)
        self._prescan_sparse_max = [int(c) for c in max_counts.tolist()]
        self._stats["bagpipe_prescan_sparse_max_sum"] = float(
            sum(self._prescan_sparse_max)
        )

    # ------------------------------------------------------------------
    #  All-reduce primitives (opt 2)
    # ------------------------------------------------------------------

    def _all_gather_sparse_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Sparse all_gather + aggregate gradients across GPUs (async).

        Sync budget: the size negotiation (a count all_gather plus one
        ``.item()``) is the price of not knowing the bucket sizes ahead of
        time — allocating the padded buffers needs max_n on the host.  The
        count exchange is issued on the side stream with the same
        ``stream.wait_stream(current_stream)`` ordering as the payload
        all_gather, and max_n is read back on that same stream (see the
        comments in the body).  The oracle prescan path avoids this entire
        exchange: :meth:`_all_gather_sparse_prescan_async` sizes the buffers
        from the prescan count metadata instead.

        Stream ordering is a correctness requirement, not an optional
        optimization: NCCL matches collectives across ranks by the
        **device-side execution order** of their kernels.  Kernels on
        different streams within one process carry no execution-order
        guarantee — if the side-stream count exchange does not wait on the
        main stream, it can run ahead of a main-stream collective that has
        not drained yet (e.g. the backward DDP all_reduce) on one rank but
        not on the other, and the two ranks' collective sequences misalign:
        the peer then matches later collectives against the wrong sequence
        numbers, corrupting data at best and deadlocking at worst (observed
        as a 2-rank RDMA training hang at step 0, with the NCCL watchdog
        SIGABRT-ing only after 10 minutes).  Every collective issued on the
        side stream must therefore ``wait_stream`` the main stream first, so
        that it is ordered after all earlier-enqueued main-stream
        collectives; the next step's prefill barrier then has the main
        stream wait on the side stream, closing the ordering loop.
        """
        if not self._is_distributed():
            return ids, grads, None

        t_start = time.perf_counter()
        world_size = dist.get_world_size()
        n = ids.numel()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        n_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        if stream is not None:
            # See the docstring: a side-stream collective must be ordered
            # after already-enqueued main-stream collectives (the backward
            # DDP all_reduce etc.), or the cross-rank matching misaligns.
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                dist.all_gather(n_list, n_tensor)
                # max_n must be read back on the same stream as the
                # collective: the ``.item()`` D2H copy is issued on the
                # current stream and blocks the host — if it lands on the
                # main stream, which does not wait on the side stream, the
                # copy can run before the allgather kernel writes and read
                # the zero-initialized n_list -> max_n=0 -> this rank takes
                # the early exit while the peer waits in the payload
                # all_gather, misaligning both ranks' collective sequences
                # from then on (this rank's barrier matches the peer's
                # payload) — a deadlock the NCCL watchdog only reports 10
                # minutes later.
                max_n = max(int(ni.item()) for ni in n_list)
        else:
            dist.all_gather(n_list, n_tensor)
            max_n = max(int(ni.item()) for ni in n_list)

        if max_n == 0:
            # max_n is the rank-consistent maximum, so this early exit
            # cannot leave a peer hanging.
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

        ids_dev = ids.to(self.device, dtype=torch.int64)
        grads_dev = grads.to(self.device, dtype=torch.float32)
        if not ids_dev.is_contiguous():
            ids_dev = ids_dev.contiguous()
        if not grads_dev.is_contiguous():
            grads_dev = grads_dev.contiguous()

        # Sentinel padding: rows beyond this rank's true count are filled
        # with -1; wait() recovers each rank's contribution with a boolean
        # mask, so no per-rank count .item() synchronization is needed.
        padded_ids = torch.full((max_n,), -1, dtype=torch.int64, device=self.device)
        padded_ids[:n] = ids_dev
        padded_grads = torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
        padded_grads[:n] = grads_dev

        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                       for _ in range(world_size)]

        if stream is not None:
            # The buffers are written on the main stream and consumed by a
            # collective issued on the side stream: the side stream must
            # wait on the main stream first, or the NCCL kernel may read a
            # stale padding buffer (a write-after-read race — kernel launch
            # latency usually masks it, but it is UB by semantics).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
                work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)
        else:
            work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
            work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)

        work = _CompositeWork(work_ids, work_grads, ids_list, grads_list,
                              world_size, dim, self.device, self._stats,
                              padded=(padded_ids, padded_grads))
        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
        return ids_dev, grads_dev, work

    def _all_gather_sparse_prescan_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        mask: torch.Tensor,
        max_n: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Sparse all_gather sized by oracle prescan counts (async, sync-free).

        ``max_n`` comes from the one-time prescan count exchange
        (:meth:`_build_prescan_sparse_max`) instead of a per-step count
        all_gather, so this path issues no host readback at all.  The bucket
        (``mask`` over the step's unique ids) is compacted GPU-side with an
        index-preserving scatter: masked rows land at their cumsum slot,
        unmasked rows pile into a trash slot that is sliced off -- boolean
        indexing would need the output size on the host and force a
        synchronization.

        Correctness notes (see also _build_prescan_sparse_max):

        - ``max_n`` is rank-consistent (identical prescan metadata on every
          rank), so the skip/issue decision when ``max_n == 0`` cannot leave
          a peer hanging in the collective.
        - The bucket size is bounded by ``|shared ∩ batch_t|`` by
          construction (the mask is computed against the same shared tensor
          the prescan counts were), so masked cumsum slots stay below
          ``max_n``.  If the dataloader order ever diverged from the prescan
          despite the numel tripwire in ``enqueue``, slots beyond the buffer
          raise a device-side indexing assert rather than corrupting data
          silently (an overflow of exactly one row collides with the trash
          slot and is dropped -- a documented residual).
        """
        if not self._is_distributed():
            return ids, grads, None

        t_start = time.perf_counter()
        world_size = dist.get_world_size()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        if max_n <= 0:
            # Every rank computed the same prescan maximum, so skipping the
            # collective here cannot leave a peer hanging.
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

        ids_dev = ids.to(self.device, dtype=torch.int64)
        grads_dev = grads.to(self.device, dtype=torch.float32)
        if not ids_dev.is_contiguous():
            ids_dev = ids_dev.contiguous()
        if not grads_dev.is_contiguous():
            grads_dev = grads_dev.contiguous()

        # GPU-side sentinel padding: masked row i -> slot cumsum(i)-1 (dense
        # order), unmasked rows -> trash slot max_n.  index_put with tensor
        # indices is shape-known, so no host synchronization is needed.
        slot = torch.cumsum(mask, 0) - 1
        trash = torch.full_like(slot, max_n)
        dest = torch.where(mask, slot, trash)

        padded_ids = torch.full((max_n + 1,), -1, dtype=torch.int64,
                                device=self.device)
        padded_ids[dest] = ids_dev
        padded_ids = padded_ids[:max_n]
        padded_grads = torch.zeros(max_n + 1, dim, dtype=torch.float32,
                                   device=self.device)
        padded_grads[dest] = grads_dev
        padded_grads = padded_grads[:max_n]

        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                    for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                      for _ in range(world_size)]

        if stream is not None:
            # Same ordering requirements as _all_gather_sparse_async: the
            # padding was written on the main stream and the collective runs
            # on the side stream, so the side stream must wait first (also
            # ordering it after any earlier main-stream collective, or the
            # cross-rank NCCL matching would misalign).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
                work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)
        else:
            work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
            work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)

        work = _CompositeWork(work_ids, work_grads, ids_list, grads_list,
                              world_size, dim, self.device, self._stats,
                              padded=(padded_ids, padded_grads))
        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
        return ids_dev, grads_dev, work

    def _dense_all_reduce_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Dense all_reduce of shared-ID gradients using a global index mapping.

        The reduced buffer is identical on every rank; the returned work
        extracts its nonzero rows so that all ranks apply the exact same
        aggregated rows (see :class:`_DenseWork` for why own-row extraction
        would fork the replicas).
        """
        if not self._is_distributed():
            return ids, grads, None

        if self._global_id_to_index is None or self._global_unique_count == 0:
            return self._all_gather_sparse_async(ids, grads, stream)

        t_start = time.perf_counter()
        world_size = dist.get_world_size()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        dense_grads = torch.zeros(
            (self._global_unique_count, dim),
            dtype=torch.float32, device=self.device,
        )

        ids_dev = ids.to(self.device, dtype=torch.int64)
        grads_dev = grads.to(self.device, dtype=torch.float32)

        # Vectorized fid -> dense-buffer row: _shared_ids_tensor is
        # sorted, so a searchsorted hit IS the row number (identical to
        # the _global_id_to_index value).  No .any() early exit -- that
        # would be a forced synchronization; with an empty valid set the
        # all_reduce aggregates a zero buffer and _apply_aggregated
        # naturally skips the empty id set.
        shared_tensor = self._shared_ids_tensor
        if shared_tensor is None or shared_tensor.numel() == 0:
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None
        pos = torch.searchsorted(shared_tensor, ids_dev)
        pos_clamped = pos.clamp(max=shared_tensor.numel() - 1)
        valid_mask_tensor = shared_tensor[pos_clamped] == ids_dev

        valid_indices = pos_clamped[valid_mask_tensor]
        valid_ids = ids_dev[valid_mask_tensor].contiguous()
        valid_grads = grads_dev[valid_mask_tensor].contiguous()

        dense_grads.index_put_((valid_indices,), valid_grads)

        if stream is not None:
            # Same as above: index_put_ writes dense_grads on the main
            # stream and the all_reduce consumes it on the side stream --
            # establish main -> side ordering first.
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                work = dist.all_reduce(dense_grads, async_op=True)
        else:
            work = dist.all_reduce(dense_grads, async_op=True)

        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ids"] += float(len(valid_indices))
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

        # The work handle extracts the nonzero rows of the reduced buffer, a
        # set identical on every rank (the all_reduced buffer is bit-equal
        # everywhere): every rank must apply the same aggregated rows or the
        # replicas fork -- see _DenseWork.  valid_ids/valid_grads (this
        # rank's own contributions) are only stats/return-value material.
        work_obj = _DenseWork(
            work, dense_grads, shared_tensor, dim, self.device
        )
        return valid_ids, valid_grads, work_obj
