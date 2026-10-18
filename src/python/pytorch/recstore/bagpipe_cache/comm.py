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
    ``_prescan_unique_ids``, ``_stats``, ``lookahead_value``.
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

    def _load_balanced_push(
        self, table_name: str, ids: torch.Tensor, grads: torch.Tensor
    ) -> None:
        """Push aggregated gradients to the PS, load-balanced across ranks.

        Each rank pushes a disjoint contiguous slice of the (already
        all_reduced) gradients, relieving rank-0 which also hosts the PS
        server.  Mirrors the original BagPipe eviction writeback that splits
        global IDs by ``chunk_size = len // world_size``.
        """
        if ids.numel() == 0:
            return
        if not self._is_distributed():
            try:
                self.kv_client.update(table_name, ids, grads)
            except Exception as exc:
                logger.warning("[BagPipe] load-balanced push failed: %s", exc)
            return
        rank = self._get_rank()
        world = dist.get_world_size()
        n = int(ids.size(0))
        chunk = n // world
        start = rank * chunk
        end = n if rank == world - 1 else (rank + 1) * chunk
        if start >= end:
            return
        my_ids = ids[start:end]
        my_grads = grads[start:end]
        if not my_ids.is_contiguous():
            my_ids = my_ids.contiguous()
        if not my_grads.is_contiguous():
            my_grads = my_grads.contiguous()
        try:
            self.kv_client.update(table_name, my_ids, my_grads)
        except Exception as exc:
            logger.warning("[BagPipe] load-balanced push failed: %s", exc)

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
        self._stats["bagpipe_shared_ids"] = float(len(self._shared_ids))
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
        # 排序张量同时服务两处: torch.isin 的 shared/local 切分, 以及
        # searchsorted 的 fid -> dense 缓冲行号映射（= sorted 序即
        # _global_id_to_index 的值, 免去逐 fid dict.get 循环）。
        self._shared_ids_tensor = torch.tensor(
            all_shared, dtype=torch.int64, device=self.device
        )
        self._global_id_to_index = {fid: i for i, fid in enumerate(all_shared)}
        self._global_unique_count = len(all_shared)
        self._shared_ids_complete = bool(complete)

    def finalize_prescan(self) -> None:
        """After all batches pre-scanned, build the complete shared-ID set."""
        if self._prescan_done:
            return
        self._build_shared_id_set_from(self._prescan_unique_ids, complete=True)
        self._prescan_done = True
        self._stats["bagpipe_shared_ids"] = float(len(self._shared_ids))
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
        ``.item()``) is unavoidable — allocating the padded buffers needs
        max_n on the host.  The count exchange is issued on the side stream
        with the same ``stream.wait_stream(current_stream)`` ordering as the
        payload all_gather, and max_n is read back on that same stream (see
        the comments in the body).

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

    def _dense_all_reduce_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Dense all_reduce of shared-ID gradients using a global index mapping."""
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

        # 向量化 fid -> dense 缓冲行号: _shared_ids_tensor 有序,
        # searchsorted 命中即行号（与 _global_id_to_index 的值一致）。
        # 不做 .any() 早退 —— 那是一次强制同步; 空 valid 集合时 all_reduce
        # 聚合的是零缓冲, _apply_aggregated 对空 id 集自然跳过。
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
            # 同上: index_put_ 在主流上写入 dense_grads, all_reduce 在侧流
            # 上消费 —— 先建立主流 -> 侧流的顺序。
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                work = dist.all_reduce(dense_grads, async_op=True)
        else:
            work = dist.all_reduce(dense_grads, async_op=True)

        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ids"] += float(len(valid_indices))
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

        work_obj = _DenseWork(
            work, dense_grads, valid_indices, valid_ids, dim, self.device
        )
        return valid_ids, valid_grads, work_obj
