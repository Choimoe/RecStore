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

        同步预算: 尺寸协商 (count all_gather + 一次 ``.item()``) 不可避免 —
        padded 缓冲的分配需要 host 侧的 max_n。但 count 交换的内容 (本 rank
        的个数) 是 host 已知常量, 不依赖主流上任何计算, 因此把它发射在侧流
        上且**不**建立主流 → 侧流顺序: ``.item()`` 只等待这个 8 字节集合通
        信本身, 而不是等主流积压 (backward/replay) 排空后再等通信。payload
        的 all_gather 读取主流写入的梯度, 保持原来的
        ``stream.wait_stream(current_stream)`` 顺序。

        本方法必须先于 ``_dense_all_reduce_async`` 调用: 侧流是串行的,
        dense all_reduce 会先 ``wait_stream`` 主流, 若 count 交换排在其后,
        ``.item()`` 又会被主流积压挡住, 顺序优化就白做了。
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
            # count 交换无主流依赖: 直接在侧流上发射 (侧流此刻空闲 —
            # 上一步的聚合已在 prefill barrier 排空)。
            with torch.cuda.stream(stream):
                dist.all_gather(n_list, n_tensor)
        else:
            dist.all_gather(n_list, n_tensor)
        max_n = max(int(ni.item()) for ni in n_list)

        if max_n == 0:
            # max_n 是全 rank 一致的最大值, 这里早退不会让对端悬空。
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

        ids_dev = ids.to(self.device, dtype=torch.int64)
        grads_dev = grads.to(self.device, dtype=torch.float32)
        if not ids_dev.is_contiguous():
            ids_dev = ids_dev.contiguous()
        if not grads_dev.is_contiguous():
            grads_dev = grads_dev.contiguous()

        # 哨兵补齐: 超出本 rank 真实个数的行填 -1, wait() 端用布尔掩码恢复
        # 各 rank 的贡献, 不再需要逐 rank 的 count .item() 同步。
        padded_ids = torch.full((max_n,), -1, dtype=torch.int64, device=self.device)
        padded_ids[:n] = ids_dev
        padded_grads = torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
        padded_grads[:n] = grads_dev

        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                       for _ in range(world_size)]

        if stream is not None:
            # 缓冲在主流上写入, 集合通信在侧流上发射: 必须先让侧流等待
            # 主流, 否则 NCCL kernel 可能读到旧的补齐缓冲 (write-after-read
            # 竞态, kernel launch 延迟通常掩盖它, 但语义上是 UB)。
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
