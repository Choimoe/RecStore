"""Exact resident BagPipe with prescanned sparse collective layouts.

This opt-in DLRM path requires the complete working set to fit in memory. It
stores authoritative FP32 rows, exchanges only FP32 gradients during training,
and averages across ranks to match TorchRec/DDP global-batch SGD semantics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _Plan:
    raw_ids: torch.Tensor
    unique_ids: torch.Tensor
    inverse: torch.Tensor
    lookup_rows: torch.Tensor
    active_rows: torch.Tensor
    local_rows: torch.Tensor


class PlannedBagPipeController:
    def __init__(
        self, embedding_module, kv_client, *, cache_capacity, embedding_dim,
        table_offsets, master_table_name, device, id_extractor,
        lookahead_value=4, **_,
    ):
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self.device = torch.device(device)
        self.embedding_dim = int(embedding_dim)
        self.master_table_name = master_table_name
        self._keys = list(table_offsets)
        self._id_extractor = id_extractor
        # Match the existing BagPipe configuration: one cache set has 64 rows.
        self._capacity_rows = int(cache_capacity) * 64
        if self._capacity_rows <= 0 or not self._keys:
            raise ValueError("Planned BagPipe requires positive capacity and feature keys")
        if len(set(table_offsets.values())) != len(self._keys):
            raise ValueError("Planned BagPipe requires distinct fused table offsets")
        self.lookahead_value = max(1, int(lookahead_value))
        self.max_inflight_prefetch = self.lookahead_value
        self._prescan = []
        self._plans = []
        self._queue = deque()
        self._enqueued = 0
        self._current_batch = -1
        self._updated = 0
        self._pending = None
        self._values = None
        self._global_ids = None
        self._closed = False
        self._stats = {}
        self._world = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    def _batch_ids(self, sparse_features):
        if list(sparse_features.keys()) != self._keys:
            raise ValueError("Planned BagPipe feature order differs from its table layout")
        ids = self._id_extractor(sparse_features).to(self.device, dtype=torch.int64)
        lengths = sparse_features.lengths()
        if ids.ndim != 1 or ids.numel() % len(self._keys) or lengths.numel() != ids.numel():
            raise ValueError("Planned BagPipe requires one ID per feature per sample")
        torch._assert_async(torch.all(lengths == 1), "Planned BagPipe requires all bag lengths to be 1")
        return ids

    @staticmethod
    def _check_ids(actual, expected):
        if actual.shape != expected.shape:
            raise RuntimeError("Planned BagPipe batch shape differs from the prescan")
        if actual is not expected:
            torch._assert_async(torch.all(actual == expected), "Planned BagPipe batch IDs differ from the prescan")

    def prescan_batch(self, batch_num, sparse_features):
        if self._plans or int(batch_num) != len(self._prescan):
            raise RuntimeError("Planned BagPipe prescan must be sequential and precede training")
        self._prescan.append(self._batch_ids(sparse_features).cpu().clone())

    def _build_layouts(self, scans):
        unique_start = time.perf_counter()
        global_ids = torch.unique(torch.cat([ids for scan in scans for ids in scan]))
        self._stats["bagpipe_plan_unique_ms"] = (time.perf_counter() - unique_start) * 1e3
        if global_ids.numel() > self._capacity_rows:
            raise RuntimeError(
                f"Planned BagPipe working set has {global_ids.numel()} rows, "
                f"exceeding the configured {self._capacity_rows}-row limit"
            )
        layout_start = time.perf_counter()
        plans = []
        for step, raw_ids in enumerate(self._prescan):
            unique_ids, inverse = torch.unique(raw_ids, return_inverse=True)
            active_ids = torch.unique(torch.cat([scan[step] for scan in scans]))
            plans.append(_Plan(*[tensor.to(self.device) for tensor in (
                raw_ids, unique_ids, inverse, torch.searchsorted(global_ids, raw_ids),
                torch.searchsorted(global_ids, active_ids), torch.searchsorted(active_ids, unique_ids),
            )]))
        self._global_ids = global_ids.to(self.device)
        self._values = torch.empty((global_ids.numel(), self.embedding_dim), dtype=torch.float32, device=self.device)
        self._stats["bagpipe_plan_layout_ms"] = (time.perf_counter() - layout_start) * 1e3
        return global_ids, plans

    def finalize_prescan(self):
        if self._plans:
            return
        start = time.perf_counter()
        scans = [self._prescan]
        if self._world > 1:
            scans = [None] * self._world
            dist.all_gather_object(scans, self._prescan)
        self._stats["bagpipe_plan_exchange_ms"] = (time.perf_counter() - start) * 1e3
        if not self._prescan or any(len(scan) != len(self._prescan) for scan in scans):
            raise RuntimeError("Planned BagPipe requires equally many nonzero prescan batches on every rank")
        previous_threads = torch.get_num_threads()
        try:
            # Small per-batch sorts/searches lose to CPU thread-pool overhead.
            # Restore the training thread budget even if planning fails.
            if previous_threads != 1:
                torch.set_num_threads(1)
            global_ids, plans = self._build_layouts(scans)
        finally:
            if previous_threads != 1:
                torch.set_num_threads(previous_threads)
        self._stats["bagpipe_plan_cpu_threads"] = 1.0
        load_start = time.perf_counter()
        pull_ms = copy_ms = 0.0
        error = None
        try:
            for begin in range(0, global_ids.numel(), 4096):
                ids = global_ids[begin:begin + 4096]
                pull_start = time.perf_counter()
                values = self.kv_client.pull(self.master_table_name, ids)
                pull_ms += (time.perf_counter() - pull_start) * 1e3
                if values.shape != (ids.numel(), self.embedding_dim) or values.dtype != torch.float32:
                    raise RuntimeError("PS returned an invalid embedding shape or dtype")
                copy_start = time.perf_counter()
                self._values[begin:begin + ids.numel()].copy_(values)
                copy_ms += (time.perf_counter() - copy_start) * 1e3
        except Exception as exc:
            error = exc
        self._stats.update(
            bagpipe_plan_load_ms=(time.perf_counter() - load_start) * 1e3,
            bagpipe_plan_pull_ms=pull_ms, bagpipe_plan_copy_ms=copy_ms,
        )
        ready_start = time.perf_counter()
        ok = torch.tensor([int(error is None)], device=self.device)
        if self._world > 1:
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not bool(ok.item()):
            raise RuntimeError("Planned BagPipe PS initialization failed on at least one rank") from error
        self._stats["bagpipe_plan_ready_ms"] = (time.perf_counter() - ready_start) * 1e3
        self._plans = plans
        self._prescan.clear()
        self._stats.update(
            bagpipe_plan_build_ms=(time.perf_counter() - start) * 1e3,
            bagpipe_plan_batches=float(len(plans)),
            bagpipe_plan_metadata_bytes=float(
                self._global_ids.numel() * self._global_ids.element_size()
                + sum(t.numel() * t.element_size() for plan in plans for t in vars(plan).values())
            ),
        )

    def enqueue(self, sparse_features):
        if not self._plans or self._closed:
            raise RuntimeError("Planned BagPipe requires a completed prescan before training")
        plan = self._plans[self._enqueued % len(self._plans)]
        self._check_ids(self._batch_ids(sparse_features), plan.raw_ids)
        self._queue.append((self._enqueued, sparse_features))
        self._enqueued += 1
        return plan.unique_ids, plan.inverse, int(plan.raw_ids.numel())

    def _wait_pending(self):
        if self._pending is None:
            return
        work, grads, rows, lr = self._pending
        if work is not None:
            work.wait()
        # Every active row appears once; the same averaged gradient reaches
        # all replicas before any next-step lookup can observe their values.
        self._values.index_add_(0, rows, grads, alpha=-lr / self._world)
        self._pending = None

    def prefill_cache(self, sparse_features, compute_device):
        start = time.perf_counter()
        self._wait_pending()
        if not self._queue:
            raise RuntimeError("Planned BagPipe consumed a batch that was not enqueued")
        step, prepared = self._queue.popleft()
        if prepared is not sparse_features:
            raise RuntimeError("Planned BagPipe consumed batches in a different order")
        self._current_batch = step
        self._stats["bagpipe_prefill_ms"] = (time.perf_counter() - start) * 1e3

    def lookup(self, sparse_features):
        from torchrec.sparse.jagged_tensor import KeyedTensor

        if self._current_batch < 0:
            raise RuntimeError("Planned BagPipe lookup requires prefill_cache first")
        plan = self._plans[self._current_batch % len(self._plans)]
        self._check_ids(self._batch_ids(sparse_features), plan.raw_ids)
        rows = self._values.index_select(0, plan.lookup_rows)
        batch_size = plan.raw_ids.numel() // len(self._keys)
        values = rows.view(len(self._keys), batch_size, self.embedding_dim).permute(1, 0, 2).reshape(batch_size, len(self._keys) * self.embedding_dim)
        return KeyedTensor(keys=self._keys, values=values, length_per_key=[self.embedding_dim] * len(self._keys))

    def update_grads(self, table_name, unique_ids, summed_grads, lr, batch_num):
        start = time.perf_counter()
        if table_name != self.master_table_name:
            raise ValueError("Planned BagPipe requires a single fused master table")
        if batch_num != self._current_batch or batch_num != self._updated or self._pending is not None:
            raise RuntimeError("Planned BagPipe requires exactly one ordered update per batch")
        plan = self._plans[batch_num % len(self._plans)]
        self._check_ids(unique_ids, plan.unique_ids)
        if summed_grads.shape != (unique_ids.numel(), self.embedding_dim) or summed_grads.dtype != torch.float32:
            raise ValueError("Planned BagPipe gradients must be FP32 with the planned shape")
        payload = torch.zeros((plan.active_rows.numel(), self.embedding_dim), device=self.device)
        payload.index_copy_(0, plan.local_rows, summed_grads)
        work = dist.all_reduce(payload, async_op=True) if self._world > 1 and payload.numel() else None
        self._pending = (work, payload, plan.active_rows, float(lr))
        self._updated += 1
        self._stats["bagpipe_update_ms"] = (time.perf_counter() - start) * 1e3
        self._stats["bagpipe_all_reduce_ids"] = float(plan.active_rows.numel())
        self._stats["bagpipe_all_reduce_payload_bytes"] = float(payload.numel() * payload.element_size())

    def schedule_cleanup(self, current_batch):
        pass

    def consume_stats(self, *, reset=True):
        stats = dict(self._stats)
        stats["bagpipe_cache_entries"] = float(self._global_ids.numel()) if self._global_ids is not None else 0.0
        stats["bagpipe_resident_bytes"] = float(self._values.numel() * 4) if self._values is not None else 0.0
        stats["bagpipe_lookahead"] = float(self.lookahead_value)
        if reset:
            self._stats.clear()
        return stats

    def shutdown(self, *, flush=True):
        if self._closed:
            return
        if not flush:
            self._closed = True
            return
        self._wait_pending()
        if self._world > 1:
            dist.barrier()
        error = None
        if self._rank == 0 and self._updated:
            try:
                for begin in range(0, self._global_ids.numel(), 4096):
                    self.kv_client.emb_write_values(
                        self.master_table_name, self._global_ids[begin:begin + 4096],
                        self._values[begin:begin + 4096].contiguous(),
                    )
            except Exception as exc:
                error = exc
        ok = torch.tensor([int(error is None)], device=self.device)
        if self._world > 1:
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not bool(ok.item()):
            raise RuntimeError("Planned BagPipe final PS writeback failed") from error
        self._closed = True
