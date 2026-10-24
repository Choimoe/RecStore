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
    pooled_rows: torch.Tensor
    gather_rows: torch.Tensor
    gather_sources: torch.Tensor
    gather_counts: tuple[int, ...]


class PlannedBagPipeController:
    def __init__(
        self, embedding_module, kv_client, *, cache_capacity, embedding_dim,
        table_offsets, master_table_name, device, id_extractor,
        lookahead_value=4, sparse_collective="all_reduce", **_,
    ):
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self.device = torch.device(device)
        self.embedding_dim = int(embedding_dim)
        self.master_table_name = master_table_name
        if sparse_collective not in ("all_reduce", "all_gather"):
            raise ValueError("Planned BagPipe sparse_collective must be all_reduce or all_gather")
        self.sparse_collective = sparse_collective
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
        self._gather_sgd_submitted = False
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
            batch_size = raw_ids.numel() // len(self._keys)
            local_rows = torch.searchsorted(active_ids, unique_ids)
            gather_rows = torch.empty((0, 0), dtype=torch.int64)
            gather_sources = torch.empty((0, 0), dtype=torch.int64)
            gather_counts = ()
            if self.sparse_collective == "all_gather":
                rank_ids = [torch.unique(scan[step]) for scan in scans]
                gather_counts = tuple(ids.numel() for ids in rank_ids)
                gather_rows = torch.zeros((self._world, max(gather_counts)), dtype=torch.int64)
                if self._world == 2:
                    gather_sources = torch.full((2, active_ids.numel()), -1, dtype=torch.int64)
                for rank, ids in enumerate(rank_ids):
                    gather_rows[rank, :ids.numel()] = torch.searchsorted(active_ids, ids)
                    if self._world == 2:
                        gather_sources[rank, gather_rows[rank, :ids.numel()]] = torch.arange(ids.numel())
            # Match the dense model's [batch, feature, dimension] layout once.
            lookup_rows = torch.searchsorted(global_ids, raw_ids).view(len(self._keys), batch_size).t().reshape(-1)
            payload_inverse = inverse if self.sparse_collective == "all_gather" else local_rows[inverse]
            pooled_rows = payload_inverse.view(len(self._keys), batch_size).t().reshape(-1)
            plans.append(_Plan(*[tensor.to(self.device) for tensor in (
                raw_ids, unique_ids, inverse, lookup_rows,
                torch.searchsorted(global_ids, active_ids), local_rows, pooled_rows, gather_rows, gather_sources,
            )], gather_counts=gather_counts))
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
            bagpipe_plan_gather_source_bytes=float(sum(
                plan.gather_sources.numel() * plan.gather_sources.element_size() for plan in plans
            )),
            bagpipe_plan_metadata_bytes=float(
                self._global_ids.numel() * self._global_ids.element_size()
                + sum(t.numel() * t.element_size() for plan in plans for t in vars(plan).values() if isinstance(t, torch.Tensor))
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
        work, grads, plan, lr, send_payload = self._pending
        if work is not None:
            work.wait()
        if self.sparse_collective == "all_gather":
            if self._world == 2 and self.device.type == "cuda":
                first_submit_start = time.perf_counter() if not self._gather_sgd_submitted else None
                from .gather_sgd import gather_sgd_
                if gather_sgd_(self._values, grads, plan.active_rows, plan.gather_sources,
                               plan.gather_rows.size(1), -lr / self._world):
                    if first_submit_start is not None:
                        self._stats["bagpipe_gather_sgd_first_submit_ms"] = (time.perf_counter() - first_submit_start) * 1e3
                    self._gather_sgd_submitted = True
                    self._stats["bagpipe_gather_sgd_fused"] = 1.0
                    self._pending = None
                    return
            aggregated = torch.zeros((plan.active_rows.numel(), self.embedding_dim), device=self.device, dtype=torch.float32)
            gathered = grads.view(self._world, plan.gather_rows.size(1), self.embedding_dim)
            # Each rank's real rows are unique. Skip padding so zero entries
            # cannot create an atomic hotspot on row 0; retain rank order.
            for rank, count in enumerate(plan.gather_counts):
                if count:
                    aggregated.index_add_(0, plan.gather_rows[rank, :count], gathered[rank, :count])
            grads = aggregated
        # Every active row appears once; the same averaged gradient reaches
        # all replicas before any next-step lookup can observe their values.
        self._values.index_add_(0, plan.active_rows, grads, alpha=-lr / self._world)
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
        values = rows.view(batch_size, len(self._keys) * self.embedding_dim)
        return KeyedTensor(keys=self._keys, values=values, length_per_key=[self.embedding_dim] * len(self._keys))

    def planned_graph_inputs(self, sparse_features):
        """Validate the consumed batch before a graph reads the resident rows."""
        if self._current_batch < 0:
            raise RuntimeError("Planned BagPipe graph lookup requires prefill_cache first")
        plan = self._update_plan(self._current_batch)
        self._check_ids(self._batch_ids(sparse_features), plan.raw_ids)
        return plan.lookup_rows, plan.pooled_rows, self._payload_rows(plan)

    def _update_plan(self, batch_num):
        if batch_num != self._current_batch or batch_num != self._updated or self._pending is not None:
            raise RuntimeError("Planned BagPipe requires exactly one ordered update per batch")
        return self._plans[batch_num % len(self._plans)]

    def _submit_update(self, plan, payload, lr, start):
        send_payload = payload
        communicate = self._world > 1 and payload.numel() > 0
        work = None
        if communicate:
            if self.sparse_collective == "all_gather":
                payload = torch.empty((self._world * send_payload.size(0), self.embedding_dim), device=self.device, dtype=torch.float32)
                work = dist.all_gather_into_tensor(payload, send_payload, async_op=True)
            else:
                work = dist.all_reduce(payload, async_op=True)
        # The asynchronous gather must retain its local input as well as output.
        self._pending = (work, payload, plan, float(lr), send_payload)
        self._updated += 1
        self._stats["bagpipe_update_ms"] = (time.perf_counter() - start) * 1e3
        self._stats["bagpipe_sparse_collective"] = self.sparse_collective
        self._stats["bagpipe_collective_active_ids"] = float(plan.active_rows.numel())
        self._stats["bagpipe_collective_input_bytes"] = float(send_payload.numel() * send_payload.element_size()) if communicate else 0.0
        self._stats["bagpipe_collective_output_bytes"] = float(payload.numel() * payload.element_size()) if communicate else 0.0
        self._stats[f"bagpipe_{self.sparse_collective}_ids"] = float(send_payload.size(0)) if communicate else 0.0
        self._stats[f"bagpipe_{self.sparse_collective}_payload_bytes"] = self._stats["bagpipe_collective_input_bytes"]

    def _payload_rows(self, plan):
        return plan.gather_rows.size(1) if self.sparse_collective == "all_gather" else plan.active_rows.numel()

    def _empty_payload(self, plan):
        return torch.zeros((self._payload_rows(plan), self.embedding_dim), dtype=torch.float32, device=self.device)

    def update_packed_grads(self, payload, lr, batch_num):
        """Submit a graph-owned payload; prefill must consume it before replay."""
        start = time.perf_counter()
        plan = self._update_plan(batch_num)
        if (
            payload.shape != (self._payload_rows(plan), self.embedding_dim)
            or payload.dtype != torch.float32 or payload.device != self._values.device
            or not payload.is_contiguous()
        ):
            raise ValueError("Planned BagPipe payload must match the planned FP32 rows and device")
        self._submit_update(plan, payload, lr, start)

    def update_grads(self, table_name, unique_ids, summed_grads, lr, batch_num):
        start = time.perf_counter()
        if table_name != self.master_table_name:
            raise ValueError("Planned BagPipe requires a single fused master table")
        plan = self._update_plan(batch_num)
        self._check_ids(unique_ids, plan.unique_ids)
        if summed_grads.shape != (unique_ids.numel(), self.embedding_dim) or summed_grads.dtype != torch.float32:
            raise ValueError("Planned BagPipe gradients must be FP32 with the planned shape")
        payload = self._empty_payload(plan)
        if self.sparse_collective == "all_gather":
            payload[:unique_ids.numel()].copy_(summed_grads)
        else:
            payload.index_copy_(0, plan.local_rows, summed_grads)
        self._submit_update(plan, payload, lr, start)

    def update_pooled_grads(self, pooled_grads, lr, batch_num):
        """Aggregate [batch, feature, dimension] gradients into the planned payload."""
        start = time.perf_counter()
        plan = self._update_plan(batch_num)
        expected_shape = (plan.raw_ids.numel() // len(self._keys), len(self._keys), self.embedding_dim)
        if pooled_grads.shape != expected_shape or pooled_grads.dtype != torch.float32:
            raise ValueError("Planned BagPipe pooled gradients must be FP32 with the planned shape")
        if pooled_grads.device != self._values.device:
            raise ValueError("Planned BagPipe pooled gradients must be on the resident values' device")
        payload = self._empty_payload(plan)
        payload.index_add_(0, plan.pooled_rows, pooled_grads.detach().reshape(-1, self.embedding_dim))
        self._submit_update(plan, payload, lr, start)

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
