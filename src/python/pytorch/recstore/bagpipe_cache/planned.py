"""Bounded BagPipe cache with CPU access plans and PS-owned SGD commits.

Both online and prescanned plans use this runtime. ``collective`` aggregates
on trainers (phase 1); ``ps`` sends local gradients directly to PS (phase 2).
Only cache slots and the current batch's tensors live on the GPU.
"""
from __future__ import annotations

import secrets
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
    active_ids: torch.Tensor
    local_rows: torch.Tensor
    exclusive: torch.Tensor
    previous_update: torch.Tensor


class BagPipeCacheController:
    def __init__(self, embedding_module, kv_client, *, cache_capacity,
                 embedding_dim, table_offsets, master_table_name, device,
                 id_extractor, lookahead_value=4, prescan=False,
                 aggregation="ps", lr=0.01, **_):
        if aggregation not in ("ps", "collective"):
            raise ValueError("BagPipe aggregation must be ps or collective")
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self.device = torch.device(device)
        self.embedding_dim = int(embedding_dim)
        self.master_table_name = master_table_name
        self._keys = list(table_offsets)
        self._id_extractor = id_extractor
        # Preserve the existing public cache unit: 64 rows per cache set.
        self._capacity_rows = int(cache_capacity) * 64
        if self._capacity_rows <= 0 or not self._keys:
            raise ValueError("BagPipe requires positive capacity and feature keys")
        if len(set(table_offsets.values())) != len(self._keys):
            raise ValueError("BagPipe requires distinct fused table offsets")
        self.lookahead_value = max(1, int(lookahead_value))
        self.max_inflight_prefetch = min(4, self.lookahead_value)
        self._use_prescan = bool(prescan)
        self.aggregation = aggregation
        self.lr = float(lr)
        self._world = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        session = torch.tensor([secrets.randbelow(2**63 - 1) + 1], dtype=torch.int64, device=self.device)
        if self._world > 1:
            dist.broadcast(session, src=0)
        self._session = int(session.item())
        self._prescan = []
        self._plans = []
        self._queue = deque()
        self._enqueued = 0
        self._updated = 0
        self._committed_step = -1
        self._current_batch = -1
        self._current_plan = None
        self._current_rows = None
        self._pending = None
        self._closed = False
        self._stats = {}
        from .. import load_ops_library

        load_ops_library()
        self._metadata = torch.classes.recstore_ops.BagPipeCacheMetadata(self._capacity_rows)
        self._values = torch.empty((self._capacity_rows, self.embedding_dim),
                                   dtype=torch.float32, device=self.device)

    def _add(self, name, value):
        self._stats[name] = self._stats.get(name, 0.0) + float(value)

    def _batch_ids(self, sparse_features):
        if list(sparse_features.keys()) != self._keys:
            raise ValueError("BagPipe feature order differs from its table layout")
        ids = self._id_extractor(sparse_features).to(dtype=torch.int64)
        lengths = sparse_features.lengths()
        if ids.ndim != 1 or ids.numel() % len(self._keys) or lengths.numel() != ids.numel():
            raise ValueError("BagPipe requires one ID per feature per sample")
        torch._assert_async(torch.all(lengths == 1), "BagPipe requires all bag lengths to be 1")
        return ids

    @staticmethod
    def _check_ids(actual, expected):
        if actual.shape != expected.shape:
            raise RuntimeError("BagPipe batch IDs differ from the prescan")
        if actual is not expected:
            torch._assert_async(torch.all(actual == expected.to(actual.device)),
                                "BagPipe batch IDs differ from the prescan")

    def prescan_batch(self, batch_num, sparse_features):
        if self._plans or int(batch_num) != len(self._prescan):
            raise RuntimeError("BagPipe prescan must be sequential and precede training")
        self._prescan.append(self._batch_ids(sparse_features).cpu().clone())

    def _make_plan(self, step, raw_ids, rank_ids):
        unique_ids, inverse = torch.unique(raw_ids, return_inverse=True)
        per_rank = [torch.unique(ids) for ids in rank_ids]
        active, counts = torch.unique(torch.cat(per_rank), return_counts=True)
        local_rows = torch.searchsorted(active, unique_ids)
        previous = self._metadata.previous(step, unique_ids, active)
        return _Plan(raw_ids, unique_ids, inverse, active, local_rows,
                     counts[local_rows] == 1, previous)

    def finalize_prescan(self):
        if self._plans:
            return
        start = time.perf_counter()
        scans = [self._prescan]
        if self._world > 1:
            scans = [None] * self._world
            dist.all_gather_object(scans, self._prescan)
        if not self._prescan or any(len(scan) != len(self._prescan) for scan in scans):
            raise RuntimeError("BagPipe requires equally many nonzero prescan batches on every rank")
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            self._plans = [self._make_plan(step, raw, [scan[step] for scan in scans])
                           for step, raw in enumerate(self._prescan)]
        finally:
            torch.set_num_threads(threads)
        self._prescan.clear()
        self._use_prescan = True
        self._metadata.clear_plan()
        self._add("bagpipe_plan_build_ms", (time.perf_counter() - start) * 1e3)
        self._add("bagpipe_plan_batches", len(self._plans))
        self._add("bagpipe_plan_metadata_bytes", sum(t.numel() * t.element_size()
                  for plan in self._plans for t in vars(plan).values()))

    def enqueue(self, sparse_features):
        if self._closed:
            raise RuntimeError("BagPipe is closed")
        step = self._enqueued
        raw = self._batch_ids(sparse_features)
        if self._use_prescan:
            if step >= len(self._plans):
                raise RuntimeError("BagPipe requires a prescan covering every training step")
            plan = self._plans[step]
            self._check_ids(raw, plan.raw_ids)
        else:
            raw = raw.cpu()
            rank_ids = [raw]
            if self._world > 1:
                rank_ids = [None] * self._world
                # Online planning exchanges IDs only, never embedding values.
                dist.all_gather_object(rank_ids, raw)
            plan = self._make_plan(step, raw, rank_ids)
        ids_dev = plan.unique_ids.to(self.device)
        inverse_dev = plan.inverse.to(self.device)
        prefetch = None
        inflight = sum(item[4] is not None for item in self._queue)
        if inflight < self.max_inflight_prefetch:
            ids = self._metadata.prefetch(plan.unique_ids, plan.previous_update,
                                          self._committed_step)
            if ids.numel():
                prefetch = (ids, self.kv_client.prefetch(ids))
                self._add("bagpipe_prefetch_ids", ids.numel())
        self._queue.append((step, sparse_features, plan, (ids_dev, inverse_dev), prefetch))
        self._enqueued += 1
        return ids_dev, inverse_dev, int(plan.raw_ids.numel())

    def _submit(self, ids, grads, step, lr):
        return self.kv_client.sparse_gradient_commit_async(
            self.master_table_name, ids, grads, session=self._session, step=step,
            rank=self._rank, world=self._world, lr=lr,
            divisor=self._world if self.aggregation == "ps" else 1)

    def _wait_pending(self):
        if self._pending is None:
            return
        handle, plan, grads, lr, step = self._pending
        start = time.perf_counter()
        self.kv_client.sparse_gradient_commit_wait(handle)
        self._pending = None
        self._committed_step = step
        # Local-only rows can be updated without a PS value round trip.
        slots, rows, invalidated = self._metadata.commit(
            plan.active_ids, plan.unique_ids, plan.exclusive,
            self.aggregation == "ps")
        self._add("bagpipe_invalidated_ids", invalidated)
        if slots.numel():
            slot_ids = slots.to(self.device)
            row_ids = rows.to(grads.device)
            local_grads = grads.index_select(0, row_ids)
            if self.aggregation == "ps":
                local_grads.div_(self._world)
            self._values.index_add_(0, slot_ids, local_grads, alpha=-lr)
        self._add("bagpipe_commit_wait_ms", (time.perf_counter() - start) * 1e3)

    def prefill_cache(self, sparse_features, compute_device):
        start = time.perf_counter()
        if not self._queue:
            raise RuntimeError("BagPipe consumed a batch that was not enqueued")
        step, prepared, plan, ticket, prefetch = self._queue[0]
        if prepared is not sparse_features:
            raise RuntimeError("BagPipe consumed batches in a different order")
        if self._pending is not None:
            if (plan.previous_update == self._pending[4]).any():
                self._wait_pending()
            else:
                self._add("bagpipe_dependency_wait_skipped", 1)
        self._queue.popleft()
        self._current_batch, self._current_plan = step, plan
        self._current_ticket = ticket
        rows = torch.empty((plan.unique_ids.numel(), self.embedding_dim), device=self.device)
        hit_positions, hit_slots, missing = self._metadata.lookup(plan.unique_ids)
        if hit_slots.numel():
            rows.index_copy_(0, hit_positions.to(self.device),
                             self._values.index_select(0, hit_slots.to(self.device)))
        self._add("bagpipe_cache_hit_ids", hit_slots.numel())
        demand = missing
        if prefetch is not None:
            pref_ids, handle = prefetch
            values = self.kv_client.wait_and_get(handle, self.embedding_dim, device=self.device)
            missing_ids = plan.unique_ids[missing]
            source = torch.searchsorted(pref_ids, missing_ids).clamp_max(pref_ids.numel() - 1)
            ready = pref_ids[source] == missing_ids
            positions = missing[ready]
            if positions.numel():
                rows.index_copy_(0, positions.to(self.device),
                                 values.index_select(0, source[ready].to(values.device)).to(self.device))
            demand = missing[~ready]
        if demand.numel():
            ids = plan.unique_ids[demand]
            values = self.kv_client.pull(self.master_table_name, ids)
            if values.shape != (demand.numel(), self.embedding_dim) or values.dtype != torch.float32:
                raise RuntimeError("PS returned invalid embedding shape or dtype")
            rows.index_copy_(0, demand.to(self.device), values.to(self.device))
            self._add("bagpipe_demand_ids", demand.numel())
        # The batch tensor also serves oversized batches; admission is bounded.
        slots, sources, evicted = self._metadata.admit(plan.unique_ids[missing])
        self._add("bagpipe_evicted_ids", evicted)
        if slots.numel():
            self._values.index_copy_(0, slots.to(self.device),
                                     rows.index_select(0, missing[sources].to(self.device)))
        self._current_rows = rows
        self._add("bagpipe_prefill_ms", (time.perf_counter() - start) * 1e3)

    def lookup(self, sparse_features):
        from torchrec.sparse.jagged_tensor import KeyedTensor
        if self._current_plan is None:
            raise RuntimeError("BagPipe lookup requires prefill_cache first")
        # enqueue validated the full schedule; consume also checks object order.
        values = self._current_rows.index_select(0, self._current_ticket[1])
        batch_size = values.size(0) // len(self._keys)
        values = values.view(len(self._keys), batch_size, self.embedding_dim).permute(1, 0, 2).reshape(batch_size, len(self._keys) * self.embedding_dim)
        return KeyedTensor(keys=self._keys, values=values,
                           length_per_key=[self.embedding_dim] * len(self._keys))

    def update_grads(self, table_name, unique_ids, summed_grads, lr, batch_num):
        if table_name != self.master_table_name or batch_num != self._current_batch or batch_num != self._updated:
            raise RuntimeError("BagPipe requires exactly one ordered fused update per batch")
        if float(lr) != self.lr:
            raise ValueError("BagPipe learning rate must match the PS session")
        plan = self._current_plan
        if unique_ids is not self._current_ticket[0]:
            self._check_ids(unique_ids, plan.unique_ids)
        if summed_grads.shape != (unique_ids.numel(), self.embedding_dim) or summed_grads.dtype != torch.float32:
            raise ValueError("BagPipe gradients must be FP32 with the planned shape")
        self._wait_pending()
        if self.aggregation == "collective":
            grads = torch.zeros((plan.active_ids.numel(), self.embedding_dim), device=self.device)
            grads.index_copy_(0, plan.local_rows.to(self.device), summed_grads)
            if self._world > 1 and grads.numel():
                dist.all_reduce(grads)
            grads.div_(self._world)
            owners = torch.arange(plan.active_ids.numel()) % self._world == self._rank
            handle = self._submit(plan.active_ids[owners], grads[owners.to(self.device)], batch_num, lr)
            self._add("bagpipe_all_reduce_payload_bytes", grads.numel() * 4)
        else:
            grads = summed_grads.detach()
            handle = self._submit(plan.unique_ids, grads, batch_num, lr)
            self._add("bagpipe_ps_gradient_bytes", grads.numel() * 4)
        self._pending = (handle, plan, grads, float(lr), batch_num)
        self._updated += 1
        self._current_rows = None

    def schedule_cleanup(self, current_batch):
        pass

    def consume_stats(self, *, reset=True):
        # CSV rows must have the same fields even before the first eviction,
        # commit, or cross-rank invalidation occurs.
        names = ("plan_build_ms", "plan_batches", "plan_metadata_bytes",
                 "prefetch_ids", "invalidated_ids", "commit_wait_ms",
                 "dependency_wait_skipped", "cache_hit_ids", "demand_ids",
                 "evicted_ids", "prefill_ms", "all_reduce_payload_bytes",
                 "ps_gradient_bytes")
        stats = {"bagpipe_" + name: self._stats.get("bagpipe_" + name, 0.0)
                 for name in names}
        stats.update(bagpipe_cache_entries=float(self._metadata.size()),
                     bagpipe_resident_bytes=float(self._values.numel() * 4),
                     bagpipe_lookahead=float(self.lookahead_value))
        if reset:
            self._stats.clear()
        return stats

    def shutdown(self, *, flush=True):
        if self._closed:
            return
        if flush:
            self._wait_pending()
        elif self._pending is not None:
            try:
                self.kv_client.sparse_gradient_commit_close(
                    self.master_table_name, session=self._session, step=self._updated - 1,
                    rank=self._rank, world=self._world, dim=self.embedding_dim, lr=self.lr,
                    divisor=self._world if self.aggregation == "ps" else 1, abort=True)
            except Exception:
                pass
            try:
                self.kv_client.sparse_gradient_commit_wait(self._pending[0])
            except Exception:
                pass
            self._pending = None
        while self._queue:
            prefetch = self._queue.popleft()[4]
            if prefetch is not None:
                try:
                    self.kv_client.wait_and_get(prefetch[1], self.embedding_dim)
                except Exception:
                    if flush:
                        raise
        if flush and self._updated:
            self.kv_client.sparse_gradient_commit_close(
                self.master_table_name, session=self._session, step=self._updated - 1,
                rank=self._rank, world=self._world, dim=self.embedding_dim, lr=self.lr,
                divisor=self._world if self.aggregation == "ps" else 1)
        self._closed = True


# Existing imports retain the name; planning no longer selects a storage mode.
PlannedBagPipeController = BagPipeCacheController
