import time
import tempfile
import unittest
from datetime import timedelta
from multiprocessing.managers import BaseManager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..bagpipe_cache.planned import PlannedBagPipeController
from ..optim import OptimizationPluginRegistry


class _Features:
    def __init__(self, ids, lengths=None, keys=None):
        self.ids = torch.tensor(ids, dtype=torch.int64)
        self._lengths = torch.ones(len(ids), dtype=torch.int32) if lengths is None else torch.tensor(lengths)
        self._keys = keys or ["f0"]

    def keys(self):
        return self._keys

    def lengths(self):
        return self._lengths


class _Server:
    """Independent global-mean SGD reference, shared by test trainers."""
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.rows = {key: [float(key), float(key * 2)] for key in range(512)}
        self.pending = {}
        self.committed = -1

    def pull(self, ids):
        with self.lock:
            return [self.rows[key][:] for key in ids]

    def submit(self, ids, grads, step, rank, world, lr, divisor):
        with self.lock:
            contributions = self.pending.setdefault(step, {})
            contributions[rank] = (ids, grads)
            if len(contributions) == world:
                total = {}
                for source in range(world):
                    keys, values = contributions[source]
                    for key, value in zip(keys, values):
                        row = total.setdefault(key, [0., 0.])
                        for col in range(2):
                            row[col] += value[col]
                for key, value in total.items():
                    self.rows[key] = [self.rows[key][col] - lr * value[col] / divisor for col in range(2)]
                self.committed = step
                del self.pending[step]

    def done(self, step):
        with self.lock:
            return self.committed >= step


class _PS:
    def __init__(self, server=None):
        self.server = server or _Server()
        self.prefetches = {}
        self.next_handle = 0
        self.fail_wait = False
        self.pulled = 0

    def pull(self, name, ids):
        self.pulled += ids.numel()
        return torch.tensor(self.server.pull(ids.tolist()), dtype=torch.float32).reshape(-1, 2)

    def prefetch(self, ids):
        self.next_handle += 1
        self.prefetches[self.next_handle] = self.pull("table", ids).clone()
        return self.next_handle

    def wait_and_get(self, handle, dim, device="cpu"):
        return self.prefetches.pop(handle).to(device)

    def training_step_async(self, name, ids, grads, **kwargs):
        self.server.submit(ids.cpu().tolist(), grads.cpu().tolist(),
                           **{k: kwargs[k] for k in ("step", "rank", "world", "lr", "divisor")})
        return kwargs["step"]

    def training_step_wait(self, step):
        if self.fail_wait:
            raise OSError("simulated commit failure")
        deadline = time.monotonic() + 20
        while not self.server.done(step):
            if time.monotonic() > deadline:
                raise TimeoutError("test PS commit timed out")
            time.sleep(.001)

    def training_step_close(self, *args, **kwargs):
        pass


def _controller(ps=None, capacity=1, **kwargs):
    return PlannedBagPipeController(
        None, ps or _PS(), cache_capacity=capacity, embedding_dim=2,
        table_offsets={"f0": 0}, master_table_name="table", device="cpu",
        id_extractor=lambda sf: sf.ids, lr=0.1, **kwargs,
    )


class _Manager(BaseManager):
    pass


_Manager.register("Server", _Server)


def _distributed_reference(rank, rendezvous, server, aggregation, prescan):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        # More than 64 rows overall, a late first touch, an empty rank, and
        # cross-rank reuse after eviction. Enqueue all steps before updating
        # to catch stale prefetch results, including same-count ID changes.
        all_ids = [[[3], list(range(100, 180)), [3], [7], [3]],
                   [[], [3], [7], list(range(200, 280)), [3]]]
        batches = [_Features(ids) for ids in all_ids[rank]]
        controller = _controller(_PS(server), aggregation=aggregation, prescan=prescan)
        if prescan:
            for step, batch in enumerate(batches):
                controller.prescan_batch(step, batch)
            controller.finalize_prescan()
            assert all(t.device.type == "cpu" for p in controller._plans for t in vars(p).values())
        tickets = [controller.enqueue(batch) for batch in batches]
        stat_fields = set(controller.consume_stats())
        reference = {key: torch.tensor([float(key), float(key * 2)]) for key in range(512)}
        for step, batch in enumerate(batches):
            controller.prefill_cache(batch, torch.device("cpu"))
            expected = torch.stack([reference[key] for key in batch.ids.tolist()]) if batch.ids.numel() else torch.empty((0, 2))
            torch.testing.assert_close(controller.lookup(batch).values(), expected)
            ids, _, _ = tickets[step]
            grads = torch.full((ids.numel(), 2), float(rank + step + 1))
            controller.update_grads("table", ids, grads, 0.1, step)
            for source in range(2):
                for key in all_ids[source][step]:
                    reference[key] -= 0.1 * (source + step + 1) / 2
            assert controller._metadata.size() <= 64
            assert controller._values.shape == (64, 2)
            assert set(controller.consume_stats()) == stat_fields
        controller.shutdown()
        touched = sorted(set(key for steps in all_ids for batch in steps for key in batch))
        torch.testing.assert_close(torch.tensor(server.pull(touched)), torch.stack([reference[key] for key in touched]))
        assert not controller.kv_client.prefetches
    finally:
        dist.destroy_process_group()


class TestPlannedBagPipe(unittest.TestCase):
    def test_validation_snapshot_includes_evicted_updated_rows(self):
        from types import SimpleNamespace
        from model_zoo.rs_demo.runtime.validation import export_validation_state

        embedding = SimpleNamespace(_fusion_k=20, _embedding_bag_configs=[
            SimpleNamespace(name="f0", num_embeddings=512, embedding_dim=2)])
        batches = [_Features(range(80)), _Features([7, 95])]
        expected_ids = torch.unique(torch.cat([batch.ids for batch in batches]))
        for prescan in (False, True):
            with self.subTest(prescan=prescan), tempfile.TemporaryDirectory() as directory:
                controller = _controller(prescan=prescan)
                try:
                    if prescan:
                        for step, batch in enumerate(batches):
                            controller.prescan_batch(step, batch)
                        controller.finalize_prescan()
                    for step, batch in enumerate(batches):
                        ids, _, _ = controller.enqueue(batch)
                        controller.prefill_cache(batch, "cpu")
                        controller.update_grads("table", ids, torch.ones((ids.numel(), 2)), .1, step)
                    path = export_validation_state(Path(directory) / "state.pt",
                                                   embedding, torch.nn.Linear(2, 1), controller)
                    state = torch.load(path, weights_only=True)
                    torch.testing.assert_close(state["global_ids"], expected_ids)
                    torch.testing.assert_close(state["values"], controller.kv_client.pull("table", expected_ids))
                    self.assertEqual(controller._committed_step, 1)
                    self.assertLess(controller._metadata.size(), expected_ids.numel())
                finally:
                    controller.shutdown()

    def test_native_metadata_lru_oversized_admission_and_remote_updates(self):
        from recstore import load_ops_library

        load_ops_library()
        metadata = torch.classes.recstore_ops.BagPipeCacheMetadata(2)
        ids = lambda values: torch.tensor(values, dtype=torch.int64)
        slots, sources, evicted = metadata.admit(ids([3, 7, 9]))
        self.assertEqual(evicted, 1)
        self.assertEqual(slots.unique().numel(), slots.numel())
        self.assertEqual(set(sources.tolist()), {1, 2})
        metadata.lookup(ids([7]))  # Touch 7, so the next admission evicts 9.
        metadata.admit(ids([11]))
        positions, _, missing = metadata.lookup(ids([3, 7, 9, 11]))
        self.assertEqual(positions.tolist(), [1, 3])
        self.assertEqual(missing.tolist(), [0, 2])
        slots, rows, invalidated = metadata.commit(
            ids([7, 11]), ids([7]), torch.tensor([True]), True)
        self.assertEqual(invalidated, 1)  # Remote-only 11 must be invalidated.
        self.assertEqual(rows.tolist(), [0])
        self.assertEqual(metadata.size(), 1)
        slots, rows, invalidated = metadata.commit(
            ids([7]), ids([7]), torch.tensor([False]), True)
        self.assertEqual((invalidated, slots.numel(), metadata.size()), (1, 0, 0))
        self.assertEqual(metadata.previous(0, ids([3]), ids([3, 7])).tolist(), [-1])
        self.assertEqual(metadata.previous(1, ids([3, 9]), ids([9])).tolist(), [0, -1])
        self.assertEqual(metadata.prefetch(ids([3, 9]), ids([0, -1]), -1).tolist(), [9])

    def test_single_runtime_for_both_planning_modes(self):
        self.assertEqual([name for name in OptimizationPluginRegistry.available() if name.startswith("bagpipe")], ["bagpipe"])
        for prescan in (False, True):
            plugin = OptimizationPluginRegistry.create(
                "bagpipe", prescan=prescan, embedding_module=None, kv_client=_PS(),
                cache_capacity=1, embedding_dim=2, lookahead=4,
                table_offsets={"f0": 0}, table_sizes={"f0": 100},
                master_table_name="table", device=torch.device("cpu"),
                id_extractor=lambda sf: sf.ids,
            )
            try:
                self.assertIsInstance(plugin._controller, PlannedBagPipeController)
                self.assertEqual(plugin._controller._use_prescan, prescan)
            finally:
                plugin.shutdown()

    def test_feature_order_duplicates_and_bounded_cache(self):
        ps = _PS()
        controller = PlannedBagPipeController(
            None, ps, cache_capacity=1, embedding_dim=2, lr=0.1,
            table_offsets={"f0": 0, "f1": 4}, master_table_name="table",
            device="cpu", id_extractor=lambda sf: sf.ids,
        )
        batch = _Features([3, 3, 7, 8], keys=["f0", "f1"])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        ids, inverse, count = controller.enqueue(batch)
        self.assertEqual(inverse.tolist(), [0, 0, 1, 2])
        controller.prefill_cache(batch, torch.device("cpu"))
        torch.testing.assert_close(controller.lookup(batch).values(), torch.tensor([[3., 6., 7., 14.], [3., 6., 8., 16.]]))
        grads = torch.zeros((ids.numel(), 2)).index_add_(0, inverse, torch.ones((count, 2)))
        controller.update_grads("table", ids, grads, .1, 0)
        controller.shutdown()
        torch.testing.assert_close(torch.tensor(ps.server.pull([3])), torch.tensor([[2.8, 5.8]]))

    def test_real_collective_and_ps_paths_match_global_mean_under_eviction(self):
        with _Manager() as manager:
            for aggregation in ("collective", "ps"):
                for prescan in (False, True):
                    with self.subTest(aggregation=aggregation, prescan=prescan), tempfile.TemporaryDirectory() as directory:
                        mp.spawn(_distributed_reference,
                                 args=(str(Path(directory) / "rendezvous"), manager.Server(), aggregation, prescan),
                                 nprocs=2, join=True)

    def test_schedule_mismatch_and_commit_failure_are_explicit(self):
        controller = _controller()
        batch = _Features([3, 7])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        with self.assertRaisesRegex(RuntimeError, "batch IDs differ"):
            controller.enqueue(_Features([7, 3]))
        with self.assertRaisesRegex(RuntimeError, "bag lengths"):
            controller.enqueue(_Features([3, 7], lengths=[0, 2]))
        ids, _, _ = controller.enqueue(batch)
        controller.prefill_cache(batch, torch.device("cpu"))
        controller.update_grads("table", ids, torch.ones((2, 2)), .1, 0)
        controller.kv_client.fail_wait = True
        with self.assertRaisesRegex(OSError, "commit failure"):
            controller.shutdown()
        self.assertFalse(controller._closed)
        controller.shutdown(flush=False)

    def test_independent_batch_does_not_wait_for_prior_commit(self):
        controller = _controller(prescan=False)
        batches = [_Features([3]), _Features([7])]
        tickets = [controller.enqueue(batch) for batch in batches]
        controller.prefill_cache(batches[0], "cpu")
        controller.update_grads("table", tickets[0][0], torch.ones((1, 2)), .1, 0)
        controller.kv_client.fail_wait = True
        controller.prefill_cache(batches[1], "cpu")
        self.assertEqual(controller._stats["bagpipe_dependency_wait_skipped"], 1)
        controller.kv_client.fail_wait = False
        controller.shutdown()


if __name__ == "__main__":
    unittest.main()
