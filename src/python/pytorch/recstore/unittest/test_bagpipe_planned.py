import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..bagpipe_cache.planned import PlannedBagPipeController
from ..bagpipe_cache.controller import BagPipeCacheController
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


class _PS:
    def __init__(self):
        self.rows = {key: torch.tensor([float(key), float(key * 2)]) for key in range(100)}
        self.writes = 0

    def pull(self, name, ids):
        return torch.stack([self.rows[key] for key in ids.tolist()])

    def emb_write_values(self, name, ids, values):
        self.writes += 1
        for key, value in zip(ids.tolist(), values):
            self.rows[key] = value.clone()


def _controller(ps=None, capacity=1):
    return PlannedBagPipeController(
        None, ps or _PS(), cache_capacity=capacity, embedding_dim=2,
        table_offsets={"f0": 0}, master_table_name="table", device="cpu",
        id_extractor=lambda sf: sf.ids,
    )


def _distributed_reference(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        ps = _PS()
        controller = _controller(ps)
        all_ids = [[[3], [7], [3]], [[], [3], [3]]]
        all_grads = [
            [[[2.0, 4.0]], [[1.0, 1.0]], [[2.0, 3.0]]],
            [[], [[3.0, 5.0]], [[4.0, 6.0]]],
        ]
        batches = [_Features(ids) for ids in all_ids[rank]]
        for step, batch in enumerate(batches):
            controller.prescan_batch(step, batch)
        controller.finalize_prescan()
        tickets = [controller.enqueue(batch) for batch in batches]
        reference = {key: ps.rows[key].clone() for key in (3, 7)}
        for step, batch in enumerate(batches):
            controller.prefill_cache(batch, torch.device("cpu"))
            expected = torch.stack([reference[key] for key in batch.ids.tolist()]) if batch.ids.numel() else torch.empty((0, 2))
            torch.testing.assert_close(controller.lookup(batch).values(), expected, rtol=0, atol=0)
            ids, _, _ = tickets[step]
            grads = torch.tensor(all_grads[rank][step]).reshape(-1, 2)
            controller.update_grads("table", ids, grads, 0.1, step)
            for key in reference:
                total = torch.zeros(2)
                for source in range(2):
                    for pos, source_key in enumerate(all_ids[source][step]):
                        if source_key == key:
                            total += torch.tensor(all_grads[source][step][pos])
                reference[key].add_(total, alpha=-0.1 / 2)
        controller.shutdown()
        for row, key in enumerate(controller._global_ids.tolist()):
            torch.testing.assert_close(controller._values[row], reference[key], rtol=0, atol=0)
            if rank == 0:
                torch.testing.assert_close(ps.rows[key], reference[key], rtol=0, atol=0)
        assert ps.writes == int(rank == 0), "Only the designated owner may persist replicas"
    finally:
        dist.destroy_process_group()


class TestPlannedBagPipe(unittest.TestCase):
    def test_single_bagpipe_entry_selects_controller_from_prescan(self):
        self.assertEqual(
            [name for name in OptimizationPluginRegistry.available() if name.startswith("bagpipe")],
            ["bagpipe"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown optimization plugin"):
            OptimizationPluginRegistry.create("bagpipe_planned")
        for prescan, controller_type in ((False, BagPipeCacheController), (True, PlannedBagPipeController)):
            with self.subTest(prescan=prescan):
                plugin = OptimizationPluginRegistry.create(
                    "bagpipe", prescan=prescan, embedding_module=None, kv_client=_PS(),
                    cache_capacity=1, embedding_dim=2, lookahead=4,
                    table_offsets={"f0": 0}, table_sizes={"f0": 100},
                    master_table_name="table", device=torch.device("cpu"),
                    id_extractor=lambda sf: sf.ids,
                )
                try:
                    self.assertIsInstance(plugin._controller, controller_type)
                    self.assertIs(plugin.create_sparse_optimizer([], lr=0.1).controller, plugin._controller)
                finally:
                    plugin.shutdown()

    def test_feature_order_and_duplicate_id_ticket_preserve_lookup_and_sgd(self):
        controller = PlannedBagPipeController(
            None, _PS(), cache_capacity=1, embedding_dim=2,
            table_offsets={"f0": 0, "f1": 4}, master_table_name="table",
            device="cpu", id_extractor=lambda sf: sf.ids,
        )
        batch = _Features([3, 3, 7, 8], keys=["f0", "f1"])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        ids, inverse, count = controller.enqueue(batch)
        self.assertEqual(count, 4)
        self.assertEqual(inverse.tolist(), [0, 0, 1, 2])
        controller.prefill_cache(batch, torch.device("cpu"))
        torch.testing.assert_close(controller.lookup(batch).values(), torch.tensor([
            [3.0, 6.0, 7.0, 14.0], [3.0, 6.0, 8.0, 16.0],
        ]))
        grads = torch.zeros((ids.numel(), 2)).index_add_(0, inverse, torch.ones((count, 2)))
        controller.update_grads("table", ids, grads, 0.1, 0)
        controller.shutdown()
        torch.testing.assert_close(controller.kv_client.rows[3], torch.tensor([2.8, 5.8]))

    def test_real_collective_matches_global_mean_sgd_with_late_touch_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_reference, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)

    def test_same_count_different_ids_and_nonunit_lengths_are_rejected(self):
        controller = _controller()
        controller.prescan_batch(0, _Features([3, 7]))
        controller.finalize_prescan()
        with self.assertRaisesRegex(RuntimeError, "batch IDs differ"):
            controller.enqueue(_Features([7, 3]))
        with self.assertRaisesRegex(RuntimeError, "bag lengths"):
            controller.enqueue(_Features([3, 7], lengths=[0, 2]))
        controller.shutdown(flush=False)

    def test_capacity_and_writeback_failures_are_explicit(self):
        controller = _controller()
        controller.prescan_batch(0, _Features(list(range(65))))
        previous_threads = torch.get_num_threads()
        with self.assertRaisesRegex(RuntimeError, "64-row limit"):
            controller.finalize_prescan()
        self.assertEqual(torch.get_num_threads(), previous_threads)
        ps = _PS()
        controller = _controller(ps)
        batch = _Features([3])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        self.assertEqual(torch.get_num_threads(), previous_threads)
        ids, _, _ = controller.enqueue(batch)
        controller.prefill_cache(batch, torch.device("cpu"))
        controller.update_grads("table", ids, torch.ones((1, 2)), 0.1, 0)
        def fail_write(*args):
            raise OSError("simulated transport failure")
        ps.emb_write_values = fail_write
        with self.assertRaisesRegex(RuntimeError, "PS writeback failed"):
            controller.shutdown()
        self.assertFalse(controller._closed)
        controller.shutdown(flush=False)


if __name__ == "__main__":
    unittest.main()
