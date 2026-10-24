import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..bagpipe_cache.planned import PlannedBagPipeController


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


def _controller(ps=None, capacity=1, sparse_collective="all_reduce"):
    return PlannedBagPipeController(
        None, ps or _PS(), cache_capacity=capacity, embedding_dim=2,
        table_offsets={"f0": 0}, master_table_name="table", device="cpu",
        id_extractor=lambda sf: sf.ids,
        sparse_collective=sparse_collective,
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


def _pooled_grads(ids, rank, step):
    return torch.arange(len(ids) * 2, dtype=torch.float32).reshape(len(ids) // 2, 2, 2) + rank * 10 + step + 1


def _distributed_pooled_reference(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        all_ids = [[[3, 3, 7, 8], [2, 8], [], []], [[3, 7], [2, 2, 8, 8], [3, 8], []]]
        batches = [_Features(ids, keys=["f0", "f1"]) for ids in all_ids[rank]]
        modes = [(collective, pooled) for collective in ("all_reduce", "all_gather") for pooled in (False, True)]
        controllers = [PlannedBagPipeController(
            None, _PS(), cache_capacity=1, embedding_dim=2,
            table_offsets={"f0": 0, "f1": 4}, master_table_name="table",
            device="cpu", id_extractor=lambda sf: sf.ids,
            sparse_collective=collective,
        ) for collective, _ in modes]
        tickets = []
        for controller in controllers:
            for step, batch in enumerate(batches):
                controller.prescan_batch(step, batch)
            controller.finalize_prescan()
            tickets.append([controller.enqueue(batch) for batch in batches])
        reference = {key: controllers[0].kv_client.rows[key].clone() for key in (2, 3, 7, 8)}
        for step, batch in enumerate(batches):
            ids = batch.ids.tolist()
            expected = (torch.stack([reference[key] for key in ids]).view(2, len(ids) // 2, 2)
                        .permute(1, 0, 2).reshape(len(ids) // 2, 4)) if ids else torch.empty((0, 4))
            for controller in controllers:
                controller.prefill_cache(batch, torch.device("cpu"))
                torch.testing.assert_close(controller.lookup(batch).values(), expected, rtol=0, atol=0)
            pooled = _pooled_grads(ids, rank, step)
            unique_ids, inverse, _ = tickets[0][step]
            batch_inverse = inverse.view(2, len(ids) // 2).t().reshape(-1)
            local_sum = torch.zeros((unique_ids.numel(), 2)).index_add_(0, batch_inverse, pooled.reshape(-1, 2))
            lr = 0.1 / (step + 1)
            for controller, (collective, use_pooled) in zip(controllers, modes):
                if use_pooled:
                    controller.update_pooled_grads(pooled, lr, step)
                else:
                    controller.update_grads("table", unique_ids, local_sum, lr, step)
                stats = controller.consume_stats(reset=False)
                assert stats["bagpipe_sparse_collective"] == collective
                if collective == "all_gather":
                    assert "bagpipe_all_reduce_payload_bytes" not in stats
                    assert stats["bagpipe_collective_output_bytes"] == 2 * stats["bagpipe_collective_input_bytes"]
                    work, gathered, plan, _, _ = controller._pending
                    assert plan.gather_counts == tuple(len(set(scan[step])) for scan in all_ids)
                    if work is not None:
                        work.wait()
                        gathered = gathered.view(2, plan.gather_rows.size(1), 2)
                        # Nonzero padding must not alter the next lookup or final SGD result.
                        for source, count in enumerate(plan.gather_counts):
                            gathered[source, count:].fill_(12345)
                if step == len(batches) - 1:
                    assert controller._pending[0] is None, "An empty global union must not submit a collective"
                    assert stats["bagpipe_collective_input_bytes"] == 0
            for key in reference:
                total = torch.zeros(2)
                for source in range(2):
                    source_ids = all_ids[source][step]
                    source_grads = _pooled_grads(source_ids, source, step).permute(1, 0, 2).reshape(-1, 2)
                    local = torch.zeros(2)
                    for source_key, grad in zip(source_ids, source_grads):
                        if source_key == key:
                            local += grad
                    total += local
                reference[key].add_(total, alpha=-lr / 2)
        for controller in controllers:
            controller.shutdown()
            assert torch.isfinite(controller._values).all()
            for row, key in enumerate(controller._global_ids.tolist()):
                torch.testing.assert_close(controller._values[row], reference[key], rtol=0, atol=0)
                if rank == 0:
                    torch.testing.assert_close(controller.kv_client.rows[key], reference[key], rtol=0, atol=0)
            assert controller.kv_client.writes == int(rank == 0)
    finally:
        dist.destroy_process_group()


class TestPlannedBagPipe(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Fused gather SGD requires CUDA")
    def test_fused_gather_sgd_exactly_matches_rank_ordered_fp32_updates(self):
        from ..bagpipe_cache.gather_sgd import gather_sgd_, triton
        if triton is None:
            self.skipTest("Optional Triton dependency is unavailable")
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(20261025)
        cases = [(([0, 2, 5], [2, 3]), 0.1), (([1, 4], [0, 3, 6]), 0.03),
                 (([], [2, 4]), 0.4), (([1, 3], []), 0.007), (([], []), 0.0)]
        for dimension in (2, 7, 128, 129):
            reference = torch.randn(8, dimension, device=device, generator=generator)
            candidate = reference.clone()
            for ranks, lr in cases:
                with self.subTest(dimension=dimension, ranks=ranks, lr=lr):
                    active = sorted(set(ranks[0] + ranks[1]))
                    active_rows = torch.tensor(active, dtype=torch.int64, device=device)
                    sources = torch.tensor(
                        [[ids.index(row) if row in ids else -1 for row in active] for ids in ranks],
                        dtype=torch.int64, device=device,
                    ).reshape(2, len(active))
                    width = max(map(len, ranks))
                    gathered = torch.randn(2, width, dimension, device=device, generator=generator)
                    aggregated = torch.zeros(len(active), dimension, device=device)
                    for rank, ids in enumerate(ranks):
                        positions = torch.tensor([active.index(row) for row in ids], device=device, dtype=torch.int64)
                        gathered[rank, len(ids):].fill_(12345)
                        aggregated.index_add_(0, positions, gathered[rank, :len(ids)])
                    reference.index_add_(0, active_rows, aggregated, alpha=-lr / 2)
                    self.assertTrue(gather_sgd_(candidate, gathered.reshape(2 * width, dimension),
                                                active_rows, sources, width, -lr / 2))
                    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

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

    def test_pooled_grad_matches_legacy_and_reference_with_duplicates_and_uneven_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_pooled_reference, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)

    def test_pooled_grad_validates_shape_dtype_device_and_update_order(self):
        controller = _controller()
        batch = _Features([3, 3])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        controller.enqueue(batch)
        with self.assertRaisesRegex(RuntimeError, "ordered update"):
            controller.update_pooled_grads(torch.ones((2, 1, 2)), 0.1, 0)
        controller.prefill_cache(batch, torch.device("cpu"))
        for invalid in (torch.ones((2, 2)), torch.ones((2, 1, 2), dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, "FP32 with the planned shape"):
                controller.update_pooled_grads(invalid, 0.1, 0)
        with self.assertRaisesRegex(ValueError, "device"):
            controller.update_pooled_grads(torch.empty((2, 1, 2), device="meta"), 0.1, 0)
        controller.update_pooled_grads(torch.ones((2, 1, 2), requires_grad=True), 0.1, 0)
        self.assertFalse(controller._pending[1].requires_grad)
        with self.assertRaisesRegex(RuntimeError, "ordered update"):
            controller.update_pooled_grads(torch.ones((2, 1, 2)), 0.1, 0)
        controller.shutdown()
        torch.testing.assert_close(controller.kv_client.rows[3], torch.tensor([2.8, 5.8]))

    def test_graph_payload_validates_batch_and_applies_before_next_lookup(self):
        for collective in ("all_reduce", "all_gather"):
            with self.subTest(collective=collective):
                controller = _controller(sparse_collective=collective)
                batches = [_Features([3, 3]), _Features([3, 7])]
                for step, batch in enumerate(batches):
                    controller.prescan_batch(step, batch)
                controller.finalize_prescan()
                for batch in batches:
                    controller.enqueue(batch)
                with self.assertRaisesRegex(RuntimeError, "prefill_cache"):
                    controller.planned_graph_inputs(batches[0])
                controller.prefill_cache(batches[0], torch.device("cpu"))
                with self.assertRaisesRegex(RuntimeError, "batch IDs differ"):
                    controller.planned_graph_inputs(_Features([3, 7]))
                lookup, pooled, count = controller.planned_graph_inputs(batches[0])
                self.assertEqual(count, 1)
                torch.testing.assert_close(controller._values[lookup], torch.tensor([[3., 6.], [3., 6.]]))
                for invalid in (torch.ones(2, 2), torch.ones(1, 2, dtype=torch.float64),
                                torch.empty(1, 2, device="meta")):
                    with self.assertRaisesRegex(ValueError, "planned FP32 rows and device"):
                        controller.update_packed_grads(invalid, 0.1, 0)
                payload = torch.zeros(count, 2).index_add_(0, pooled, torch.ones(2, 2))
                controller.update_packed_grads(payload, 0.1, 0)
                with self.assertRaisesRegex(RuntimeError, "ordered update"):
                    controller.planned_graph_inputs(batches[0])
                controller.prefill_cache(batches[1], torch.device("cpu"))
                lookup, _, count = controller.planned_graph_inputs(batches[1])
                payload.zero_()  # A graph may reuse its payload only after prefill consumed it.
                torch.testing.assert_close(controller._values[lookup], torch.tensor([[2.8, 5.8], [7., 14.]]))
                controller.update_packed_grads(torch.zeros(count, 2), 0.1, 1)
                controller.shutdown()

    def test_collective_option_and_single_rank_gather(self):
        with self.assertRaisesRegex(ValueError, "sparse_collective"):
            _controller(sparse_collective="invalid")
        controller = _controller(sparse_collective="all_gather")
        batch = _Features([3, 3])
        controller.prescan_batch(0, batch)
        controller.finalize_prescan()
        controller.enqueue(batch)
        controller.prefill_cache(batch, torch.device("cpu"))
        controller.update_pooled_grads(torch.ones((2, 1, 2)), 0.1, 0)
        self.assertIsNone(controller._pending[0])
        self.assertEqual(controller.consume_stats(reset=False)["bagpipe_collective_input_bytes"], 0)
        controller.shutdown()
        torch.testing.assert_close(controller.kv_client.rows[3], torch.tensor([2.8, 5.8]))

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
