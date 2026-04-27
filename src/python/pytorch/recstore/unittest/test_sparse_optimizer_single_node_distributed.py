import unittest

import torch

from .. import optimizer as optimizer_module
from ..optimizer import SparseSGD
from ..single_node_exchange import SparseGradPayload


class _FakeLegacyKVClient:
    def __init__(self):
        self.update_async_calls = []
        self.wait_calls = []
        self.local_update_flat_calls = []
        self._next_handle = 1

    def update_async(self, name, ids, grads):
        handle = self._next_handle
        self._next_handle += 1
        self.update_async_calls.append((name, ids.clone(), grads.clone(), handle))
        return handle

    def wait(self, handle):
        self.wait_calls.append(int(handle))

    def local_update_flat(self, name, ids, grads):
        self.local_update_flat_calls.append((name, ids.clone(), grads.clone()))


class _FakeModule:
    def __init__(self, trace, kv_client):
        self._config_names = ["table0"]
        self._trace = list(trace)
        self.kv_client = kv_client
        self.reset_trace_calls = 0

    def reset_trace(self):
        self.reset_trace_calls += 1
        self._trace = []


class _FakeFastPathModule(_FakeModule):
    def __init__(self, trace, kv_client, *, backend: str = "local_shm"):
        super().__init__(trace, kv_client)
        self.enable_single_node_distributed_fast_path = True
        self.single_node_distributed_mode = "single_node"
        self.single_node_owner_policy = "hash_mod_world_size"
        self.single_node_ps_backend = str(backend)


class _FakeDist:
    def __init__(self, *, rank, world_size, initialized=True):
        self._rank = rank
        self._world_size = world_size
        self._initialized = initialized

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


class TestSparseOptimizerSingleNodeDistributed(unittest.TestCase):
    def setUp(self):
        self._original_dist = optimizer_module.torch.distributed
        self._original_exchange_sparse_grads = getattr(
            optimizer_module,
            "exchange_sparse_grads",
            None,
        )

    def tearDown(self):
        optimizer_module.torch.distributed = self._original_dist
        if self._original_exchange_sparse_grads is None:
            delattr(optimizer_module, "exchange_sparse_grads")
        else:
            optimizer_module.exchange_sparse_grads = self._original_exchange_sparse_grads

    def test_fast_path_disabled_keeps_legacy_async_update_and_flush_wait(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeModule(
            trace=[
                (
                    "table0",
                    torch.tensor([5, 5, 8], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [2.0, 2.0],
                            [4.0, 4.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)

        optimizer.step()

        self.assertEqual(len(kv_client.update_async_calls), 1)
        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(kv_client.local_update_flat_calls, [])
        _, ids, grads, handle = kv_client.update_async_calls[0]
        self.assertTrue(torch.equal(ids, torch.tensor([5, 8], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor(
                    [
                        [3.0, 3.0],
                        [4.0, 4.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(handle, 1)
        self.assertEqual(mod.reset_trace_calls, 1)

        optimizer.flush()

        self.assertEqual(kv_client.wait_calls, [1])

    def test_fast_path_step_uses_owner_local_update_and_flush_is_noop(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([4, 5], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [10.0, 10.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        exchange_calls = []

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            exchange_calls.append((payload.clone(), world_size, backend))
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([1], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([4], dtype=torch.int64),
                    grads=torch.tensor(
                        [
                            [2.5, 2.5],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(exchange_calls), 1)
        self.assertEqual(len(kv_client.update_async_calls), 0)
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        self.assertEqual(kv_client.wait_calls, [])
        table_name, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(table_name, "table0")
        self.assertTrue(torch.equal(ids, torch.tensor([4], dtype=torch.int64)))
        self.assertTrue(
            torch.allclose(
                grads,
                torch.tensor([[3.5, 3.5]], dtype=torch.float32),
            )
        )
        self.assertEqual(mod.reset_trace_calls, 1)

        optimizer.flush()

        self.assertEqual(kv_client.wait_calls, [])
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)

    def test_hierkv_fast_path_uses_same_owner_local_update_flow(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([6, 7], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [9.0, 9.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
            backend="hierkv",
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            del payload, world_size, backend
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                ),
                SparseGradPayload(
                    rank=1,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([1], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[4.0, 4.0]], dtype=torch.float32),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.step()

        self.assertEqual(len(kv_client.update_async_calls), 0)
        self.assertEqual(len(kv_client.local_update_flat_calls), 1)
        table_name, ids, grads = kv_client.local_update_flat_calls[0]
        self.assertEqual(table_name, "table0")
        self.assertTrue(torch.equal(ids, torch.tensor([6], dtype=torch.int64)))
        self.assertTrue(torch.allclose(grads, torch.tensor([[5.0, 5.0]], dtype=torch.float32)))

    def test_perf_stats_capture_fast_path_exchange_and_local_update(self):
        kv_client = _FakeLegacyKVClient()
        mod = _FakeFastPathModule(
            trace=[
                (
                    "table0",
                    torch.tensor([6, 7], dtype=torch.int64),
                    torch.tensor(
                        [
                            [1.0, 1.0],
                            [9.0, 9.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            ],
            kv_client=kv_client,
            backend="hierkv",
        )
        optimizer = SparseSGD([mod], lr=0.1)
        optimizer_module.torch.distributed = _FakeDist(rank=0, world_size=2)

        def fake_exchange_sparse_grads(payload, *, world_size, backend):
            del payload, world_size, backend
            return [
                SparseGradPayload(
                    rank=0,
                    destination_ranks=torch.tensor([0], dtype=torch.int64),
                    source_ranks=torch.tensor([0], dtype=torch.int64),
                    row_positions=torch.tensor([0], dtype=torch.int64),
                    fused_ids=torch.tensor([6], dtype=torch.int64),
                    grads=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
                ),
            ]

        optimizer_module.exchange_sparse_grads = fake_exchange_sparse_grads

        optimizer.reset_perf_stats()
        optimizer.step()
        stats = optimizer.consume_perf_stats()

        self.assertGreaterEqual(stats["update_trace_merge_ms"], 0.0)
        self.assertGreaterEqual(stats["update_owner_exchange_ms"], 0.0)
        self.assertGreaterEqual(stats["update_local_apply_ms"], 0.0)
        self.assertEqual(stats["update_async_enqueue_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
