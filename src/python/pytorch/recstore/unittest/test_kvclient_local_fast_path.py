import unittest

import torch

from ..KVClient import RecStoreClient


class _FakeOps:
    def __init__(self, backend: str = "local_shm"):
        self.backend = backend
        self.lookup_calls = []
        self.update_calls = []
        self.backend_switch_calls = []

    def current_ps_backend(self) -> str:
        return self.backend

    def set_ps_backend(self, backend: str) -> None:
        self.backend = backend
        self.backend_switch_calls.append(backend)

    def local_lookup_flat(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        self.lookup_calls.append((keys.clone(), int(embedding_dim)))
        rows = keys.numel()
        return torch.arange(rows * int(embedding_dim), dtype=torch.float32).view(rows, int(embedding_dim))

    def local_update_flat(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        self.update_calls.append((table_name, keys.clone(), grads.clone()))


class TestKVClientLocalFastPath(unittest.TestCase):
    def _build_client(self, backend: str = "local_shm") -> RecStoreClient:
        client = object.__new__(RecStoreClient)
        client.ops = _FakeOps(backend=backend)
        client._part_policy = {}
        client._tensor_meta = {"table_a": {"shape": (16, 4), "dtype": torch.float32}}
        client._full_data_shape = {"table_a": (16, 4)}
        client._data_name_list = {"table_a"}
        client._gdata_name_list = {"table_a"}
        client._role = "default"
        client._next_async_handle = 1
        client._pending_async_ops = {}
        client._initialized = True
        return client

    def test_local_lookup_flat_uses_explicit_local_op(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        out = client.local_lookup_flat("table_a", keys)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(len(client.ops.lookup_calls), 1)
        called_keys, called_dim = client.ops.lookup_calls[0]
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertEqual(called_dim, 4)

    def test_local_update_flat_uses_explicit_local_op(self):
        client = self._build_client()

        keys = torch.tensor([7, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        client.local_update_flat("table_a", keys, grads)

        self.assertEqual(len(client.ops.update_calls), 1)
        table_name, called_keys, called_grads = client.ops.update_calls[0]
        self.assertEqual(table_name, "table_a")
        self.assertTrue(torch.equal(called_keys, keys))
        self.assertTrue(torch.equal(called_grads, grads))

    def test_local_flat_ops_allow_hierkv_backend(self):
        client = self._build_client(backend="hierkv")

        keys = torch.tensor([7, 3], dtype=torch.int64)
        grads = torch.ones((2, 4), dtype=torch.float32)
        out = client.local_lookup_flat("table_a", keys)
        client.local_update_flat("table_a", keys, grads)

        self.assertEqual(out.shape, (2, 4))
        self.assertEqual(len(client.ops.lookup_calls), 1)
        self.assertEqual(len(client.ops.update_calls), 1)

    def test_local_lookup_flat_fails_loudly_for_non_local_backend(self):
        client = self._build_client(backend="grpc")

        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_lookup_flat("table_a", torch.tensor([1], dtype=torch.int64))

        self.assertEqual(client.ops.lookup_calls, [])

    def test_local_update_flat_fails_loudly_for_non_local_backend(self):
        client = self._build_client(backend="brpc")

        with self.assertRaisesRegex(RuntimeError, "local_shm or hierkv"):
            client.local_update_flat(
                "table_a",
                torch.tensor([1], dtype=torch.int64),
                torch.ones((1, 4), dtype=torch.float32),
            )

        self.assertEqual(client.ops.update_calls, [])

    def test_set_ps_backend_switches_backend_explicitly(self):
        client = self._build_client(backend="grpc")

        client.set_ps_backend("local_shm")

        self.assertEqual(client.ops.current_ps_backend(), "local_shm")
        self.assertEqual(client.ops.backend_switch_calls, ["local_shm"])


if __name__ == "__main__":
    unittest.main()
