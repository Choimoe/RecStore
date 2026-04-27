from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import torch

from ..KVClient import RecStoreClient


class TestHierKVIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path("/app/RecStore")
        cls.library_path = cls.repo_root / "build/lib/lib_recstore_ops.so"
        cls.config_path = cls.repo_root / "recstore_config.json"
        if not cls.library_path.exists():
            raise unittest.SkipTest(f"missing ops library: {cls.library_path}")
        if not cls.config_path.exists():
            raise unittest.SkipTest(f"missing config file: {cls.config_path}")

    def setUp(self) -> None:
        os.environ["RECSTORE_CONFIG"] = str(self.config_path)
        RecStoreClient._instance = None
        self.client = RecStoreClient(str(self.library_path))
        self.client.set_ps_backend("hierkv")

    def tearDown(self) -> None:
        RecStoreClient._instance = None

    def _new_table_name(self) -> str:
        return f"hierkv_it_{time.time_ns()}"

    def test_hierkv_local_lookup_and_update_round_trip(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(
            name=table_name,
            shape=(16, 4),
            dtype=torch.float32,
        )

        ids = torch.tensor([1, 3], dtype=torch.int64)
        before = self.client.local_lookup_flat(table_name, ids)
        self.assertEqual(self.client.ops.current_ps_backend(), "hierkv")
        self.assertTrue(torch.allclose(before, torch.zeros((2, 4), dtype=torch.float32)))

        grads = torch.ones((2, 4), dtype=torch.float32)
        self.client.local_update_flat(table_name, ids, grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32)
        self.assertTrue(torch.allclose(after, expected))

    def test_hierkv_prefetch_and_async_update_paths(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(
            name=table_name,
            shape=(16, 4),
            dtype=torch.float32,
        )

        ids = torch.tensor([2, 5], dtype=torch.int64)
        prefetch_id = self.client.prefetch(ids)
        prefetched = self.client.wait_and_get(prefetch_id, embedding_dim=4)
        self.assertTrue(torch.allclose(prefetched, torch.zeros((2, 4), dtype=torch.float32)))

        grads = torch.full((2, 4), 2.0, dtype=torch.float32)
        handle = self.client.update_async(table_name, ids, grads)
        self.client.wait(handle)

        updated = self.client.local_lookup_flat(table_name, ids)
        expected = torch.full((2, 4), -0.02, dtype=torch.float32)
        self.assertTrue(torch.allclose(updated, expected))


if __name__ == "__main__":
    unittest.main()
