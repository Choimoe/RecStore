from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import torch

from model_zoo.rs_demo.runners.recstore_runner import _add_sparse_id_stats
from model_zoo.rs_demo.data.dlrm_source import build_sparse_features
from tools.config.recstore_config_path import resolve_recstore_config_path
from ..KVClient import RecStoreClient


class TestSparseIdStats(unittest.TestCase):
    def test_rs_demo_adds_sparse_id_scale_fields(self) -> None:
        features = build_sparse_features(
            ["f1", "f2"],
            torch.tensor([1, 1, 2, 3, 3, 4], dtype=torch.int64),
            torch.tensor([2, 1, 1, 2], dtype=torch.int32),
        )
        row = {}

        _add_sparse_id_stats(
            row,
            features,
            {"f1": 0, "f2": 1 << 30},
            cache_capacity=1024,
            prefetch_depth=2,
        )

        self.assertEqual(row["batch_raw_ids"], 6)
        self.assertEqual(row["batch_unique_ids"], 4)
        self.assertEqual(row["batch_dedup_ratio"], 2.0 / 6.0)
        self.assertEqual(row["gpu_cache_capacity"], 1024)
        self.assertEqual(row["prefetch_depth"], 2)


class TestGpuTrainingCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.repo_root = Path(__file__).resolve().parents[5]
        cls.config_path = resolve_recstore_config_path()
        if not cls.config_path.exists():
            raise unittest.SkipTest(f"missing config file: {cls.config_path}")

    def setUp(self) -> None:
        os.environ["RECSTORE_CONFIG"] = str(self.config_path)
        RecStoreClient._instance = None
        self.client = RecStoreClient()
        self.client.set_ps_backend("hierkv")
        if not self.client.enable_gpu_cache(capacity=128, embedding_dim=4):
            raise unittest.SkipTest("GPU cache ops are not enabled")

    def tearDown(self) -> None:
        try:
            self.client.disable_gpu_cache()
        finally:
            RecStoreClient._instance = None

    def _new_table_name(self) -> str:
        return f"gpu_cache_it_{time.time_ns()}"

    def test_lookup_miss_fills_cache_and_second_lookup_hits(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")

        first = self.client.local_lookup_flat(table_name, ids)
        second = self.client.local_lookup_flat(table_name, ids)

        self.assertEqual(first.device.type, "cuda")
        self.assertTrue(torch.allclose(first, torch.zeros((3, 4), device="cuda")))
        self.assertTrue(torch.allclose(second, first))

    def test_update_invalidates_cached_rows_before_next_lookup_refills(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 9], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((2, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.client.local_update_flat(table_name, ids, grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

        refilled = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(refilled, expected))

    def test_apply_sgd_update_gpu_cache_keeps_cached_rows_visible(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 9], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((2, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))

        grads = torch.ones((2, 4), dtype=torch.float32, device="cuda")
        self.assertTrue(
            self.client.apply_sgd_update_gpu_cache(
                table_name,
                ids,
                grads,
                learning_rate=0.01,
            )
        )
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

    def test_invalidate_gpu_cache_evicts_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([11], dtype=torch.int64, device="cuda")

        before = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(before, torch.zeros((1, 4), device="cuda")))
        cached = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(cached, before))

        grads = torch.ones((1, 4), dtype=torch.float32)
        self.client.local_update_flat(table_name, ids.cpu(), grads)
        self.client.invalidate_gpu_cache(table_name, ids)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((1, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

    def test_update_invalidation_preserves_duplicate_id_updates(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([7, 7], dtype=torch.int64, device="cuda")

        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)

        grads = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.client.local_update_flat(table_name, ids, grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.03, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

    def test_emb_update_table_invalidates_cached_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([21, 23], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)

        grads = torch.ones((2, 4), dtype=torch.float32)
        self.client.ops.emb_update_table(table_name, ids.cpu(), grads)
        after = self.client.local_lookup_flat(table_name, ids)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

    def test_cpu_ids_update_clears_gpu_cache(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids_cuda = torch.tensor([17, 19], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids_cuda)
        self.client.local_lookup_flat(table_name, ids_cuda)

        grads_cpu = torch.ones((2, 4), dtype=torch.float32)
        self.client.local_update_flat(table_name, ids_cuda.cpu(), grads_cpu)
        after = self.client.local_lookup_flat(table_name, ids_cuda)

        expected = torch.full((2, 4), -0.01, dtype=torch.float32, device="cuda")
        self.assertTrue(torch.allclose(after, expected))

    def test_push_invalidates_cached_rows_before_next_lookup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([11, 12], dtype=torch.int64, device="cuda")
        self.client.local_lookup_flat(table_name, ids)
        self.client.local_lookup_flat(table_name, ids)

        replacement = torch.full((2, 4), 3.0, dtype=torch.float32)
        self.client.push(table_name, ids.cpu(), replacement)
        out = self.client.local_lookup_flat(table_name, ids)

        self.assertTrue(torch.allclose(out, replacement.to("cuda")))

    def test_switching_tables_invalidates_cache(self) -> None:
        table_a = self._new_table_name()
        table_b = self._new_table_name()
        self.client.init_data(name=table_a, shape=(64, 4), dtype=torch.float32)
        self.client.init_data(name=table_b, shape=(64, 4), dtype=torch.float32)
        ids = torch.tensor([13, 14], dtype=torch.int64, device="cuda")
        self.client.push(table_b, ids.cpu(), torch.full((2, 4), 5.0, dtype=torch.float32))

        self.client.local_lookup_flat(table_a, ids)
        self.client.local_lookup_flat(table_a, ids)

        out_b = self.client.local_lookup_flat(table_b, ids)

        self.assertTrue(torch.allclose(out_b, torch.full((2, 4), 5.0, device="cuda")))

    def test_duplicate_ids_preserve_request_order_after_cache_fill(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        seed_ids = torch.tensor([4, 6, 8, 10], dtype=torch.int64, device="cuda")
        seed_grads = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.client.local_update_flat(table_name, seed_ids, seed_grads)
        self.client.clear_gpu_cache()

        self.client.local_lookup_flat(
            table_name, torch.tensor([4, 6], dtype=torch.int64, device="cuda")
        )

        ids = torch.tensor([4, 8, 6, 10, 8], dtype=torch.int64, device="cuda")
        out = self.client.local_lookup_flat(table_name, ids)
        expected = torch.tensor(
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.03, -0.03, -0.03, -0.03],
                [-0.02, -0.02, -0.02, -0.02],
                [-0.04, -0.04, -0.04, -0.04],
                [-0.03, -0.03, -0.03, -0.03],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.assertTrue(torch.allclose(out, expected))

        cached = self.client.local_lookup_flat(table_name, ids)
        cached_expected = torch.tensor(
            [
                [-0.01, -0.01, -0.01, -0.01],
                [-0.03, -0.03, -0.03, -0.03],
                [-0.02, -0.02, -0.02, -0.02],
                [-0.04, -0.04, -0.04, -0.04],
                [-0.03, -0.03, -0.03, -0.03],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        self.assertTrue(torch.allclose(cached, cached_expected))

    def test_large_low_hit_lookups_bypass_gpu_cache_after_warmup(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(20_000, 4), dtype=torch.float32)

        ids = torch.arange(0, 4096, dtype=torch.int64, device="cuda")
        out = self.client.local_lookup_flat(table_name, ids)
        self.assertTrue(torch.allclose(out, torch.zeros((4096, 4), device="cuda")))

        bypass_ids = torch.arange(4096, 4096 + 4096, dtype=torch.int64, device="cuda")
        bypassed = self.client.local_lookup_flat(table_name, bypass_ids)
        self.assertTrue(torch.allclose(bypassed, torch.zeros((4096, 4), device="cuda")))
        self.assertTrue(self.client.is_gpu_cache_lookup_bypassed())

    def test_gpu_cache_rejects_reserved_sentinel_keys(self) -> None:
        table_name = self._new_table_name()
        self.client.init_data(name=table_name, shape=(64, 4), dtype=torch.float32)
        reserved_ids = torch.tensor(
            [
                torch.iinfo(torch.int64).max,
                torch.iinfo(torch.int64).max - 1,
            ],
            dtype=torch.int64,
            device="cuda",
        )

        old_value = os.environ.get("RECSTORE_VALIDATE_GPU_CACHE_KEYS")
        os.environ["RECSTORE_VALIDATE_GPU_CACHE_KEYS"] = "1"
        try:
            with self.assertRaisesRegex(RuntimeError, "reserved GPU cache sentinel"):
                self.client.local_lookup_flat(table_name, reserved_ids)
        finally:
            if old_value is None:
                os.environ.pop("RECSTORE_VALIDATE_GPU_CACHE_KEYS", None)
            else:
                os.environ["RECSTORE_VALIDATE_GPU_CACHE_KEYS"] = old_value


if __name__ == "__main__":
    unittest.main()
