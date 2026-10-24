from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from model_zoo.rs_demo.data import dlrm_source
from model_zoo.torchrec_dlrm.data.custom_dataloader import CustomCriteoDataset


class TestDlrmSourceFallback(unittest.TestCase):
    def test_get_default_cat_names_falls_back_without_torchrec(self) -> None:
        with mock.patch(
            "model_zoo.rs_demo.data.dlrm_source.importlib.import_module",
            side_effect=ModuleNotFoundError("torchrec"),
        ):
            cat_names = dlrm_source.get_default_cat_names()

        self.assertEqual(len(cat_names), 26)
        self.assertEqual(cat_names[0], "cat_0")
        self.assertEqual(cat_names[-1], "cat_25")

    def test_build_kjt_batch_uses_fallback_sparse_container(self) -> None:
        real_import_module = dlrm_source.importlib.import_module

        def _patched_import(name: str):
            if name in {"torchrec.datasets.criteo", "torchrec.sparse.jagged_tensor"}:
                raise ModuleNotFoundError(name)
            return real_import_module(name)

        dense = torch.zeros((2, 13), dtype=torch.float32)
        sparse = torch.arange(52, dtype=torch.int64).reshape(2, 26)
        labels = torch.zeros((2, 1), dtype=torch.float32)

        with mock.patch(
            "model_zoo.rs_demo.data.dlrm_source.importlib.import_module",
            side_effect=_patched_import,
        ):
            _, sparse_features = dlrm_source.build_kjt_batch_from_dense_sparse_labels(
                dense,
                sparse,
                labels,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(sparse_features.keys()), 26)
        self.assertTrue(
            torch.equal(
                sparse_features["cat_0"].values(),
                torch.tensor([0, 26], dtype=torch.int64),
            )
        )
        self.assertEqual(sparse_features["cat_0"].values().device.type, "cpu")

    def test_custom_criteo_dataset_batch_getitems_matches_single_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dense = np.arange(8 * 13, dtype=np.float32).reshape(8, 13)
            sparse = np.arange(8 * 26, dtype=np.int64).reshape(8, 26) - 50
            labels = np.arange(8, dtype=np.float32).reshape(8, 1)
            np.save(f"{tmpdir}/day_0_dense.npy", dense)
            np.save(f"{tmpdir}/day_0_sparse.npy", sparse)
            np.save(f"{tmpdir}/day_0_labels.npy", labels)

            dataset = CustomCriteoDataset(
                data_dir=tmpdir,
                stage="train",
                train_ratio=1.0,
                num_embeddings_per_feature=[7] * 26,
            )

            dense_b, sparse_b, labels_b = dataset.__getitems__([1, 3, 5])
            singles = [dataset[idx] for idx in [1, 3, 5]]

        self.assertEqual(tuple(dense_b.shape), (3, 13))
        self.assertEqual(tuple(sparse_b.shape), (3, 26))
        self.assertEqual(tuple(labels_b.shape), (3, 1))
        self.assertTrue(torch.equal(dense_b, torch.stack([s[0] for s in singles])))
        self.assertTrue(torch.equal(sparse_b, torch.stack([s[1] for s in singles])))
        self.assertTrue(torch.equal(labels_b, torch.stack([s[2] for s in singles])))
        self.assertEqual(dense_b.dtype, singles[0][0].dtype)
        self.assertEqual(sparse_b.dtype, singles[0][1].dtype)
        self.assertEqual(labels_b.dtype, singles[0][2].dtype)

    def test_cpu_fused_ids_match_kjt_feature_order(self) -> None:
        sparse = torch.arange(3 * 26, dtype=torch.int64).reshape(3, 26)
        names = [f"cat_{idx}" for idx in range(26)]
        offsets = {name: idx << 10 for idx, name in enumerate(names)}
        feature_offsets = torch.tensor(list(offsets.values()), dtype=torch.int64)

        unique_ids, inverse, raw_count = dlrm_source.prepare_fused_ids_from_sparse_batch(
            sparse, feature_offsets
        )
        _, sparse_features = dlrm_source.build_kjt_batch_from_dense_sparse_labels(
            torch.zeros((3, 13)), sparse, torch.zeros((3, 1))
        )
        expected = dlrm_source.convert_kjt_ids_to_fused_ids(sparse_features, offsets)
        expected_unique, expected_inverse = torch.unique(expected, return_inverse=True)

        self.assertEqual(raw_count, expected.numel())
        self.assertTrue(torch.equal(unique_ids, expected_unique))
        self.assertTrue(torch.equal(inverse, expected_inverse))

    def test_device_fused_ids_match_cpu_extraction(self) -> None:
        sparse = torch.arange(3 * 26, dtype=torch.int64).reshape(3, 26)
        names = [f"cat_{idx}" for idx in range(26)]
        offsets = {name: idx << 10 for idx, name in enumerate(names)}
        _, sparse_features = dlrm_source.build_kjt_batch_from_dense_sparse_labels(
            torch.zeros((3, 13)), sparse, torch.zeros((3, 1))
        )

        cpu_ids = dlrm_source.convert_kjt_ids_to_fused_ids(
            sparse_features, offsets
        )
        device_ids = dlrm_source.convert_kjt_ids_to_fused_ids_device(
            sparse_features, offsets
        )

        self.assertTrue(torch.equal(cpu_ids, device_ids))

    def test_resolve_default_table_sizes_uses_cap_instead_of_uniform_size(self) -> None:
        sizes = dlrm_source.resolve_num_embeddings_per_feature(5000)

        self.assertEqual(len(sizes), 26)
        self.assertEqual(sizes[0], 5000)
        self.assertEqual(sizes[5], 3)
        self.assertEqual(sizes[8], 63)
        self.assertLess(sum(sizes), 5000 * 26)


class _CountingBatchDataset(torch.utils.data.Dataset):
    def __init__(self, size=160000):
        self.size = size
        self.reads = 0
        self.samples_read = 0

    def __len__(self):
        return self.size

    def __getitems__(self, indices):
        self.reads += 1
        self.samples_read += len(indices)
        return torch.tensor(indices), self.reads


class TestContinuousDlrmLoader(unittest.TestCase):
    def _build(self, dataset, *, continuous, rank=0, world_size=2, steps=320):
        module = SimpleNamespace(CustomCriteoDataset=lambda **_: dataset)
        with mock.patch.dict("sys.modules", {"data.custom_dataloader": module}), mock.patch.dict(
            dlrm_source.os.environ,
            {"RS_DEMO_DATALOADER_CONTINUOUS": str(int(continuous)),
             "RS_DEMO_DATALOADER_NUM_WORKERS": "0", "RS_DEMO_DATALOADER_PIN_MEMORY": "0"},
        ):
            return dlrm_source.build_train_dataloader(
                Path.cwd(), ".", 1.0, 100, 1024,
                rank=rank, world_size=world_size, seed=20260330, steps=steps,
            )[1]

    def test_320_batches_match_original_indices_tails_and_fresh_reads_on_both_ranks(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                original_data, continuous_data = _CountingBatchDataset(), _CountingBatchDataset()
                original = self._build(original_data, continuous=False, rank=rank)
                continuous = self._build(continuous_data, continuous=True, rank=rank)
                self.assertEqual(original.batch_size, 1024)
                self.assertEqual(len(original), 79)
                self.assertIsNone(continuous.batch_size)
                self.assertEqual(len(continuous.batch_sampler.sampler), 80000)
                self.assertEqual(len(continuous), 320)
                self.assertIsNone(continuous.batch_sampler.batches)
                self.assertEqual(continuous_data.reads, 0)
                original_iter = iter(original)
                continuous_iter = iter(continuous)
                tails = []
                samples = 0
                first_indices = None
                for step in range(320):
                    try:
                        expected, expected_read = next(original_iter)
                    except StopIteration:
                        original_iter = iter(original)
                        expected, expected_read = next(original_iter)
                    actual, actual_read = next(continuous_iter)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual_read, step + 1)
                    self.assertEqual(actual_read, expected_read)
                    samples += actual.numel()
                    if actual.numel() != 1024:
                        tails.append((step, actual.numel()))
                    if step == 0:
                        first_indices = actual.clone()
                    if step == 79:
                        torch.testing.assert_close(actual, first_indices, rtol=0, atol=0)
                        self.assertEqual(actual_read, 80, "Repeated indices must trigger a new dataset read")
                with self.assertRaises(StopIteration):
                    next(continuous_iter)
                self.assertEqual(tails, [(78, 128), (157, 128), (236, 128), (315, 128)])
                self.assertEqual(samples, 324096)
                self.assertEqual(continuous_data.reads, 320)
                self.assertEqual(continuous_data.samples_read, samples)
                self.assertEqual(original_data.samples_read, samples)
                self.assertEqual(len(continuous.batch_sampler.batches), 79)
                self.assertGreaterEqual(continuous.batch_sampler.index_prepare_ms, 0)

    def test_continuous_rejects_unpartitioned_missing_steps_and_empty_data(self):
        for world_size, steps in ((None, 320), (1, 320), (2, None), (2, 0), (2, -1)):
            with self.subTest(world_size=world_size, steps=steps):
                with self.assertRaisesRegex(ValueError, "world_size > 1 and positive steps"):
                    self._build(_CountingBatchDataset(), continuous=True, world_size=world_size, steps=steps)
        with self.assertRaisesRegex(ValueError, "nonempty sampler"):
            self._build(_CountingBatchDataset(0), continuous=True)

    def test_continuous_flag_is_forwarded_to_both_benchmark_lanes(self):
        from tools.benchmarks.e2e.custom.runtime import _recstore_nccl_env, _torchrec_nccl_env

        with mock.patch.dict(dlrm_source.os.environ, {"RS_DEMO_DATALOADER_CONTINUOUS": "1"}):
            for environment in (_recstore_nccl_env(), _torchrec_nccl_env()):
                self.assertEqual(environment["RS_DEMO_DATALOADER_CONTINUOUS"], "1")


if __name__ == "__main__":
    unittest.main()
