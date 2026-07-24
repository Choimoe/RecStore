from __future__ import annotations

import unittest
import tempfile
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

            batch = dataset.__getitems__([1, 3, 5])
            singles = [dataset[idx] for idx in [1, 3, 5]]

        self.assertEqual(len(batch), 3)
        for batched, single in zip(batch, singles):
            for batched_tensor, single_tensor in zip(batched, single):
                self.assertTrue(torch.equal(batched_tensor, single_tensor))
                self.assertEqual(batched_tensor.dtype, single_tensor.dtype)

    def test_resolve_default_table_sizes_uses_cap_instead_of_uniform_size(self) -> None:
        sizes = dlrm_source.resolve_num_embeddings_per_feature(5000)

        self.assertEqual(len(sizes), 26)
        self.assertEqual(sizes[0], 5000)
        self.assertEqual(sizes[5], 3)
        self.assertEqual(sizes[8], 63)
        self.assertLess(sum(sizes), 5000 * 26)


if __name__ == "__main__":
    unittest.main()
