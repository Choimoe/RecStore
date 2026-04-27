from __future__ import annotations

import unittest
from unittest import mock

import torch

from model_zoo.rs_demo.data import dlrm_source


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
            )

        self.assertEqual(len(sparse_features.keys()), 26)
        self.assertTrue(
            torch.equal(
                sparse_features["cat_0"].values(),
                torch.tensor([0, 26], dtype=torch.int64),
            )
        )

    def test_recstore_embeddingbag_imports_without_torchrec(self) -> None:
        import importlib
        import sys

        original_modules = {
            name: sys.modules.get(name)
            for name in [
                "python.pytorch.torchrec_kv.EmbeddingBag",
                "torchrec",
                "torchrec.sparse.jagged_tensor",
                "torchrec.modules.embedding_configs",
            ]
        }
        for name in list(original_modules.keys()):
            sys.modules.pop(name, None)

        real_import_module = importlib.import_module

        def _patched_import(name: str, package: str | None = None):
            if name.startswith("torchrec"):
                raise ModuleNotFoundError(name)
            return real_import_module(name, package)

        with mock.patch("importlib.import_module", side_effect=_patched_import):
            module = importlib.import_module("src.python.pytorch.torchrec_kv.EmbeddingBag")

        cfg = {"name": "table0", "embedding_dim": 4, "num_embeddings": 8, "feature_names": ["cat_0"]}
        fake_client = mock.Mock()
        fake_client.init_data.return_value = None
        ebc = module.RecStoreEmbeddingBagCollection(
            embedding_bag_configs=[cfg],
            kv_client=fake_client,
            initialize_tables=True,
        )
        keyed = module.KeyedTensor(
            keys=["cat_0"],
            values=torch.ones((2, 4), dtype=torch.float32),
            length_per_key=[4],
        )

        self.assertEqual(ebc.embedding_bag_configs()[0].embedding_dim, 4)
        self.assertEqual(ebc.feature_keys, ["cat_0"])
        self.assertTrue(torch.equal(keyed["cat_0"], torch.ones((2, 4), dtype=torch.float32)))

        for name, mod in original_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
