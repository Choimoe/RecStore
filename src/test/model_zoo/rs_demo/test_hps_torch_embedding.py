from __future__ import annotations

import tempfile
import types
import unittest
import importlib
import sys
from pathlib import Path
from unittest import mock

import torch

from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners.hps_torch_runner import HpsTorchRunner
from model_zoo.rs_demo.runtime.hps_torch_embedding import (
    HpsTableSpec,
    HpsTorchEmbeddingBagCollection,
    import_hps_torch_module,
)


class _FakeFeature:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


class _FakeKJT:
    def __init__(self, mapping: dict[str, _FakeFeature]) -> None:
        self._mapping = mapping

    def keys(self) -> list[str]:
        return list(self._mapping.keys())

    def __getitem__(self, key: str) -> _FakeFeature:
        return self._mapping[key]


class TestHpsTorchEmbedding(unittest.TestCase):
    def test_import_hps_torch_module_falls_back_to_local_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            package_dir = repo_root / "third_party" / "HugeCTR" / "hps_torch" / "hps_torch"
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text(
                "from .lookup_layer import LookupLayer\n",
                encoding="utf-8",
            )
            (package_dir / "lookup_layer.py").write_text(
                "class LookupLayer:\n"
                "    pass\n",
                encoding="utf-8",
            )

            real_import_module = importlib.import_module
            import_calls = 0

            def fake_import_module(name: str):
                nonlocal import_calls
                import_calls += 1
                if import_calls == 1 and name == "hps_torch":
                    raise ModuleNotFoundError(
                        "No module named 'hps_torch'",
                        name="hps_torch",
                    )
                return real_import_module(name)

            sys.modules.pop("hps_torch", None)
            with mock.patch(
                "model_zoo.rs_demo.runtime.hps_torch_embedding.importlib.import_module",
                side_effect=fake_import_module,
            ):
                module = import_hps_torch_module(repo_root=repo_root)

        self.assertEqual(module.__name__, "hps_torch")

    def _make_embedding(self, table_specs=None) -> HpsTorchEmbeddingBagCollection:
        fake_module = types.SimpleNamespace(
            LookupLayer=lambda **kwargs: torch.nn.Identity()
        )
        extra = {"table_specs": table_specs} if table_specs is not None else {}
        with mock.patch(
            "model_zoo.rs_demo.runtime.hps_torch_embedding.import_hps_torch_module",
            return_value=fake_module,
        ):
            return HpsTorchEmbeddingBagCollection(
                [
                    {
                        "name": "t_cat_0",
                        "num_embeddings": 16,
                        "embedding_dim": 4,
                        "feature_names": ["cat_0"],
                    }
                ],
                ps_config_file="/tmp/hps.json",
                model_name="demo",
                **extra,
            )

    def test_forward_rejects_mismatched_sparse_lengths(self) -> None:
        embedding = self._make_embedding(
            table_specs=[
                HpsTableSpec(
                    name="t_cat_0",
                    feature_name="cat_0",
                    num_embeddings=16,
                    embedding_dim=4,
                    key_offset=0,
                    sparse_file="/tmp/table",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "mismatched lengths and values"):
            embedding(
                _FakeKJT(
                    {
                        "cat_0": _FakeFeature(
                            torch.tensor([1, 2], dtype=torch.int64, device="cpu"),
                            torch.tensor([1], dtype=torch.int64, device="cpu"),
                        )
                    }
                )
            )

    def test_forward_rejects_unknown_features(self) -> None:
        embedding = self._make_embedding()

        with self.assertRaisesRegex(KeyError, "unknown feature: cat_1"):
            embedding(
                _FakeKJT(
                    {
                        "cat_1": _FakeFeature(
                            torch.tensor([1], dtype=torch.int64, device="cpu"),
                            torch.tensor([1], dtype=torch.int64, device="cpu"),
                        )
                    }
                )
            )

    def test_runner_builds_torchrun_command_for_hps_torch(self) -> None:
        runner = HpsTorchRunner(Path("/tmp/runtime"))
        cfg = RunConfig(
            backend="hps_torch",
            nnodes=1,
            node_rank=0,
            nproc_per_node=2,
            master_addr="127.0.0.1",
            master_port=29657,
            rdzv_backend="c10d",
            rdzv_id="hps-torch-case",
            output_root="/tmp/rs_demo",
            run_id="hps-torch-case",
            hps_torch_config_file="/tmp/rs_demo/outputs/hps-torch-case/hps_torch.json",
            hps_torch_model_dir="/tmp/rs_demo/outputs/hps-torch-case/hps_torch_model",
            hps_torch_main_csv="/tmp/rs_demo/outputs/hps-torch-case/hps_torch_main.csv",
            hps_torch_main_agg_csv="/tmp/rs_demo/outputs/hps-torch-case/hps_torch_main_agg.csv",
            hps_torch_gpucacheper=0.5,
        )

        cmd = runner._build_torchrun_cmd(Path("/app/RecStore"), cfg)

        self.assertIn("--backend", cmd)
        self.assertIn("hps_torch", cmd)
        self.assertIn("--nproc_per_node", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--hps-torch-gpucacheper", cmd)
        self.assertIn("0.5", cmd)
