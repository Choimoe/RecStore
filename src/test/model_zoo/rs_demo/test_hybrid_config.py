from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from model_zoo.rs_demo.config import (
    dump_run_config, parse_config, populate_default_paths, validate_recstore_config,
    validate_torchrec_config,
)


class TestHybridConfig(unittest.TestCase):
    def test_parse_config_echoes_flags_to_attributes(self) -> None:
        cases = [
            (
                [
                    "--embedding-dim",
                    "128",
                    "--dense-arch-layer-sizes",
                    "512,256,128",
                    "--over-arch-layer-sizes",
                    "1024,1024,512,256,1",
                ],
                {
                    "embedding_dim": 128,
                    "dense_arch_layer_sizes": "512,256,128",
                    "over_arch_layer_sizes": "1024,1024,512,256,1",
                },
            ),
            (
                ["--backend", "recstore", "--ps-kv-backend", "hps_rocksdb"],
                {"ps_kv_backend": "hps_rocksdb"},
            ),
            (
                [
                    "--backend",
                    "recstore",
                    "--ps-kv-backend",
                    "recstore_tiered",
                    "--tiered-dram-capacity-multiplier",
                    "0.02",
                ],
                {"tiered_dram_capacity_multiplier": 0.02},
            ),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                cfg = parse_config(args)
                for attr, value in expected.items():
                    self.assertEqual(getattr(cfg, attr), value)

    def test_populate_default_paths_makes_relative_output_root_absolute(self) -> None:
        cfg = parse_config(
            [
                "--output-root",
                "relative-output",
                "--run-id",
                "case-relative-output",
            ]
        )

        populate_default_paths(cfg)

        self.assertTrue(Path(cfg.output_root).is_absolute())
        self.assertTrue(Path(cfg.recstore_main_csv).is_absolute())
        self.assertIn("relative-output", cfg.recstore_main_csv)

    def test_planned_pipeline_flags_survive_worker_config_round_trip(self) -> None:
        for graphs, collective in ((False, "all_reduce"), (True, "all_gather")):
            with self.subTest(graphs=graphs, collective=collective):
                args = [
                    "--backend", "recstore", "--optimization-plugin", "bagpipe_planned",
                    "--optimization-prescan", "--optimization-lookahead", "4",
                    "--optimization-sparse-collective", collective,
                ]
                if graphs:
                    args.append("--optimization-cuda-graphs")
                config = parse_config(args)
                validate_recstore_config(config)
                with tempfile.TemporaryDirectory() as directory:
                    path = dump_run_config(config, Path(directory) / "worker.json")
                    restored = parse_config(["--run-config-json", str(path)])
                for value in (config, restored):
                    self.assertEqual(value.optimization.cuda_graphs, graphs)
                    self.assertEqual(value.optimization.sparse_collective, collective)
                    validate_recstore_config(value)

    def test_planned_pipeline_flags_reject_other_plugins(self) -> None:
        for args in (["--optimization-cuda-graphs"], ["--optimization-sparse-collective", "all_gather"]):
            with self.subTest(args=args):
                config = parse_config([
                    "--backend", "recstore", "--optimization-plugin", "bagpipe",
                    "--optimization-lookahead", "4", *args,
                ])
                with self.assertRaisesRegex(RuntimeError, "requires.*bagpipe_planned"):
                    validate_recstore_config(config)

    def test_torchrec_dense_graphs_do_not_require_a_recstore_plugin(self) -> None:
        config = parse_config(["--backend", "torchrec", "--optimization-cuda-graphs"])
        self.assertEqual(config.optimization.plugin, "none")
        validate_torchrec_config(config)
        with tempfile.TemporaryDirectory() as directory:
            path = dump_run_config(config, Path(directory) / "worker.json")
            restored = parse_config(["--run-config-json", str(path)])
        self.assertTrue(restored.optimization.cuda_graphs)
        validate_torchrec_config(restored)
        restored.model = "unsupported"
        with self.assertRaisesRegex(RuntimeError, "TorchRec DLRM"):
            validate_torchrec_config(restored)
        restored.model = "dlrm"
        restored.torchrec_dist_mode = "fair_remote"
        restored.nnodes = 2
        with self.assertRaisesRegex(RuntimeError, "replicated TorchRec"):
            validate_torchrec_config(restored)


if __name__ == "__main__":
    unittest.main()
