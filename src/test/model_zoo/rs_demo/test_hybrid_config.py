from __future__ import annotations

import unittest
from pathlib import Path

from model_zoo.rs_demo.config import parse_config
from model_zoo.rs_demo.config import populate_default_paths


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


if __name__ == "__main__":
    unittest.main()
