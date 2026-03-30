import tempfile
import unittest
import types
import sys
from argparse import Namespace
from pathlib import Path

from model_zoo.torchrec_dlrm.launch_config import (
    SingleDayLaunchConfig,
    apply_launch_config,
    build_config_from_sources,
)


class LaunchConfigMergeTest(unittest.TestCase):
    def test_gin_file_populates_launch_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "single_day.gin"
            config_path.write_text(
                "\n".join(
                    [
                        "SingleDayLaunchConfig.use_torchrec = True",
                        "SingleDayLaunchConfig.processed_dataset_path = '/tmp/day0'",
                        "SingleDayLaunchConfig.batch_size = 2048",
                        "SingleDayLaunchConfig.learning_rate = 0.01",
                        "SingleDayLaunchConfig.enable_prefetch = False",
                        "SingleDayLaunchConfig.embedding_storage = 'uvm'",
                    ]
                ),
                encoding="utf-8",
            )

            config = build_config_from_sources(
                gin_config=str(config_path),
                gin_bindings=[],
                cli_overrides={},
            )

        self.assertEqual(
            config,
            SingleDayLaunchConfig(
                use_torchrec=True,
                processed_dataset_path="/tmp/day0",
                batch_size=2048,
                learning_rate=0.01,
                epochs=1,
                enable_prefetch=False,
                prefetch_depth=2,
                fuse_emb_tables=True,
                fuse_k=30,
                trace_file="",
                allow_tf32=False,
                embedding_storage="uvm",
            ),
        )

    def test_cli_override_wins_over_gin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "single_day.gin"
            config_path.write_text(
                "\n".join(
                    [
                        "SingleDayLaunchConfig.batch_size = 4096",
                        "SingleDayLaunchConfig.enable_prefetch = True",
                    ]
                ),
                encoding="utf-8",
            )

            config = build_config_from_sources(
                gin_config=str(config_path),
                gin_bindings=[],
                cli_overrides={
                    "batch_size": 1024,
                    "enable_prefetch": False,
                },
            )

        self.assertEqual(config.batch_size, 1024)
        self.assertFalse(config.enable_prefetch)

    def test_multi_config_layer_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base.gin"
            backend = Path(tmpdir) / "backend.gin"
            preset = Path(tmpdir) / "preset.gin"
            base.write_text(
                "\n".join(
                    [
                        "SingleDayLaunchConfig.batch_size = 256",
                        "SingleDayLaunchConfig.use_torchrec = False",
                    ]
                ),
                encoding="utf-8",
            )
            backend.write_text(
                "\n".join(
                    [
                        "SingleDayLaunchConfig.use_torchrec = True",
                        "SingleDayLaunchConfig.embedding_storage = 'uvm'",
                    ]
                ),
                encoding="utf-8",
            )
            preset.write_text(
                "SingleDayLaunchConfig.batch_size = 1024",
                encoding="utf-8",
            )

            config = build_config_from_sources(
                gin_configs=[str(base), str(backend), str(preset)],
                gin_bindings=[],
                cli_overrides={},
            )
        self.assertTrue(config.use_torchrec)
        self.assertEqual(config.embedding_storage, "uvm")
        self.assertEqual(config.batch_size, 1024)


class LaunchConfigApplyTest(unittest.TestCase):
    def test_apply_config_only_sets_missing_values(self) -> None:
        args = Namespace(
            batch_size=32,
            learning_rate=0.02,
            epochs=1,
            in_memory_binary_criteo_path=None,
            enable_prefetch=None,
            prefetch_depth=None,
            fuse_emb_tables=None,
            fuse_k=None,
            trace_file=None,
            allow_tf32=False,
            embedding_storage=None,
        )
        explicit = {"batch_size", "allow_tf32"}
        config = SingleDayLaunchConfig(
            batch_size=2048,
            learning_rate=0.01,
            epochs=3,
            processed_dataset_path="/data/day0",
            enable_prefetch=True,
            prefetch_depth=4,
            fuse_emb_tables=False,
            fuse_k=11,
            trace_file="trace.json",
            allow_tf32=True,
            embedding_storage="ssd",
        )

        apply_launch_config(args, config, explicit)

        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_rate, 0.01)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.in_memory_binary_criteo_path, "/data/day0")
        self.assertTrue(args.enable_prefetch)
        self.assertEqual(args.prefetch_depth, 4)
        self.assertFalse(args.fuse_emb_tables)
        self.assertEqual(args.fuse_k, 11)
        self.assertEqual(args.trace_file, "trace.json")
        self.assertFalse(args.allow_tf32)
        self.assertEqual(args.embedding_storage, "ssd")


class LaunchConfigParserIntegrationTest(unittest.TestCase):
    def test_parser_accepts_gin_flags(self) -> None:
        sys.modules["report_uploader"] = types.SimpleNamespace(
            report_metric=lambda *args, **kwargs: True
        )
        from model_zoo.torchrec_dlrm.tests import dlrm_main_single_day

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "demo.gin"
            config_path.write_text("", encoding="utf-8")

            args = dlrm_main_single_day.parse_args(
                [
                    "--gin_config",
                    str(config_path),
                    "--gin_binding",
                    "SingleDayLaunchConfig.batch_size = 128",
                    "--single_day_mode",
                    "--in_memory_binary_criteo_path",
                    "/tmp/day0",
                ]
            )

            self.assertEqual(args.gin_config, str(config_path))
            self.assertEqual(
                args.gin_bindings,
                ["SingleDayLaunchConfig.batch_size = 128"],
            )
