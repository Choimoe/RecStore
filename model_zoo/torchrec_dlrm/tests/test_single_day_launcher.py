import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_zoo.torchrec_dlrm.launch_single_day import (
    build_torchrun_command,
    detect_dataset_size,
    resolve_run_name,
)
from model_zoo.torchrec_dlrm.launch_config import SingleDayLaunchConfig


class SingleDayLauncherTest(unittest.TestCase):
    def test_resolve_run_name_with_defaults(self) -> None:
        cfg = SingleDayLaunchConfig(
            use_torchrec=True,
            batch_size=256,
            prefetch_depth=4,
            fuse_emb_tables=False,
        )
        run_name = resolve_run_name(cfg)
        self.assertTrue(run_name.startswith("torchrec-bs256-pf4-f0-"))

    def test_detect_dataset_size_from_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = np.zeros((13,), dtype=np.float32)
            np.save(root / "day_0_labels.npy", labels)
            size = detect_dataset_size(str(root))
        self.assertEqual(size, 13)

    def test_build_torchrun_command_for_torchrec(self) -> None:
        cfg = SingleDayLaunchConfig(
            use_torchrec=True,
            processed_dataset_path="./processed_day_0_data",
            batch_size=512,
            learning_rate=0.01,
            epochs=1,
            trace_file="trace.json",
            embedding_storage="uvm",
        )
        cmd = build_torchrun_command(
            python_bin="python3",
            repo_root="/repo/model_zoo/torchrec_dlrm",
            config=cfg,
            run_id="run-123",
        )
        cmd_str = " ".join(cmd)
        self.assertIn("tests/dlrm_main_torchrec_single.py", cmd_str)
        self.assertIn("--embedding_storage uvm", cmd_str)
        self.assertIn("--single_day_mode", cmd_str)
