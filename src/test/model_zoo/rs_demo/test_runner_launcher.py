from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.config import RunConfig
from model_zoo.rs_demo.runners.hps_torch_runner import HpsTorchRunner
from model_zoo.rs_demo.runners.recstore_runner import RecStoreRunner
from model_zoo.rs_demo.runners.torchrec_runner import TorchRecRunner
from model_zoo.rs_demo.runtime.worker_common import write_rows


def _cfg(tmpdir: str, **overrides) -> RunConfig:
    values = dict(
        backend="recstore",
        output_root=tmpdir,
        run_id="launcher",
        nnodes=1,
        node_rank=0,
        nproc_per_node=2,
    )
    values.update(overrides)
    return RunConfig(**values)


class TestSharedLauncher(unittest.TestCase):
    def test_json_backends_share_the_launcher_prefix(self) -> None:
        repo_root = Path("/app/RecStore")
        config_json = Path("/tmp/worker_config.json")
        for runner in (RecStoreRunner(Path("/tmp/runtime")), TorchRecRunner(Path("/tmp/runtime"))):
            cmd = runner._build_torchrun_cmd(repo_root, _cfg("/tmp/x"), config_json)
            self.assertEqual(cmd[:3], [sys.executable, "-m", "torch.distributed.run"])
            self.assertEqual(cmd[-2:], ["--run-config-json", str(config_json)])
            self.assertIn(str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"), cmd)

    def test_hps_uses_argv_instead_of_config_json(self) -> None:
        cmd = HpsTorchRunner(Path("/tmp/runtime"))._build_torchrun_cmd(
            Path("/app/RecStore"), _cfg("/tmp/x", backend="hps_torch")
        )
        self.assertIn("--backend", cmd)
        self.assertIn("hps_torch", cmd)
        self.assertNotIn("--run-config-json", cmd)

    def test_multi_host_run_skips_merge_when_remote_ranks_are_invisible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _cfg(tmpdir, nnodes=2, node_rank=0, nproc_per_node=1)
            runner = RecStoreRunner(Path("/tmp/runtime"))
            rank_dir = runner._rank_output_dir(cfg)
            rank_dir.mkdir(parents=True)
            write_rows(rank_dir / "rank0.csv", [{"backend": "recstore", "rank": 0, "step": 0}])

            self.assertEqual(runner._merge_rank_csvs(cfg, rank_dir), [])

    def test_missing_local_rank_csv_still_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _cfg(tmpdir, nnodes=2, node_rank=0, nproc_per_node=1)
            runner = RecStoreRunner(Path("/tmp/runtime"))
            rank_dir = runner._rank_output_dir(cfg)
            rank_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "missing local rank csv outputs"):
                runner._merge_rank_csvs(cfg, rank_dir)


if __name__ == "__main__":
    unittest.main()
