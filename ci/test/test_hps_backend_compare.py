import subprocess
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path

from tools.benchmarks import run_hps_backend_compare


class HpsBackendCompareTest(unittest.TestCase):
    def test_ensure_build_passes_parallelism_after_build_tool_separator(self):
        calls = []

        def fake_run(cmd, cwd, check):
            calls.append((cmd, cwd, check))
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(subprocess, "run", fake_run):
            run_hps_backend_compare.ensure_build(build_jobs=7)

        self.assertEqual(
            calls,
            [
                (
                    [
                        "cmake",
                        "--build",
                        "build",
                        "--target",
                        "hps_backend_benchmark",
                        "--",
                        "-j7",
                    ],
                    run_hps_backend_compare.ROOT,
                    True,
                )
            ],
        )

    def test_hps_rocksdb_uses_dedicated_serial_load_threads_by_default(self):
        args = Namespace(
            mode="fetch",
            read_ratio=100,
            record_count=1000000,
            runtime_seconds=5,
            threads=32,
            load_threads=0,
            hps_rocksdb_load_threads=1,
            hps_rocksdb_db_threads=1,
            batch_size=1024,
            value_size=512,
            distribution="uniform",
            zipfian_alpha=0.9,
            dram_allocator="PERSIST_LOOP_SLAB",
            dram_capacity_bytes=0,
            extra_arg=[],
        )

        cmd = run_hps_backend_compare.command_for(
            "hps_rocksdb",
            run_hps_backend_compare.BACKEND_ALIASES["hps_rocksdb"],
            Path("/tmp/hps_rocksdb"),
            args,
        )

        self.assertIn("--thread_num=32", cmd)
        self.assertIn("--load_thread_num=1", cmd)
        self.assertIn("--hps_rocksdb_thread_num=1", cmd)


if __name__ == "__main__":
    unittest.main()
