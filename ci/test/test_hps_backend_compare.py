import subprocess
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
