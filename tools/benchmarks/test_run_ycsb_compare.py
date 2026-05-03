from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "benchmarks" / "run_ycsb_compare.py"


def run_compare(output_dir: Path, ycsb_bin: Path, *, keep_data: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--build-dir",
        str(ycsb_bin.parent.parent),
        "--ycsb-bin",
        str(ycsb_bin),
        "--engines",
        "basic",
        "--workloads",
        "workloada",
        "--record-count",
        "1",
        "--operation-count",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    if keep_data:
        cmd.append("--keep-data")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class RunYcsbCompareTest(unittest.TestCase):
    def test_default_removes_per_case_data_after_successful_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Load operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(root / "out", ycsb_bin)

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("data_size=", completed.stdout)
            self.assertTrue((root / "out" / "summary.csv").exists())
            self.assertTrue((root / "out" / "logs" / "workloada_basic_r0.log").exists())
            self.assertFalse((root / "out" / "data" / "workloada_basic_r0").exists())

    def test_keep_data_preserves_per_case_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(root / "out", ycsb_bin, keep_data=True)

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue((root / "out" / "data" / "workloada_basic_r0").exists())


if __name__ == "__main__":
    unittest.main()
