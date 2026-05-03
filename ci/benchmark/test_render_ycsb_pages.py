from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ci" / "benchmark" / "render_ycsb_pages.py"


def write_summary(path: Path) -> None:
    rows = [
        {
            "workload": "workloada",
            "engine": "kvdb",
            "db": "kvdb",
            "repeat": "0",
            "record_count": "1000",
            "operation_count": "1000",
            "threads": "1",
            "phase": "load-run",
            "exit_code": "0",
            "load_runtime_sec": "0.50",
            "load_operations": "1000",
            "load_throughput_ops_sec": "2000.0",
            "run_runtime_sec": "0.25",
            "run_operations": "1000",
            "run_throughput_ops_sec": "4000.0",
            "data_path": "/tmp/data",
            "log_path": "/tmp/log",
            "error_tail": "",
        },
        {
            "workload": "workloadc",
            "engine": "sqlite",
            "db": "sqlite",
            "repeat": "0",
            "record_count": "1000",
            "operation_count": "1000",
            "threads": "1",
            "phase": "load-run",
            "exit_code": "0",
            "load_runtime_sec": "1.0",
            "load_operations": "1000",
            "load_throughput_ops_sec": "1000.0",
            "run_runtime_sec": "0.40",
            "run_operations": "1000",
            "run_throughput_ops_sec": "2500.0",
            "data_path": "/tmp/data2",
            "log_path": "/tmp/log2",
            "error_tail": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RenderYcsbPagesTest(unittest.TestCase):
    def test_merges_history_and_writes_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            summary = tmp_path / "summary.csv"
            output = tmp_path / "site"
            history = tmp_path / "old-history.jsonl"
            write_summary(summary)
            history.write_text(
                json.dumps(
                    {
                        "run": {"run_id": "old", "sha": "1111111", "branch": "main"},
                        "rows": [
                            {
                                "workload": "workloada",
                                "engine": "kvdb",
                                "exit_code": 0,
                                "run_throughput_ops_sec": 3000.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--summary-csv",
                    str(summary),
                    "--output-dir",
                    str(output),
                    "--existing-history",
                    str(history),
                    "--run-id",
                    "12345-1",
                    "--sha",
                    "abcdef123456",
                    "--branch",
                    "main",
                    "--workflow-url",
                    "https://example.invalid/actions/runs/12345",
                ],
                cwd=ROOT,
                check=True,
            )

            self.assertTrue((output / "index.html").exists())
            self.assertTrue(
                (output / "latest" / "summary.csv")
                .read_text(encoding="utf-8")
                .startswith("workload,engine")
            )
            history_lines = (output / "history.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(history_lines), 2)
            latest = json.loads(
                (output / "latest" / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(latest["run"]["sha"], "abcdef123456")
            self.assertEqual(len(latest["rows"]), 2)

            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("RecStore YCSB CI Dashboard", html)
            self.assertIn("Run ID", html)
            self.assertIn("summary.csv", html)
            self.assertIn("Workload Views", html)
            self.assertIn("theme-toggle", html)
            self.assertIn("recstore-ycsb-theme", html)
            self.assertIn("Latest Change", html)
            self.assertIn("run_throughput_ops_sec", html)
            self.assertIn('fetch("latest/run.json"', html)
            self.assertIn('fetch("history.jsonl"', html)
            self.assertNotIn("4000.0", html)
            self.assertNotIn("abcdef123456", html)
            self.assertNotIn("Metric Meaning", html)
            self.assertNotIn("页面内嵌 latest/history JSON", html)


if __name__ == "__main__":
    unittest.main()
