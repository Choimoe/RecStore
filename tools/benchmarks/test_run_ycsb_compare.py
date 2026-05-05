from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "benchmarks" / "run_ycsb_compare.py"


def run_compare(
    output_dir: Path,
    ycsb_bin: Path,
    *,
    keep_data: bool = False,
    engines: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--build-dir",
        str(ycsb_bin.parent.parent),
        "--ycsb-bin",
        str(ycsb_bin),
        "--engines",
        *(engines or ["basic"]),
        "--workloads",
        "workloada",
        "--record-count",
        "1",
        "--operation-count",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    if keep_data:
        cmd.append("--keep-data")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_batch_compare(output_dir: Path, ycsb_bin: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--build-dir",
        str(ycsb_bin.parent.parent),
        "--ycsb-bin",
        str(ycsb_bin),
        "--engines",
        "kvdb_batch",
        "cceh_batch",
        "--workloads",
        "workloadc",
        "--record-count",
        "1",
        "--operation-count",
        "1",
        "--batch-size",
        "4",
        "--output-dir",
        str(output_dir),
    ]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_batch_compare_append(output_dir: Path, ycsb_bin: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--build-dir",
        str(ycsb_bin.parent.parent),
        "--ycsb-bin",
        str(ycsb_bin),
        "--engines",
        "kvdb_batch",
        "--workloads",
        "workloadc",
        "--record-count",
        "1",
        "--operation-count",
        "1",
        "--batch-size",
        "4",
        "--output-dir",
        str(output_dir),
        "--append-summary",
    ]
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

    def test_batch_engines_enable_ycsb_batch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_batch_compare(root / "out", ycsb_bin)

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            kvdb_log = (root / "out" / "logs" / "workloadc_kvdb_batch_r0.log").read_text(
                encoding="utf-8"
            )
            cceh_log = (root / "out" / "logs" / "workloadc_cceh_batch_r0.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("-p ycsb.batch=true", kvdb_log)
            self.assertIn("-p ycsb.batch_size=4", kvdb_log)
            self.assertIn("-p ycsb.batch=true", cceh_log)
            self.assertIn("-p ycsb.batch_size=4", cceh_log)

    def test_kvdb_uses_correct_ycsb_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(root / "out", ycsb_bin, engines=["kvdb"])

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            log = (root / "out" / "logs" / "workloada_kvdb_r0.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("-p hybridkv.mode=compat", log)
            self.assertIn("-p hybridkv.read_return=parse", log)
            self.assertNotIn("-p hybridkv.mode=perf", log)
            self.assertNotIn("-p hybridkv.read_return=none", log)

    def test_embedding_engines_enable_embedding_lane_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(
                root / "out",
                ycsb_bin,
                engines=["kvdb_embedding", "cceh_embedding"],
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            kvdb_log = (root / "out" / "logs" / "workloada_kvdb_embedding_r0.log").read_text(
                encoding="utf-8"
            )
            cceh_log = (root / "out" / "logs" / "workloada_cceh_embedding_r0.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("-p hybridkv.mode=embedding", kvdb_log)
            self.assertIn("-p cceh.mode=embedding", cceh_log)
            self.assertIn("-p fieldcount=1", kvdb_log)
            self.assertIn("-p fieldlength=128", cceh_log)

    def test_external_embedding_engines_use_blob_shape_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(
                root / "out",
                ycsb_bin,
                engines=["rocksdb_embedding", "leveldb_embedding", "sqlite_embedding"],
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            for engine in ("rocksdb_embedding", "leveldb_embedding", "sqlite_embedding"):
                log = (root / "out" / "logs" / f"workloada_{engine}_r0.log").read_text(
                    encoding="utf-8"
                )
                self.assertIn("-p fieldcount=1", log)
                self.assertIn("-p fieldlength=128", log)
                self.assertIn("-p readallfields=true", log)
                self.assertIn("-p writeallfields=true", log)

    def test_sqlite_embedding_uses_database_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            completed = run_compare(
                root / "out",
                ycsb_bin,
                engines=["sqlite_embedding"],
                keep_data=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            log = (root / "out" / "logs" / "workloada_sqlite_embedding_r0.log").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                f"-p sqlite.dbpath={root / 'out' / 'data' / 'workloada_sqlite_embedding_r0' / 'ycsb.sqlite3'}",
                log,
            )

    def test_append_summary_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            ycsb_bin.chmod(0o755)

            first = run_compare(root / "out", ycsb_bin)
            second = run_batch_compare_append(root / "out", ycsb_bin)

            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            summary = (root / "out" / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("workloada,basic,basic,0,1,1,1,load-run,0", summary)
            self.assertIn("workloadc,kvdb_batch,kvdb,0,1,1,1,load-run,0", summary)

    def test_runner_disables_gperftools_sampling_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ycsb_bin = root / "build" / "bin" / "ycsb"
            ycsb_bin.parent.mkdir(parents=True)
            ycsb_bin.write_text(
                "#!/bin/sh\n"
                "test \"${CPUPROFILE_FREQUENCY:-}\" = 0 || exit 23\n"
                "printf 'Run operations(ops): 1\\n'\n",
                encoding="utf-8",
            )
            ycsb_bin.chmod(0o755)

            completed = run_compare(root / "out", ycsb_bin, engines=["rocksdb"])

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_engine_specific_ycsb_binary_overrides_default_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_bin = root / "default" / "bin" / "ycsb"
            rocksdb_bin = root / "rocksdb" / "bin" / "ycsb"
            default_bin.parent.mkdir(parents=True)
            rocksdb_bin.parent.mkdir(parents=True)
            default_bin.write_text("#!/bin/sh\nexit 31\n", encoding="utf-8")
            rocksdb_bin.write_text("#!/bin/sh\nprintf 'Run operations(ops): 1\\n'\n", encoding="utf-8")
            default_bin.chmod(0o755)
            rocksdb_bin.chmod(0o755)

            completed = run_compare(
                root / "out",
                default_bin,
                engines=["rocksdb"],
                extra_args=[f"--engine-ycsb-bin=rocksdb:{rocksdb_bin}"],
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)


if __name__ == "__main__":
    unittest.main()
