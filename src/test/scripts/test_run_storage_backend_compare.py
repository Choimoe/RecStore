import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "benchmarks"))

from run_storage_backend_compare import (  # noqa: E402
    BACKEND_ALIASES,
    HIERKV_ALIASES,
    build_hierkv_command,
    collect_hierkv_rows,
    merge_rows,
)
from run_hps_backend_compare import command_for  # noqa: E402


class TestRunStorageBackendCompare(unittest.TestCase):
    def test_collect_hierkv_rows_parses_backend_result(self):
        text = (
            "HIERKV_BACKEND_RESULT phase=run backend=hierkv_hbm mode=fetch "
            "distribution=zipfian zipfian_alpha=0.900000 threads=4 "
            "batch_size=128 records=10000 runtime_s=3.000000 batches=100 "
            "key_ops=12800 misses=0 throughput_batches_sec=33.333333 "
            "throughput_keys_sec=4266.666667\n"
        )

        rows = collect_hierkv_rows(
            text,
            alias="hierkv_hbm",
            repeat=2,
            record_count=10000,
            threads=4,
            batch_size=128,
            value_size=512,
            distribution="zipfian",
            zipfian_alpha=0.9,
            log_path=Path("/tmp/hierkv.log"),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["alias"], "hierkv_hbm")
        self.assertEqual(row["backend"], "hierkv")
        self.assertEqual(row["phase"], "run")
        self.assertEqual(row["repeat"], 2)
        self.assertEqual(row["throughput_keys_sec"], "4266.666667")

    def test_build_hierkv_command_contains_storage_parameters(self):
        cmd = build_hierkv_command(
            binary=Path("/tmp/hierkv_backend_benchmark"),
            alias="hierkv_0hbm",
            mode="fetch_insert",
            record_count=10000,
            runtime_seconds=3,
            threads=4,
            batch_size=128,
            value_size=512,
            distribution="uniform",
            zipfian_alpha=0.9,
            max_hbm_for_vectors=0,
        )

        self.assertEqual(cmd[0], "/tmp/hierkv_backend_benchmark")
        self.assertIn("--mode=fetch_insert", cmd)
        self.assertIn("--record_count=10000", cmd)
        self.assertIn("--batch_size=128", cmd)
        self.assertIn("--value_size=512", cmd)
        self.assertIn("--max_hbm_for_vectors=0", cmd)

    def test_default_backend_sets_include_hps_recstore_and_hierkv(self):
        self.assertIn("hps_rocksdb", BACKEND_ALIASES)
        self.assertIn("dram_pet_dram", BACKEND_ALIASES)
        self.assertIn("hierkv_hbm", HIERKV_ALIASES)
        self.assertIn("hierkv_0hbm", HIERKV_ALIASES)

    def test_hps_rocksdb_options_survive_storage_runner_namespace(self):
        class Args:
            mode = "fetch"
            read_ratio = 100
            record_count = 1000000
            runtime_seconds = 5
            threads = 32
            load_threads = 0
            hps_rocksdb_load_threads = 1
            hps_rocksdb_db_threads = 1
            batch_size = 1024
            value_size = 128
            distribution = "uniform"
            zipfian_alpha = 0.9
            dram_allocator = "PERSIST_LOOP_SLAB"
            dram_capacity_bytes = 0
            extra_arg = []

        cmd = command_for(
            "hps_rocksdb",
            BACKEND_ALIASES["hps_rocksdb"],
            Path("/tmp/hps_rocksdb"),
            Args(),
        )

        self.assertIn("--thread_num=32", cmd)
        self.assertIn("--load_thread_num=1", cmd)
        self.assertIn("--hps_rocksdb_thread_num=1", cmd)
        self.assertIn("--ssd_io_backend=IOURING", cmd)
        self.assertIn("--ssd_queue_depth=512", cmd)

    def test_merge_rows_preserves_common_fields(self):
        merged = merge_rows(
            [
                {
                    "alias": "dram_eh_dram",
                    "backend": "recstore",
                    "phase": "run",
                    "throughput_keys_sec": "1000",
                }
            ],
            [
                {
                    "alias": "hierkv_hbm",
                    "backend": "hierkv",
                    "phase": "run",
                    "throughput_keys_sec": "2000",
                }
            ],
        )

        self.assertEqual([row["alias"] for row in merged], ["dram_eh_dram", "hierkv_hbm"])
        self.assertEqual([row["backend"] for row in merged], ["recstore", "hierkv"])


if __name__ == "__main__":
    unittest.main()
