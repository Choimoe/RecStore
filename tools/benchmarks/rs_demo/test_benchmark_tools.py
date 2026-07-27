from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo import config
from model_zoo.rs_demo.config import RunConfig


class TestBenchmarkTools(unittest.TestCase):
    def test_shared_ps_summary_parser_keeps_measure_rows_only(self) -> None:
        from tools.benchmarks.summary import collect_ps_transport_summary_rows

        sample = (
            "transport=RDMA op=put phase=warmup summary rounds=1 iterations=2 "
            "batch_keys=4 elapsed_us_mean=232 elapsed_us_p50=232 "
            "elapsed_us_p95=232 elapsed_us_p99=232 ops_per_sec=8620.69 "
            "key_ops_per_sec=34482.8\n"
            "transport=RDMA op=get phase=measure summary rounds=1 iterations=2 "
            "batch_keys=4 elapsed_us_mean=13 elapsed_us_p50=13 "
            "elapsed_us_p95=13 elapsed_us_p99=13 ops_per_sec=153846 "
            "key_ops_per_sec=615385\n"
        )

        rows = collect_ps_transport_summary_rows(sample)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "RDMA")
        self.assertEqual(rows[0]["op"], "get")
        self.assertEqual(rows[0]["phase"], "measure")

    def test_shared_mixed_summary_parser_keeps_all_phases(self) -> None:
        from tools.benchmarks.summary import collect_mixed_summary_rows

        sample = (
            "system=RecStore transport=LOCAL_SHM phase=init summary rounds=1 "
            "iterations=1 batch_keys=0 num_embeddings=64 elapsed_us_mean=300 "
            "elapsed_us_p50=300 elapsed_us_p95=300 elapsed_us_p99=300 "
            "ops_per_sec=50 key_ops_per_sec=50\n"
            "system=RecStore transport=LOCAL_SHM phase=measure summary rounds=1 "
            "iterations=5 batch_keys=8 num_embeddings=64 elapsed_us_mean=100 "
            "elapsed_us_p50=90 elapsed_us_p95=120 elapsed_us_p99=130 "
            "ops_per_sec=200 key_ops_per_sec=1600\n"
        )

        rows = collect_mixed_summary_rows(sample)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["phase"], "init")
        self.assertEqual(rows[1]["phase"], "measure")
        self.assertEqual(rows[1]["num_embeddings"], 64)

    def test_benchmark_runner_modules_are_exposed_under_tools_package(self) -> None:
        from tools.benchmarks import (
            run_benchmark_ps,
            run_hierkv_recstore_mixed_benchmark,
            run_local_shm_mixed_benchmark,
            run_local_shm_multi_process_benchmark,
            run_ps_dram_transport_benchmark,
            run_rdma_rc_transport_benchmark,
            run_rdma_transport_benchmarks,
        )

        self.assertTrue(callable(run_benchmark_ps.parse_args))
        self.assertTrue(callable(run_ps_dram_transport_benchmark.build_runtime_config))
        self.assertTrue(callable(run_rdma_transport_benchmarks.build_benchmark_cmd))
        self.assertTrue(callable(run_rdma_rc_transport_benchmark.build_benchmark_cmd))
        self.assertTrue(callable(run_local_shm_mixed_benchmark.build_runtime_config))
        self.assertTrue(callable(run_local_shm_multi_process_benchmark.build_worker_env))
        self.assertTrue(callable(run_hierkv_recstore_mixed_benchmark.build_recstore_cmd))

    def test_validate_recstore_config_keeps_default_lane_behavior_when_fast_path_disabled(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=2,
            nproc_per_node=1,
            node_rank=1,
            recstore_runtime_dir="/tmp/shared-runtime",
        )

        config.validate_recstore_config(cfg)

    def test_validate_recstore_config_rejects_invalid_fast_path_owner_policy(self) -> None:
        cfg = RunConfig(
            backend="recstore",
            nnodes=1,
            nproc_per_node=2,
            single_node_owner_policy="rank_zero",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node path only supports --single-node-owner-policy=hash_mod_world_size",
        ):
            config.validate_recstore_config(cfg)

    def test_build_suites_contains_expected_lanes(self) -> None:
        from tools.benchmarks.lanes import build_suites

        suites = build_suites()
        self.assertIn("main_results", suites)
        self.assertIn("stage_breakdown", suites)
        self.assertIn("recstore_chain", suites)

        stage_lane_names = [lane.name for lane in suites["stage_breakdown"].lanes]
        self.assertIn("RecStore-本地BRPC", stage_lane_names)
        self.assertIn("TorchRec-单机单卡", stage_lane_names)
        self.assertIn("TorchRec-双机双进程", stage_lane_names)

    def test_aggregate_repeat_metrics(self) -> None:
        from tools.benchmarks.aggregate import aggregate_metric_rows

        rows = [
            {"lane": "TorchRec-单机单卡", "metric": "step_total_ms", "value": 100.0},
            {"lane": "TorchRec-单机单卡", "metric": "step_total_ms", "value": 120.0},
            {"lane": "TorchRec-单机单卡", "metric": "step_total_ms", "value": 140.0},
        ]

        aggregated = aggregate_metric_rows(rows)
        self.assertEqual(len(aggregated), 1)
        row = aggregated[0]
        self.assertEqual(row["lane"], "TorchRec-单机单卡")
        self.assertEqual(row["metric"], "step_total_ms")
        self.assertEqual(row["count"], 3)
        self.assertEqual(row["mean"], 120.0)
        self.assertEqual(row["min"], 100.0)
        self.assertEqual(row["max"], 140.0)

    def test_main_results_lane_metrics_are_unique(self) -> None:
        from tools.benchmarks.lanes import build_suites

        suites = build_suites()
        for lane in suites["main_results"].lanes:
            metric_names = [metric.name for metric in lane.metrics]
            self.assertEqual(
                len(metric_names),
                len(set(metric_names)),
                msg=f"duplicate metrics found in lane {lane.name}: {metric_names}",
            )

    def test_write_summary_csv(self) -> None:
        from tools.benchmarks.aggregate import write_summary_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.csv"
            write_summary_csv(
                out_path,
                [
                    {
                        "lane": "RecStore-本地BRPC",
                        "metric": "step_total_ms",
                        "count": 2,
                        "mean": 10.0,
                        "std": 1.0,
                        "p50": 10.0,
                        "p95": 11.5,
                        "min": 9.0,
                        "max": 11.0,
                    }
                ],
            )

            content = out_path.read_text(encoding="utf-8")

        self.assertIn("RecStore-本地BRPC", content)
        self.assertIn("step_total_ms", content)


if __name__ == "__main__":
    unittest.main()
