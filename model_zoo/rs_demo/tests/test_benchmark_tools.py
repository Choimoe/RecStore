from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo import config
from model_zoo.rs_demo.config import RunConfig


class TestBenchmarkTools(unittest.TestCase):
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
            enable_single_node_distributed_fast_path=True,
            single_node_owner_policy="rank_zero",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "single-node distributed fast path only supports --single-node-owner-policy=hash_mod_world_size",
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
