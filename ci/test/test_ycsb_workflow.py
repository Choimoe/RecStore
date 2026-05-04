from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ycsb.yml"


class YcsbWorkflowTest(unittest.TestCase):
    def test_ycsb_container_disables_seccomp_for_iouring(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("--security-opt seccomp=unconfined", workflow)

    def test_ycsb_workflow_runs_batch_engines_as_append_only_workloadc(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("--engines kvdb_batch cceh_batch", workflow)
        self.assertIn("--workloads workloadc", workflow)
        self.assertIn("--append-summary", workflow)

    def test_ycsb_workflow_overrides_hybridkv_capacity_for_ci(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("YCSB_HYBRIDKV_SHM_CAPACITY", workflow)
        self.assertIn("YCSB_HYBRIDKV_SSD_CAPACITY", workflow)
        self.assertIn("kvdb:hybridkv.shmcapacity=${YCSB_HYBRIDKV_SHM_CAPACITY}", workflow)
        self.assertIn("kvdb:hybridkv.ssdcapacity=${YCSB_HYBRIDKV_SSD_CAPACITY}", workflow)
        self.assertIn("kvdb_batch:hybridkv.shmcapacity=${YCSB_HYBRIDKV_SHM_CAPACITY}", workflow)
        self.assertIn("kvdb_batch:hybridkv.ssdcapacity=${YCSB_HYBRIDKV_SSD_CAPACITY}", workflow)

    def test_ycsb_workflow_uses_larger_ci_scale(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('YCSB_RECORD_COUNT: "100000"', workflow)
        self.assertIn('YCSB_OPERATION_COUNT: "100000"', workflow)
        self.assertIn('YCSB_HYBRIDKV_SHM_CAPACITY: "268435456"', workflow)
        self.assertIn('YCSB_HYBRIDKV_SSD_CAPACITY: "1073741824"', workflow)


if __name__ == "__main__":
    unittest.main()
