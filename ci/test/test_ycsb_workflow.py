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

    def test_ycsb_workflow_runs_embedding_lane_before_publish(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("--engines kvdb_embedding cceh_embedding rocksdb_embedding leveldb_embedding sqlite_embedding", workflow)
        self.assertIn("kvdb_embedding:hybridkv.shmcapacity=${YCSB_HYBRIDKV_SHM_CAPACITY}", workflow)
        self.assertIn("kvdb_embedding:hybridkv.ssdcapacity=${YCSB_HYBRIDKV_SSD_CAPACITY}", workflow)
        self.assertIn("--engine-ycsb-bin \"rocksdb_embedding:/workspace/build_ycsb_external/bin/ycsb\"", workflow)
        self.assertIn("--engine-ycsb-bin \"leveldb_embedding:/workspace/build_ycsb_external/bin/ycsb\"", workflow)
        self.assertIn("--engine-ycsb-bin \"sqlite_embedding:/workspace/build_ycsb_external/bin/ycsb\"", workflow)

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

    def test_ycsb_workflow_uses_separate_external_store_binary(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("-B build_ycsb_recstore", workflow)
        self.assertIn("-B build_ycsb_external", workflow)
        self.assertIn("-DBIND_RECSTORE=OFF", workflow)
        self.assertIn("--engine-ycsb-bin \"rocksdb:/workspace/build_ycsb_external/bin/ycsb\"", workflow)
        self.assertIn("--engine-ycsb-bin \"leveldb:/workspace/build_ycsb_external/bin/ycsb\"", workflow)
        self.assertIn("--engine-ycsb-bin \"sqlite:/workspace/build_ycsb_external/bin/ycsb\"", workflow)


if __name__ == "__main__":
    unittest.main()
