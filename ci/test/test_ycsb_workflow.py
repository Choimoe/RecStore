from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ycsb.yml"


class YcsbWorkflowTest(unittest.TestCase):
    def test_ycsb_container_disables_seccomp_for_iouring(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("--security-opt seccomp=unconfined", workflow)


if __name__ == "__main__":
    unittest.main()
