import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ctest_annotations


class CtestAnnotationsTest(unittest.TestCase):
    def test_skipped_tests_do_not_emit_failure_annotations(self) -> None:
        log = """\
===== ctest lane: rdma_integration =====
    Start 1: skipped_test
1/2 Test #1: skipped_test ....***Skipped   0.04 sec
    Start 2: failed_test
2/2 Test #2: failed_test .....***Failed    0.05 sec
AssertionError: expected true
"""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as handle:
            handle.write(log)
            handle.flush()
            output = io.StringIO()
            with mock.patch.object(sys, "argv", ["ctest_annotations.py", handle.name]):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(ctest_annotations.main(), 0)

        rendered = output.getvalue()
        self.assertIn("ctest: 1 test(s) failed or did not run", rendered)
        self.assertIn("failed_test FAILED", rendered)
        self.assertNotIn("skipped_test FAILED", rendered)


if __name__ == "__main__":
    unittest.main()
