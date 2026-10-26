import tempfile
import unittest
from pathlib import Path

from tools.benchmarks.run_ssd_layout_microbench import parse_layout_result


class TestSsdLayoutMicrobench(unittest.TestCase):
    def test_parses_page_io_and_bandwidth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "plain.log"
            log.write_text(
                "noise\n"
                "LAYOUT_RESULT queries=2 rows=8 seconds=0.500000 "
                "layout_strategy=plain layout_page_reads=4 "
                "layout_bytes_read=16384\n"
            )
            result = parse_layout_result(log)
        self.assertEqual(result["page_io_count"], 4)
        self.assertEqual(result["bytes_read"], 16384)
        self.assertEqual(result["bandwidth_bytes_sec"], 32768.0)

    def test_rejects_logs_without_layout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "broken.log"
            log.write_text("LAYOUT_RESULT queries=2 rows=8 seconds=0.5\n")
            with self.assertRaisesRegex(ValueError, "missing layout metric"):
                parse_layout_result(log)


if __name__ == "__main__":
    unittest.main()
