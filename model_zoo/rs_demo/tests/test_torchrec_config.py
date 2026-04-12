import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_PARENT = str(_THIS_DIR.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rs_demo.config import parse_config  # noqa: E402
from rs_demo.runners.torchrec_runner import validate_torchrec_config  # noqa: E402


class TorchRecConfigTest(unittest.TestCase):
    def test_torchrec_backend_parses_profiler_flags(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "torchrec",
                "--steps",
                "12",
                "--torchrec-profiler",
                "--torchrec-profiler-warmup",
                "1",
                "--torchrec-profiler-active",
                "3",
                "--torchrec-profiler-repeat",
                "2",
                "--torchrec-trace-dir",
                "/tmp/example/trace",
                "--torchrec-main-csv",
                "/tmp/example/main.csv",
                "--torchrec-trace-csv",
                "/tmp/example/trace.csv",
            ]
        )

        self.assertEqual(cfg.backend, "torchrec")
        self.assertTrue(cfg.torchrec_profiler)
        self.assertEqual(cfg.torchrec_profiler_warmup, 1)
        self.assertEqual(cfg.torchrec_profiler_active, 3)
        self.assertEqual(cfg.torchrec_profiler_repeat, 2)
        self.assertEqual(cfg.torchrec_trace_dir, "/tmp/example/trace")
        self.assertEqual(cfg.torchrec_main_csv, "/tmp/example/main.csv")
        self.assertEqual(cfg.torchrec_trace_csv, "/tmp/example/trace.csv")

    def test_torchrec_no_start_server_flag(self) -> None:
        cfg = parse_config(["--backend", "torchrec", "--no-start-server"])
        self.assertEqual(cfg.backend, "torchrec")
        self.assertFalse(cfg.start_server)

    def test_torchrec_profiler_rejected(self) -> None:
        cfg = parse_config(["--backend", "torchrec", "--torchrec-profiler"])
        with self.assertRaises(NotImplementedError):
            validate_torchrec_config(cfg)


if __name__ == "__main__":
    unittest.main()
