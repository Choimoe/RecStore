import importlib.util
import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_PARENT = str(_THIS_DIR.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rs_demo.cli import build_runner  # noqa: E402
from rs_demo.config import RunConfig  # noqa: E402


class TorchRecDispatchTest(unittest.TestCase):
    def test_build_runner_torchrec(self) -> None:
        cfg = RunConfig(backend="torchrec")
        original_find_spec = importlib.util.find_spec
        try:
            importlib.util.find_spec = lambda name: None if name == "torchrec" else original_find_spec(name)
            with self.assertRaises(RuntimeError) as ctx:
                build_runner(cfg, Path("/tmp"))
        finally:
            importlib.util.find_spec = original_find_spec
        self.assertIn("torchrec dependency is not installed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
