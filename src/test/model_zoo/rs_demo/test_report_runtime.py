import unittest
from pathlib import Path
from unittest import mock

from model_zoo.rs_demo.runtime.report import analyze_stage_table


class ReportRuntimeTest(unittest.TestCase):
    def test_analyze_stage_table_passes_requested_table_name(self) -> None:
        repo_root = Path("/repo")
        with mock.patch("model_zoo.rs_demo.runtime.report.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")

            result = analyze_stage_table(
                repo_root=repo_root,
                jsonl_path="/tmp/recstore_events.jsonl",
                csv_path="/tmp/local_shm.csv",
                table_name="local_shm_server_stages",
                top_n=3,
            )

        self.assertEqual(result, "ok")
        cmd = (
            run_mock.call_args.kwargs["args"]
            if "args" in run_mock.call_args.kwargs
            else run_mock.call_args.args[0]
        )
        self.assertIn("--table-name", cmd)
        self.assertEqual("local_shm_server_stages", cmd[cmd.index("--table-name") + 1])


if __name__ == "__main__":
    unittest.main()
