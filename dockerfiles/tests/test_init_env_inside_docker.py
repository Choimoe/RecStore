import os
import pathlib
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "dockerfiles" / "init_env_inside_docker.sh"


class InitEnvInsideDockerCacheReuseTest(unittest.TestCase):
    def _probe_step(self, step_name, existing_paths=()):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            project_root = workspace / "project"
            target_root = workspace / "target"
            project_root.mkdir()
            target_root.mkdir()

            for relpath in existing_paths:
                full_path = project_root / relpath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_PATH_OVERRIDE": str(project_root),
                    "TARGET_DIR_OVERRIDE": str(target_root),
                    "LIST_STEP_ACTION_ONLY": step_name,
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip().splitlines()

    def test_libtorch_abi_step_reports_cached_when_archive_and_extract_exist(self):
        output = self._probe_step(
            "step_libtorch_abi",
            existing_paths=(
                "third_party/libtorch/libtorch.zip",
                "third_party/libtorch/libtorch/.keep",
            ),
        )
        self.assertEqual(output, ["CACHED"])

    def test_libtorch_abi_step_runs_without_cached_outputs(self):
        output = self._probe_step("step_libtorch_abi")
        self.assertEqual(output, ["RUN"])

    def test_grpc_step_reports_cached_when_install_tree_exists(self):
        output = self._probe_step(
            "step_GRPC",
            existing_paths=("third_party/grpc-install/lib/libgrpc++.so",),
        )
        self.assertEqual(output, ["CACHED"])

    def test_grpc_step_runs_without_install_tree(self):
        output = self._probe_step("step_GRPC")
        self.assertEqual(output, ["RUN"])

    def test_brpc_step_reports_cached_when_install_tree_exists(self):
        output = self._probe_step(
            "step_brpc",
            existing_paths=("third_party/brpc-install/lib/libbrpc.so",),
        )
        self.assertEqual(output, ["CACHED"])

    def test_brpc_step_runs_without_install_tree(self):
        output = self._probe_step("step_brpc")
        self.assertEqual(output, ["RUN"])


if __name__ == "__main__":
    unittest.main()
