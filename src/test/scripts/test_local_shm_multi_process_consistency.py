import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_local_shm_mixed_benchmark import (  # noqa: E402
    build_local_shm_server_cmd,
    build_runtime_config,
    resolve_kv_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def build_common_args(
    worker_binary: Path,
    config_path: Path,
    worker_count: int,
    iterations: int,
    rows: int,
    embedding_dim: int,
) -> list[str]:
    return [
        str(worker_binary),
        f"--config_path={config_path}",
        "--table_name=consistency",
        f"--worker_count={worker_count}",
        f"--iterations={iterations}",
        f"--rows={rows}",
        f"--embedding_dim={embedding_dim}",
    ]


class TestLocalShmMultiProcessConsistency(unittest.TestCase):
    def test_concurrent_workers_update_shared_rows_consistently(self):
        server_binary = REPO_ROOT / "build/bin/local_shm_ps_server"
        worker_binary = REPO_ROOT / "build/bin/local_shm_consistency_worker"
        self.assertTrue(server_binary.exists(), server_binary)
        self.assertTrue(worker_binary.exists(), worker_binary)

        worker_count = 4
        rows = 3
        embedding_dim = 4
        iterations = 50

        with tempfile.TemporaryDirectory(prefix="recstore_local_shm_consistency_") as tmpdir:
            runtime_dir = Path(tmpdir)
            config_path = runtime_dir / "local_shm_config.json"
            config = build_runtime_config(
                region_name="recstore_local_ps_consistency",
                slot_count=32,
                ready_queue_count=worker_count,
                ready_queue_burst_limit=8,
                slot_buffer_bytes=1 << 20,
                client_timeout_ms=30000,
                kv_path=str(resolve_kv_path(runtime_dir)),
                capacity=1024,
                value_size=embedding_dim * 4,
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")

            server_log_path = runtime_dir / "local_shm_server.log"
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    build_local_shm_server_cmd(server_binary, config_path),
                    cwd=str(REPO_ROOT),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    time.sleep(0.5)
                    self.assertIsNone(server.poll(), f"server exited: {server_log_path}")

                    common_args = build_common_args(
                        worker_binary,
                        config_path,
                        worker_count,
                        iterations,
                        rows,
                        embedding_dim,
                    )
                    run_checked(common_args + ["--mode=init"], REPO_ROOT)

                    workers = []
                    for worker_id in range(worker_count):
                        env = dict(os.environ)
                        env["RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX"] = str(worker_id)
                        workers.append(
                            (
                                worker_id,
                                subprocess.Popen(
                                    common_args
                                    + [
                                        "--mode=update",
                                        f"--worker_id={worker_id}",
                                    ],
                                    cwd=str(REPO_ROOT),
                                    env=env,
                                    text=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                ),
                            )
                        )

                    for worker_id, process in workers:
                        stdout, stderr = process.communicate(timeout=20)
                        self.assertEqual(
                            process.returncode,
                            0,
                            f"worker {worker_id} failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
                        )

                    run_checked(common_args + ["--mode=verify"], REPO_ROOT)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()

    def test_concurrent_readers_never_observe_torn_uniform_rows(self):
        server_binary = REPO_ROOT / "build/bin/local_shm_ps_server"
        worker_binary = REPO_ROOT / "build/bin/local_shm_consistency_worker"
        self.assertTrue(server_binary.exists(), server_binary)
        self.assertTrue(worker_binary.exists(), worker_binary)

        writer_count = 4
        reader_count = 4
        rows = 8
        embedding_dim = 32
        iterations = 200
        read_iterations = 1500
        ready_queue_count = writer_count + reader_count

        with tempfile.TemporaryDirectory(prefix="recstore_local_shm_read_stress_") as tmpdir:
            runtime_dir = Path(tmpdir)
            config_path = runtime_dir / "local_shm_config.json"
            config = build_runtime_config(
                region_name="recstore_local_ps_read_stress",
                slot_count=64,
                ready_queue_count=ready_queue_count,
                ready_queue_burst_limit=8,
                slot_buffer_bytes=1 << 20,
                client_timeout_ms=30000,
                kv_path=str(resolve_kv_path(runtime_dir)),
                capacity=4096,
                value_size=embedding_dim * 4,
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")

            server_log_path = runtime_dir / "local_shm_server.log"
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    build_local_shm_server_cmd(server_binary, config_path),
                    cwd=str(REPO_ROOT),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    time.sleep(0.5)
                    self.assertIsNone(server.poll(), f"server exited: {server_log_path}")

                    common_args = build_common_args(
                        worker_binary,
                        config_path,
                        writer_count,
                        iterations,
                        rows,
                        embedding_dim,
                    )
                    run_checked(common_args + ["--mode=init_uniform"], REPO_ROOT)

                    processes = []
                    for writer_id in range(writer_count):
                        env = dict(os.environ)
                        env["RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX"] = str(writer_id)
                        processes.append(
                            (
                                f"writer{writer_id}",
                                subprocess.Popen(
                                    common_args
                                    + [
                                        "--mode=update_uniform",
                                        f"--worker_id={writer_id}",
                                    ],
                                    cwd=str(REPO_ROOT),
                                    env=env,
                                    text=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                ),
                            )
                        )
                    for reader_id in range(reader_count):
                        env = dict(os.environ)
                        env["RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX"] = str(
                            writer_count + reader_id
                        )
                        processes.append(
                            (
                                f"reader{reader_id}",
                                subprocess.Popen(
                                    common_args
                                    + [
                                        "--mode=read_stress",
                                        f"--read_iterations={read_iterations}",
                                    ],
                                    cwd=str(REPO_ROOT),
                                    env=env,
                                    text=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                ),
                            )
                        )

                    for label, process in processes:
                        stdout, stderr = process.communicate(timeout=30)
                        self.assertEqual(
                            process.returncode,
                            0,
                            f"{label} failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
                        )
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()


if __name__ == "__main__":
    unittest.main()
