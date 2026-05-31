#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from run_local_shm_mixed_benchmark import (
    build_benchmark_cmd,
    build_local_shm_server_cmd,
    build_runtime_config,
    collect_summary_rows,
    resolve_kv_path,
)


def build_worker_env(
    worker_rank: int, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX"] = str(worker_rank)
    return env


def build_worker_label(worker_rank: int) -> str:
    return f"worker{worker_rank}"


def resolve_ready_queue_index(worker_rank: int, ready_queue_count: int) -> int:
    if ready_queue_count <= 0:
        raise ValueError("ready_queue_count must be positive")
    return worker_rank % ready_queue_count


def run_cmd(
    cmd: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/app/RecStore")
    parser.add_argument(
        "--benchmark-binary",
        default="/app/RecStore/build/bin/recstore_mixed_benchmark",
    )
    parser.add_argument(
        "--server-binary",
        default="/app/RecStore/build/bin/local_shm_ps_server",
    )
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument(
        "--ready-queue-count",
        type=int,
        default=0,
        help="0 means use worker_count",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--batch-keys", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--slot-count", type=int, default=64)
    parser.add_argument("--slot-buffer-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--ready-queue-burst-limit", type=int, default=8)
    parser.add_argument("--client-timeout-ms", type=int, default=30000)
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--startup-delay", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_binary = Path(args.benchmark_binary).resolve()
    server_binary = Path(args.server_binary).resolve()
    ready_queue_count = (
        args.worker_count if args.ready_queue_count == 0 else args.ready_queue_count
    )

    with tempfile.TemporaryDirectory(prefix="recstore_local_shm_mp_bench_") as tmpdir:
        runtime_dir = Path(tmpdir)
        config_path = runtime_dir / "local_shm_config.json"
        config = build_runtime_config(
            region_name="recstore_local_ps",
            slot_count=args.slot_count,
            ready_queue_count=ready_queue_count,
            ready_queue_burst_limit=args.ready_queue_burst_limit,
            slot_buffer_bytes=args.slot_buffer_bytes,
            client_timeout_ms=args.client_timeout_ms,
            kv_path=str(resolve_kv_path(runtime_dir)),
            capacity=args.capacity,
            value_size=args.embedding_dim * 4,
        )
        config_path.write_text(json.dumps(config), encoding="utf-8")

        server = subprocess.Popen(
            build_local_shm_server_cmd(server_binary, config_path),
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            import time

            time.sleep(args.startup_delay)
            preload_cmd = build_benchmark_cmd(
                benchmark_binary=benchmark_binary,
                iterations=1,
                rounds=0,
                warmup_rounds=0,
                batch_keys=args.batch_keys,
                embedding_dim=args.embedding_dim,
                num_embeddings=args.num_embeddings,
                report_mode="summary",
                update_scale=0.001,
                table_name="default",
            )
            preload_cmd.append(f"--host={config_path}")
            run_checked = subprocess.run(
                preload_cmd,
                cwd=str(repo_root),
                env=build_worker_env(0),
                text=True,
                capture_output=True,
                check=False,
            )
            if run_checked.returncode != 0:
                raise RuntimeError(
                    "preload failed\n"
                    f"stdout:\n{run_checked.stdout}\n"
                    f"stderr:\n{run_checked.stderr}"
                )

            workers: list[tuple[str, subprocess.Popen[str]]] = []
            for worker_rank in range(args.worker_count):
                label = build_worker_label(worker_rank)
                cmd = build_benchmark_cmd(
                    benchmark_binary=benchmark_binary,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    warmup_rounds=args.warmup_rounds,
                    batch_keys=args.batch_keys,
                    embedding_dim=args.embedding_dim,
                    num_embeddings=args.num_embeddings,
                    report_mode="summary",
                    update_scale=0.001,
                    table_name="default",
                )
                cmd.append(f"--host={config_path}")
                cmd.append("--skip_init=true")
                workers.append(
                    (
                        label,
                        run_cmd(
                            cmd,
                            repo_root,
                            env=build_worker_env(
                                resolve_ready_queue_index(
                                    worker_rank, ready_queue_count
                                )
                            ),
                        ),
                    )
                )

            for label, worker in workers:
                stdout, stderr = worker.communicate()
                if worker.returncode != 0:
                    raise RuntimeError(
                        f"{label} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
                    )
                rows = collect_summary_rows(stdout)
                measure_rows = [row for row in rows if row["phase"] == "measure"]
                if not measure_rows:
                    raise RuntimeError(f"{label} produced no measure summary")
                best = measure_rows[-1]
                print(
                    f"{label} mean_us={best['mean']:.2f} "
                    f"ops_per_sec={best['ops']:.2f} "
                    f"key_ops_per_sec={best['key_ops']:.2f}"
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
