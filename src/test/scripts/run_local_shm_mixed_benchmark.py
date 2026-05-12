#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


SUMMARY_RE = re.compile(
    r"system=(?P<system>\S+) "
    r"transport=(?P<transport>\S+) "
    r"phase=(?P<phase>\S+) "
    r"summary "
    r"rounds=(?P<rounds>\d+) "
    r"iterations=(?P<iterations>\d+) "
    r"batch_keys=(?P<batch_keys>\d+) "
    r"num_embeddings=(?P<num_embeddings>\d+) "
    r"elapsed_us_mean=(?P<mean>[0-9.eE+-]+) "
    r"elapsed_us_p50=(?P<p50>[0-9.eE+-]+) "
    r"elapsed_us_p95=(?P<p95>[0-9.eE+-]+) "
    r"elapsed_us_p99=(?P<p99>[0-9.eE+-]+) "
    r"ops_per_sec=(?P<ops>[0-9.eE+-]+) "
    r"key_ops_per_sec=(?P<key_ops>[0-9.eE+-]+)"
)


def build_runtime_config(
    region_name: str,
    slot_count: int,
    ready_queue_count: int,
    ready_queue_burst_limit: int,
    slot_buffer_bytes: int,
    client_timeout_ms: int,
    kv_path: str,
    capacity: int,
    value_size: int,
) -> dict:
    return {
        "cache_ps": {
            "ps_type": "LOCAL_SHM",
            "num_threads": 1,
            "base_kv_config": {
                "capacity": capacity,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "path": f"{kv_path}/value",
                    "default_value_size_hint": value_size,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": capacity * value_size,
                    },
                },
            },
        },
        "local_shm": {
            "region_name": region_name,
            "slot_count": slot_count,
            "ready_queue_count": ready_queue_count,
            "ready_queue_burst_limit": ready_queue_burst_limit,
            "slot_buffer_bytes": slot_buffer_bytes,
            "client_timeout_ms": client_timeout_ms,
        },
    }


def build_local_shm_server_cmd(
    server_binary: Path, config_path: Path
) -> list[str]:
    return [str(server_binary), f"--config_path={config_path}"]


def build_benchmark_cmd(
    benchmark_binary: Path,
    iterations: int,
    rounds: int,
    warmup_rounds: int,
    batch_keys: int,
    embedding_dim: int,
    num_embeddings: int,
    report_mode: str,
    update_scale: float,
    table_name: str,
) -> list[str]:
    return [
        str(benchmark_binary),
        "--transport=local_shm",
        f"--iterations={iterations}",
        f"--rounds={rounds}",
        f"--warmup_rounds={warmup_rounds}",
        f"--batch_keys={batch_keys}",
        f"--embedding_dim={embedding_dim}",
        f"--num_embeddings={num_embeddings}",
        f"--report_mode={report_mode}",
        f"--update_scale={update_scale}",
        f"--table_name={table_name}",
    ]


def collect_summary_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match is None:
            continue
        rows.append(
            {
                "system": match.group("system"),
                "transport": match.group("transport"),
                "phase": match.group("phase"),
                "rounds": int(match.group("rounds")),
                "iterations": int(match.group("iterations")),
                "batch_keys": int(match.group("batch_keys")),
                "num_embeddings": int(match.group("num_embeddings")),
                "mean": float(match.group("mean")),
                "p50": float(match.group("p50")),
                "p95": float(match.group("p95")),
                "p99": float(match.group("p99")),
                "ops": float(match.group("ops")),
                "key_ops": float(match.group("key_ops")),
            }
        )
    return rows


def print_summary_table(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        print("[summary] no parsed summary rows found")
        return

    header = [
        "system",
        "transport",
        "phase",
        "rounds",
        "iterations",
        "batch_keys",
        "num_embeddings",
        "mean_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "ops/s",
        "key_ops/s",
    ]
    table = [header]
    for row in rows:
        table.append(
            [
                str(row["system"]),
                str(row["transport"]),
                str(row["phase"]),
                str(row["rounds"]),
                str(row["iterations"]),
                str(row["batch_keys"]),
                str(row["num_embeddings"]),
                f"{row['mean']:,.2f}",
                f"{row['p50']:,.2f}",
                f"{row['p95']:,.2f}",
                f"{row['p99']:,.2f}",
                f"{row['ops']:,.2f}",
                f"{row['key_ops']:,.2f}",
            ]
        )

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(row))
        ) + " |"

    separator = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    print("\n=== Local SHM Mixed Benchmark Summary ===")
    print(render(table[0]))
    print(separator)
    for row in table[1:]:
        print(render(row))


def wait_process_ready(process: subprocess.Popen[str], delay_s: float) -> None:
    time.sleep(delay_s)
    if process.poll() is not None:
        raise RuntimeError(f"local_shm_ps_server exited early with code {process.returncode}")


def run_cmd(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
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
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/app/RecStore")
    parser.add_argument(
        "--benchmark-binary",
        default="/app/RecStore/build/src/benchmark/recstore_mixed_benchmark",
    )
    parser.add_argument(
        "--server-binary",
        default="/app/RecStore/build/bin/local_shm_ps_server",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--batch-keys", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument(
        "--report-mode", choices=["summary", "per_round", "both"], default="summary"
    )
    parser.add_argument("--update-scale", type=float, default=0.001)
    parser.add_argument("--table-name", default="default")
    parser.add_argument("--region-name", default="recstore_local_ps")
    parser.add_argument("--slot-count", type=int, default=64)
    parser.add_argument("--ready-queue-count", type=int, default=1)
    parser.add_argument("--ready-queue-burst-limit", type=int, default=8)
    parser.add_argument("--slot-buffer-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--client-timeout-ms", type=int, default=30000)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument(
        "--value-size",
        type=int,
        default=0,
        help="KV value_size in bytes; 0 means derive from embedding_dim * sizeof(float)",
    )
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument("--keep-runtime-dir", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_binary = Path(args.benchmark_binary).resolve()
    server_binary = Path(args.server_binary).resolve()

    if not benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {benchmark_binary}")
    if not server_binary.exists():
        raise FileNotFoundError(f"server binary not found: {server_binary}")

    runtime_dir_obj = (
        tempfile.TemporaryDirectory(prefix="recstore_local_shm_bench_")
        if not args.keep_runtime_dir
        else None
    )
    runtime_dir = (
        Path(runtime_dir_obj.name)
        if runtime_dir_obj is not None
        else Path(tempfile.mkdtemp(prefix="recstore_local_shm_bench_keep_"))
    )
    kv_path = runtime_dir / "kv_store"
    config_path = runtime_dir / "local_shm_config.json"
    server_log_path = runtime_dir / "local_shm_server.log"
    derived_value_size = args.embedding_dim * 4
    if args.value_size == 0:
        value_size = derived_value_size
    else:
        if args.value_size != derived_value_size:
            raise ValueError(
                f"value_size ({args.value_size}) must equal embedding_dim * 4 "
                f"({derived_value_size}) for local_shm mixed benchmark"
            )
        value_size = args.value_size

    config = build_runtime_config(
        region_name=args.region_name,
        slot_count=args.slot_count,
        ready_queue_count=args.ready_queue_count,
        ready_queue_burst_limit=args.ready_queue_burst_limit,
        slot_buffer_bytes=args.slot_buffer_bytes,
        client_timeout_ms=args.client_timeout_ms,
        kv_path=str(kv_path),
        capacity=args.capacity,
        value_size=value_size,
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    server_cmd = build_local_shm_server_cmd(server_binary, config_path)
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            server_cmd,
            cwd=str(repo_root),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_process_ready(server, args.startup_delay)
            benchmark_cmd = build_benchmark_cmd(
                benchmark_binary=benchmark_binary,
                iterations=args.iterations,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                batch_keys=args.batch_keys,
                embedding_dim=args.embedding_dim,
                num_embeddings=args.num_embeddings,
                report_mode=args.report_mode,
                update_scale=args.update_scale,
                table_name=args.table_name,
            )
            output = run_cmd(benchmark_cmd, repo_root)
            print(output, end="" if output.endswith("\n") else "\n")
            print_summary_table(collect_summary_rows(output))
            print(f"[local_shm] runtime_dir: {runtime_dir}")
            print(f"[local_shm] server_log: {server_log_path}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    if args.keep_runtime_dir and runtime_dir_obj is None:
        pass
    elif runtime_dir_obj is None:
        shutil.rmtree(runtime_dir, ignore_errors=True)
    else:
        runtime_dir_obj.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
