#!/usr/bin/env python3

import argparse
import csv
import json
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from petps_cluster_runner import PetPSClusterRunner


SUMMARY_RE = re.compile(
    r"transport=(?P<transport>\S+) "
    r"op=(?P<op>\S+) "
    r"phase=(?P<phase>\S+) "
    r"summary "
    r"rounds=(?P<rounds>\d+) "
    r"iterations=(?P<iterations>\d+) "
    r"batch_keys=(?P<batch_keys>\d+) "
    r"elapsed_us_mean=(?P<mean>[0-9.eE+-]+) "
    r"elapsed_us_p50=(?P<p50>[0-9.eE+-]+) "
    r"elapsed_us_p95=(?P<p95>[0-9.eE+-]+) "
    r"elapsed_us_p99=(?P<p99>[0-9.eE+-]+) "
    r"ops_per_sec=(?P<ops>[0-9.eE+-]+) "
    r"key_ops_per_sec=(?P<key_ops>[0-9.eE+-]+)"
)

PS_RESULT_PREFIX = "PS_BENCHMARK_RESULT "

DEFAULT_INDEX_TYPES = (
    "DRAM_EXTENDIBLE_HASH",
    "DRAM_UNORDERED_MAP",
    "DRAM_PET_HASH",
)
DEFAULT_TRANSPORTS = ("GRPC", "BRPC", "LOCAL_SHM", "RDMA")
DEFAULT_DRAM_DATA_ROOT = Path("/dev/shm/recstore_ps_dram_bench")


@dataclass(frozen=True)
class TransportSpec:
    transport: str
    server_binary: str
    base_port: int


TRANSPORT_SPECS = {
    "GRPC": TransportSpec("GRPC", "ps_server", 15000),
    "BRPC": TransportSpec("BRPC", "ps_server", 25000),
    "LOCAL_SHM": TransportSpec("LOCAL_SHM", "local_shm_ps_server", 0),
    "RDMA": TransportSpec("RDMA", "petps_server", 25000),
}


def build_runtime_config(
    transport: str,
    index_type: str,
    runtime_dir: Path,
    num_shards: int,
    base_port: int,
    capacity: int,
    value_size: int,
    max_keys_per_request: int,
    num_threads: int,
    dram_allocator: str,
    local_shm_region: str,
    local_shm_slot_count: int,
    local_shm_ready_queue_count: int,
    local_shm_ready_queue_burst_limit: int,
    local_shm_slot_buffer_bytes: int,
    local_shm_client_timeout_ms: int,
    dram_capacity_multiplier: float,
) -> dict:
    normalized_transport = transport.upper()
    servers = [
        {"host": "127.0.0.1", "port": base_port + shard, "shard": shard}
        for shard in range(num_shards)
    ]
    value_path = (
        DEFAULT_DRAM_DATA_ROOT
        / runtime_dir.name
        / f"{normalized_transport.lower()}_{index_type.lower()}"
        / "value"
    )
    config = {
        "cache_ps": {
            "ps_type": normalized_transport,
            "max_batch_keys_size": max_keys_per_request,
            "num_threads": num_threads,
            "num_shards": num_shards,
            "servers": servers,
            "base_kv_config": {
                "capacity": capacity,
                "index": {"type": index_type},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": value_size,
                    "dram_allocator": {
                        "type": dram_allocator,
                        "capacity_bytes": int(
                            capacity * value_size * dram_capacity_multiplier
                        ),
                    },
                    "path": str(value_path),
                },
            },
        },
        "distributed_client": {
            "num_shards": num_shards,
            "hash_method": "city_hash",
            "max_keys_per_request": max_keys_per_request,
            "servers": servers,
        },
        "client": {
            "host": "127.0.0.1",
            "port": base_port,
            "shard": 0,
        },
    }
    if normalized_transport == "LOCAL_SHM":
        config["local_shm"] = {
            "region_name": local_shm_region,
            "slot_count": local_shm_slot_count,
            "ready_queue_count": local_shm_ready_queue_count,
            "ready_queue_burst_limit": local_shm_ready_queue_burst_limit,
            "slot_buffer_bytes": local_shm_slot_buffer_bytes,
            "client_timeout_ms": local_shm_client_timeout_ms,
        }
    return config


def build_benchmark_cmd(
    benchmark_binary: Path,
    transport: str,
    host: str,
    port: int,
    num_shards: int,
    config_path: Path,
    mode: str,
    record_count: int,
    runtime_seconds: int,
    threads: int,
    load_threads: int,
    batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    read_ratio: int,
    report_mode: str,
) -> list[str]:
    return [
        str(benchmark_binary),
        f"--transport={transport.lower()}",
        f"--host={host}",
        f"--port={port}",
        f"--num_shards={num_shards}",
        f"--config_path={config_path}",
        "--workload=transactions",
        f"--mode={mode}",
        f"--record_count={record_count}",
        f"--running_seconds={runtime_seconds}",
        f"--thread_num={threads}",
        f"--load_thread_num={load_threads}",
        f"--batch_keys={batch_size}",
        f"--value_size={value_size}",
        f"--distribution={distribution}",
        f"--zipfian_alpha={zipfian_alpha}",
        f"--read_ratio={read_ratio}",
        f"--report_mode={report_mode}",
    ]


def collect_summary_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match is None or match.group("phase") != "measure":
            continue
        rows.append(
            {
                "transport": match.group("transport"),
                "op": match.group("op"),
                "phase": match.group("phase"),
                "rounds": int(match.group("rounds")),
                "iterations": int(match.group("iterations")),
                "batch_keys": int(match.group("batch_keys")),
                "mean": float(match.group("mean")),
                "p50": float(match.group("p50")),
                "p95": float(match.group("p95")),
                "p99": float(match.group("p99")),
                "ops": float(match.group("ops")),
                "key_ops": float(match.group("key_ops")),
            }
        )
    return rows


def collect_ps_result_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith(PS_RESULT_PREFIX):
            continue
        row: dict[str, str | int | float] = {}
        for part in line.strip().split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key] = value
        for key in ("threads", "batch_size", "records", "batches", "key_ops"):
            if key in row:
                row[key] = int(str(row[key]))
        for key in (
            "zipfian_alpha",
            "runtime_s",
            "throughput_batches_sec",
            "throughput_keys_sec",
        ):
            if key in row:
                row[key] = float(str(row[key]))
        rows.append(row)
    return rows


def collect_case_rows(
    process_outputs: list[tuple[int, str]],
    index_type: str,
    value_size: int,
    capacity: int,
    read_ratio: int,
    client_processes: int,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for process_id, stdout in process_outputs:
        for row in collect_ps_result_rows(stdout):
            row["index_type"] = index_type
            row["value_store_type"] = "DRAM_VALUE_STORE"
            row["value_size"] = value_size
            row["capacity"] = capacity
            row["read_ratio"] = read_ratio
            row["client_processes"] = client_processes
            row["process_id"] = process_id
            row["aggregate"] = "false"
            rows.append(row)

    aggregate_groups: dict[tuple[str, str, str, str], list[dict[str, str | int | float]]] = {}
    for row in rows:
        key = (
            str(row.get("phase", "")),
            str(row.get("transport", "")),
            str(row.get("mode", "")),
            str(row.get("distribution", "")),
        )
        aggregate_groups.setdefault(key, []).append(row)

    for group_rows in aggregate_groups.values():
        if len(group_rows) <= 1:
            continue
        first = group_rows[0]
        aggregate = dict(first)
        aggregate["process_id"] = "all"
        aggregate["aggregate"] = "true"
        aggregate["threads"] = sum(int(row["threads"]) for row in group_rows)
        aggregate["client_processes"] = len(group_rows)
        aggregate["runtime_s"] = max(float(row["runtime_s"]) for row in group_rows)
        aggregate["batches"] = sum(int(row["batches"]) for row in group_rows)
        aggregate["key_ops"] = sum(int(row["key_ops"]) for row in group_rows)
        aggregate["throughput_batches_sec"] = sum(
            float(row["throughput_batches_sec"]) for row in group_rows
        )
        aggregate["throughput_keys_sec"] = sum(
            float(row["throughput_keys_sec"]) for row in group_rows
        )
        rows.append(aggregate)

    return rows


def is_port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def wait_process_ready(process: subprocess.Popen[str], delay_s: float) -> None:
    time.sleep(delay_s)
    if process.poll() is not None:
        raise RuntimeError(f"ps server exited early with code {process.returncode}")


def wait_tcp_ports_ready(
    process: subprocess.Popen[str],
    servers: list[dict],
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ps server exited early with code {process.returncode}")
        if all(
            is_port_open(str(server["host"]), int(server["port"]))
            for server in servers
        ):
            return
        time.sleep(0.1)
    ports = [f"{server['host']}:{server['port']}" for server in servers]
    raise RuntimeError(f"timed out waiting for ps server ports: {ports}")


def run_one_case(
    repo_root: Path,
    server_binary: Path,
    benchmark_binary: Path,
    config_path: Path,
    server_log_path: Path,
    transport: str,
    num_shards: int,
    mode: str,
    record_count: int,
    runtime_seconds: int,
    threads: int,
    load_threads: int,
    batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    read_ratio: int,
    report_mode: str,
    startup_delay: float,
    client_timeout_s: int,
    rdma_thread_num: int,
    rdma_use_local_memcached: str,
    rdma_memcached_host: str,
    rdma_memcached_port: int,
    rdma_transport_mode: str,
    rdma_put_protocol_version: int,
    rdma_put_v2_transfer_mode: str,
    rdma_wait_timeout_ms: int,
    rdma_server_ready_timeout_sec: int,
    rdma_client_receive_arena_bytes: int,
    rdma_put_client_send_arena_bytes: int,
    rdma_per_thread_response_limit_bytes: int,
    rdma_put_server_scratch_bytes: int,
    rdma_put_v2_push_slot_bytes: int,
    rdma_put_v2_push_slots_per_client: int,
    rdma_put_v2_push_region_offset: int,
    show_runner_logs: bool,
    client_processes: int,
) -> list[tuple[int, str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    client = config["client"]
    cmd = build_benchmark_cmd(
        benchmark_binary=benchmark_binary,
        transport=transport,
        host=client["host"],
        port=int(client["port"]),
        num_shards=num_shards,
        config_path=config_path,
        mode=mode,
        record_count=record_count,
        runtime_seconds=runtime_seconds,
        threads=threads,
        load_threads=load_threads,
        batch_size=batch_size,
        value_size=value_size,
        distribution=distribution,
        zipfian_alpha=zipfian_alpha,
        read_ratio=read_ratio,
        report_mode=report_mode,
    )
    if transport.upper() == "RDMA":
        runner = PetPSClusterRunner(
            server_path=server_binary,
            config_path=config_path,
            num_servers=num_shards,
            num_clients=client_processes,
            thread_num=rdma_thread_num,
            value_size=value_size,
            max_kv_num_per_request=max(
                batch_size, int(config["distributed_client"]["max_keys_per_request"])
            ),
            timeout=max(startup_delay, 1.0) + 30.0,
            startup_delay=startup_delay,
            log_dir=str(server_log_path.parent),
            verbose=show_runner_logs,
            use_local_memcached=rdma_use_local_memcached,
            memcached_host=rdma_memcached_host,
            memcached_port=rdma_memcached_port,
            show_status_logs=show_runner_logs,
            show_memcached_logs=show_runner_logs,
            rdma_transport_mode=rdma_transport_mode or None,
            rdma_put_protocol_version=rdma_put_protocol_version,
            rdma_put_v2_transfer_mode=rdma_put_v2_transfer_mode,
            rdma_wait_timeout_ms=rdma_wait_timeout_ms or None,
            rdma_server_ready_timeout_sec=rdma_server_ready_timeout_sec or None,
            rdma_client_receive_arena_bytes=rdma_client_receive_arena_bytes or None,
            rdma_put_client_send_arena_bytes=rdma_put_client_send_arena_bytes or None,
            rdma_per_thread_response_limit_bytes=rdma_per_thread_response_limit_bytes
            or None,
            rdma_put_server_scratch_bytes=rdma_put_server_scratch_bytes or None,
            rdma_put_v2_push_slot_bytes=rdma_put_v2_push_slot_bytes or None,
            rdma_put_v2_push_slots_per_client=rdma_put_v2_push_slots_per_client
            or None,
            rdma_put_v2_push_region_offset=rdma_put_v2_push_region_offset or None,
            rdma_use_dram=True,
        )
        with server_log_path.open("w", encoding="utf-8") as server_log:
            server_log.write(" ".join(runner.build_server_cmd(0)) + "\n")
            server_log.write(" ".join(cmd) + "\n")
        outputs: list[tuple[int, str, str]] = []
        with runner.run():
            for process_id in range(client_processes):
                completed = runner.run_client(
                    cmd,
                    client_index=process_id,
                    stream_output=show_runner_logs,
                    timeout=(client_timeout_s if client_timeout_s > 0 else None),
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "benchmark command failed\n"
                        f"process_id={process_id}\n"
                        f"cmd={' '.join(cmd)}\n"
                        f"stdout:\n{completed.stdout}\n"
                        f"stderr:\n{completed.stderr}\n"
                        f"server_log={server_log_path}"
                    )
                outputs.append((process_id, completed.stdout, completed.stderr))
        return outputs

    with server_log_path.open("w", encoding="utf-8") as server_log:
        server_cmd = [str(server_binary), f"--config_path={config_path}"]
        if transport.upper() == "BRPC" and num_shards == 1:
            server_cmd.append(
                f"--brpc_server_port={int(config['cache_ps']['servers'][0]['port'])}"
            )
        server = subprocess.Popen(
            server_cmd,
            cwd=str(repo_root),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if transport.upper() == "LOCAL_SHM":
                wait_process_ready(server, startup_delay)
            else:
                wait_tcp_ports_ready(
                    server,
                    config["cache_ps"]["servers"],
                    timeout_s=max(startup_delay, 1.0) + 30.0,
                )
            processes = [
                subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(client_processes)
            ]
            deadline = (
                time.time() + client_timeout_s if client_timeout_s > 0 else None
            )
            outputs: list[tuple[int, str, str]] = []
            for process_id, process in enumerate(processes):
                timeout = None
                if deadline is not None:
                    timeout = max(0.1, deadline - time.time())
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    for pending in processes:
                        if pending.poll() is None:
                            pending.kill()
                    for pending in processes:
                        pending.wait()
                    raise
                if process.returncode != 0:
                    raise RuntimeError(
                        "benchmark command failed\n"
                        f"process_id={process_id}\n"
                        f"cmd={' '.join(cmd)}\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}\n"
                        f"server_log={server_log_path}"
                    )
                outputs.append((process_id, stdout, stderr))
            return outputs
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def print_summary_table(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        print("[summary] no parsed measure summary rows found")
        return

    header = [
        "index_type",
        "transport",
        "mode",
        "phase",
        "threads",
        "client_processes",
        "process_id",
        "aggregate",
        "batch_size",
        "records",
        "M keys/s",
    ]
    table = [header]
    for row in rows:
        if str(row.get("phase")) != "run":
            continue
        if str(row.get("aggregate", "true")) == "false":
            continue
        table.append(
            [
                str(row["index_type"]),
                str(row["transport"]),
                str(row["mode"]),
                str(row["phase"]),
                str(row["threads"]),
                str(row.get("client_processes", 1)),
                str(row.get("process_id", "all")),
                str(row.get("aggregate", "true")),
                str(row["batch_size"]),
                str(row["records"]),
                f"{float(row['throughput_keys_sec']) / 1e6:,.3f}",
            ]
        )

    widths = [max(len(row[i]) for row in table) for i in range(len(header))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(row))
        ) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print("\n=== PS DRAM Transport Benchmark Summary ===")
    print(render(table[0]))
    print(separator)
    for row in table[1:]:
        print(render(row))


def write_csv(rows: list[dict[str, str | int | float]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index_type",
        "value_store_type",
        "value_size",
        "capacity",
        "transport",
        "phase",
        "mode",
        "read_ratio",
        "threads",
        "client_processes",
        "process_id",
        "aggregate",
        "batch_size",
        "records",
        "distribution",
        "zipfian_alpha",
        "runtime_s",
        "batches",
        "key_ops",
        "throughput_batches_sec",
        "throughput_keys_sec",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/app/RecStore")
    parser.add_argument(
        "--benchmark-binary",
        default="/app/RecStore/build/bin/ps_transport_benchmark",
    )
    parser.add_argument("--server-bin-dir", default="/app/RecStore/build/bin")
    parser.add_argument("--transports", default=",".join(DEFAULT_TRANSPORTS))
    parser.add_argument("--index-types", default=",".join(DEFAULT_INDEX_TYPES))
    parser.add_argument(
        "--mode",
        choices=["fetch", "insert", "mixed", "fetch_insert"],
        default="fetch",
    )
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--client-processes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=1000000)
    parser.add_argument(
        "--distribution",
        choices=["uniform", "zipfian"],
        default="uniform",
    )
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--grpc-base-port", type=int, default=15000)
    parser.add_argument("--brpc-base-port", type=int, default=25000)
    parser.add_argument("--max-keys-per-request", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--dram-capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--client-timeout-s", type=int, default=120)
    parser.add_argument(
        "--report-mode",
        choices=["summary", "per_round", "both"],
        default="summary",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--keep-runtime-dir", action="store_true", default=False)
    parser.add_argument("--local-shm-region", default="recstore_local_ps")
    parser.add_argument("--local-shm-slot-count", type=int, default=64)
    parser.add_argument("--local-shm-ready-queue-count", type=int, default=1)
    parser.add_argument("--local-shm-ready-queue-burst-limit", type=int, default=8)
    parser.add_argument(
        "--local-shm-slot-buffer-bytes",
        type=int,
        default=8 * 1024 * 1024,
    )
    parser.add_argument("--local-shm-client-timeout-ms", type=int, default=30000)
    parser.add_argument("--rdma-thread-num", type=int, default=1)
    parser.add_argument(
        "--rdma-use-local-memcached",
        choices=["always", "auto", "never"],
        default="auto",
    )
    parser.add_argument("--rdma-memcached-host", default="127.0.0.1")
    parser.add_argument("--rdma-memcached-port", type=int, default=21211)
    parser.add_argument(
        "--rdma-transport-mode",
        choices=["raw_message", "descriptor_doorbell"],
        default="descriptor_doorbell",
    )
    parser.add_argument(
        "--rdma-put-protocol-version",
        type=int,
        choices=[1, 2],
        default=2,
    )
    parser.add_argument(
        "--rdma-put-v2-transfer-mode",
        choices=["read", "push"],
        default="read",
    )
    parser.add_argument("--rdma-wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--rdma-server-ready-timeout-sec", type=int, default=30)
    parser.add_argument("--rdma-client-receive-arena-bytes", type=int, default=0)
    parser.add_argument("--rdma-put-client-send-arena-bytes", type=int, default=0)
    parser.add_argument(
        "--rdma-per-thread-response-limit-bytes",
        type=int,
        default=1024 * 1024,
    )
    parser.add_argument("--rdma-put-server-scratch-bytes", type=int, default=0)
    parser.add_argument("--rdma-put-v2-push-slot-bytes", type=int, default=0)
    parser.add_argument("--rdma-put-v2-push-slots-per-client", type=int, default=0)
    parser.add_argument("--rdma-put-v2-push-region-offset", type=int, default=0)
    parser.add_argument("--show-runner-logs", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_binary = Path(args.benchmark_binary).resolve()
    server_bin_dir = Path(args.server_bin_dir).resolve()
    if not benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {benchmark_binary}")
    if args.client_processes <= 0:
        raise ValueError("--client-processes must be positive")

    transports = parse_csv_list(args.transports)
    index_types = parse_csv_list(args.index_types)
    for transport in transports:
        if transport not in TRANSPORT_SPECS:
            raise ValueError(f"unsupported transport: {transport}")

    if args.output_dir:
        runtime_dir = Path(args.output_dir).resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
    else:
        runtime_dir = Path(tempfile.mkdtemp(prefix="recstore_ps_dram_bench_"))

    rows = []
    try:
        for index_type in index_types:
            for transport in transports:
                spec = TRANSPORT_SPECS[transport]
                server_binary = server_bin_dir / spec.server_binary
                if not server_binary.exists():
                    raise FileNotFoundError(f"server binary not found: {server_binary}")
                base_port = (
                    args.grpc_base_port
                    if transport == "GRPC"
                    else args.brpc_base_port
                    if transport == "BRPC"
                    else spec.base_port
                )
                case_num_shards = 1 if transport == "LOCAL_SHM" else args.num_shards
                config = build_runtime_config(
                    transport=transport,
                    index_type=index_type,
                    runtime_dir=runtime_dir,
                    num_shards=case_num_shards,
                    base_port=base_port,
                    capacity=args.capacity,
                    value_size=args.value_size,
                    max_keys_per_request=args.max_keys_per_request,
                    num_threads=args.num_threads,
                    dram_allocator=args.dram_allocator,
                    local_shm_region=args.local_shm_region,
                    local_shm_slot_count=args.local_shm_slot_count,
                    local_shm_ready_queue_count=args.local_shm_ready_queue_count,
                    local_shm_ready_queue_burst_limit=args.local_shm_ready_queue_burst_limit,
                    local_shm_slot_buffer_bytes=args.local_shm_slot_buffer_bytes,
                    local_shm_client_timeout_ms=args.local_shm_client_timeout_ms,
                    dram_capacity_multiplier=args.dram_capacity_multiplier,
                )
                case_slug = f"{index_type.lower()}_{transport.lower()}"
                config_path = runtime_dir / f"{case_slug}.json"
                log_path = runtime_dir / f"{case_slug}_server.log"
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                print(
                    f"[case] index={index_type} transport={transport} "
                    f"config={config_path}"
                )
                outputs = run_one_case(
                    repo_root=repo_root,
                    server_binary=server_binary,
                    benchmark_binary=benchmark_binary,
                    config_path=config_path,
                    server_log_path=log_path,
                    transport=transport,
                    num_shards=case_num_shards,
                    mode=args.mode,
                    record_count=args.capacity,
                    runtime_seconds=args.runtime_seconds,
                    threads=args.threads,
                    load_threads=args.load_threads,
                    batch_size=args.batch_size,
                    value_size=args.value_size,
                    distribution=args.distribution,
                    zipfian_alpha=args.zipfian_alpha,
                    read_ratio=args.read_ratio,
                    report_mode=args.report_mode,
                    startup_delay=args.startup_delay,
                    client_timeout_s=args.client_timeout_s,
                    rdma_thread_num=args.rdma_thread_num,
                    rdma_use_local_memcached=args.rdma_use_local_memcached,
                    rdma_memcached_host=args.rdma_memcached_host,
                    rdma_memcached_port=args.rdma_memcached_port,
                    rdma_transport_mode=args.rdma_transport_mode,
                    rdma_put_protocol_version=args.rdma_put_protocol_version,
                    rdma_put_v2_transfer_mode=args.rdma_put_v2_transfer_mode,
                    rdma_wait_timeout_ms=args.rdma_wait_timeout_ms,
                    rdma_server_ready_timeout_sec=args.rdma_server_ready_timeout_sec,
                    rdma_client_receive_arena_bytes=args.rdma_client_receive_arena_bytes,
                    rdma_put_client_send_arena_bytes=(
                        args.rdma_put_client_send_arena_bytes
                    ),
                    rdma_per_thread_response_limit_bytes=(
                        args.rdma_per_thread_response_limit_bytes
                    ),
                    rdma_put_server_scratch_bytes=args.rdma_put_server_scratch_bytes,
                    rdma_put_v2_push_slot_bytes=args.rdma_put_v2_push_slot_bytes,
                    rdma_put_v2_push_slots_per_client=(
                        args.rdma_put_v2_push_slots_per_client
                    ),
                    rdma_put_v2_push_region_offset=(
                        args.rdma_put_v2_push_region_offset
                    ),
                    show_runner_logs=args.show_runner_logs,
                    client_processes=args.client_processes,
                )
                process_outputs = []
                for process_id, stdout, stderr in outputs:
                    print(f"[client] process_id={process_id}")
                    print(stdout, end="" if stdout.endswith("\n") else "\n")
                    if stderr:
                        print(stderr, end="" if stderr.endswith("\n") else "\n")
                    process_outputs.append((process_id, stdout))
                rows.extend(
                    collect_case_rows(
                        process_outputs,
                        index_type=index_type,
                        value_size=args.value_size,
                        capacity=args.capacity,
                        read_ratio=args.read_ratio,
                        client_processes=args.client_processes,
                    )
                )

        print_summary_table(rows)
        csv_path = (
            Path(args.csv_path).resolve()
            if args.csv_path
            else runtime_dir / "ps_dram_transport_benchmark.csv"
        )
        write_csv(rows, csv_path)
        print(f"[output] csv={csv_path}")
        print(f"[output] runtime_dir={runtime_dir}")
    finally:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
