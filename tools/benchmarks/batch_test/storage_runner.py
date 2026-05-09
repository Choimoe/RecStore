#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "build/bin/backend_benchmark"
RESULT_PREFIX = "BACKEND_BENCHMARK_RESULT "

SUMMARY_FIELDS = [
    "alias",
    "backend",
    "index_type",
    "value_store_type",
    "mode",
    "read_ratio",
    "repeat",
    "record_count",
    "threads",
    "batch_size",
    "value_size",
    "distribution",
    "zipfian_alpha",
    "phase",
    "exit_code",
    "runtime_s",
    "batches",
    "key_ops",
    "misses",
    "throughput_batches_sec",
    "throughput_keys_sec",
    "data_path",
    "log_path",
    "error_tail",
]


@dataclass(frozen=True)
class BackendSpec:
    alias: str
    backend: str
    index_type: str = ""
    value_store_type: str = ""
    path_required: bool = False
    extra_flags: tuple[tuple[str, object], ...] = ()


BACKENDS = {
    "fasterkv": BackendSpec("fasterkv", "fasterkv"),
    "hps_hash_map": BackendSpec("hps_hash_map", "hps_hash_map"),
    "hps_rocksdb": BackendSpec(
        "hps_rocksdb",
        "hps_rocksdb",
        path_required=True,
        extra_flags=(("hps_rocksdb_thread_num", 1),),
    ),
    "raw_rocksdb": BackendSpec("raw_rocksdb", "raw_rocksdb", path_required=True),
    "dram_map_dram": BackendSpec(
        "dram_map_dram", "recstore", "DRAM_UNORDERED_MAP", "DRAM_VALUE_STORE", True
    ),
    "dram_eh_dram": BackendSpec(
        "dram_eh_dram", "recstore", "DRAM_EXTENDIBLE_HASH", "DRAM_VALUE_STORE", True
    ),
    "dram_pet_dram": BackendSpec(
        "dram_pet_dram", "recstore", "DRAM_PET_HASH", "DRAM_VALUE_STORE", True
    ),
    "dram_eh_ssd": BackendSpec(
        "dram_eh_ssd", "recstore", "DRAM_EXTENDIBLE_HASH", "SSD_VALUE_STORE", True
    ),
    "dram_pet_ssd": BackendSpec(
        "dram_pet_ssd", "recstore", "DRAM_PET_HASH", "SSD_VALUE_STORE", True
    ),
    "dram_eh_tiered": BackendSpec(
        "dram_eh_tiered", "recstore", "DRAM_EXTENDIBLE_HASH", "TIERED_VALUE_STORE", True
    ),
}


def gflag(name: str, value: object) -> str:
    return f"--{name}={value}"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_result_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        row: dict[str, str] = {}
        for part in line.strip().split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key] = value
        rows.append(row)
    return rows


def error_tail(output: str, exit_code: int) -> str:
    if exit_code == 0:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[:1500]


def cleanup_generated_data(data_path: Path) -> None:
    shutil.rmtree(data_path, ignore_errors=True)
    for suffix in ("_value.db", "_value.db.lock"):
        data_path.with_name(data_path.name + suffix).unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0].keys()) if rows else SUMMARY_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_command(spec: BackendSpec, args: argparse.Namespace, data_path: Path) -> list[str]:
    cmd = [
        str(args.benchmark),
        gflag("backend", spec.backend),
        gflag("mode", args.mode),
        gflag("read_ratio", args.read_ratio),
        gflag("record_count", args.record_count),
        gflag("running_seconds", args.runtime_seconds),
        gflag("thread_num", args.threads),
        gflag("load_thread_num", args.load_threads),
        gflag("batch_size", args.batch_size),
        gflag("value_size", args.value_size),
        gflag("distribution", args.distribution),
        gflag("zipfian_alpha", args.zipfian_alpha),
        gflag("dram_allocator", args.dram_allocator),
        gflag("ssd_io_backend", args.ssd_io_backend),
        gflag("ssd_queue_depth", args.ssd_queue_depth),
    ]
    if spec.path_required:
        cmd.append(gflag("path", data_path))
    if spec.index_type:
        cmd.append(gflag("index_type", spec.index_type))
    if spec.value_store_type:
        cmd.append(gflag("value_store_type", spec.value_store_type))
    if args.dram_capacity_bytes:
        cmd.append(gflag("dram_capacity_bytes", args.dram_capacity_bytes))
    if args.ssd_capacity_bytes:
        cmd.append(gflag("ssd_capacity_bytes", args.ssd_capacity_bytes))
    for key, value in spec.extra_flags:
        cmd.append(gflag(key, value))
    return cmd


def run_one(spec: BackendSpec, repeat: int, args: argparse.Namespace) -> list[dict[str, object]]:
    run_name = "_".join(
        [sanitize(args.label), sanitize(spec.alias), sanitize(args.mode), sanitize(args.distribution), f"r{repeat}"]
    )
    data_path = args.output_dir / "data" / run_name
    log_path = args.output_dir / "logs" / f"{run_name}.log"
    if data_path.exists() and not args.keep_data:
        cleanup_generated_data(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(spec, args, data_path)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_seconds,
        )
        returncode = proc.returncode
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.output or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stdout += f"\nBENCHMARK_TIMEOUT timed out after {args.timeout_seconds} seconds\n"
    elapsed = time.time() - started
    log_path.write_text(" ".join(cmd) + f"\n# wall_seconds={elapsed:.6f}\n\n" + stdout, encoding="utf-8")

    parsed = parse_result_lines(stdout) or [{"phase": "missing"}]
    rows: list[dict[str, object]] = []
    for result in parsed:
        rows.append(
            {
                "alias": spec.alias,
                "backend": result.get("backend", spec.backend),
                "index_type": result.get("index_type", spec.index_type),
                "value_store_type": result.get("value_store_type", spec.value_store_type),
                "mode": args.mode,
                "read_ratio": args.read_ratio,
                "repeat": repeat,
                "record_count": args.record_count,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "value_size": args.value_size,
                "distribution": args.distribution,
                "zipfian_alpha": args.zipfian_alpha,
                "phase": result.get("phase", ""),
                "exit_code": returncode,
                "runtime_s": result.get("runtime_s", ""),
                "batches": result.get("batches", ""),
                "key_ops": result.get("key_ops", ""),
                "misses": result.get("misses", ""),
                "throughput_batches_sec": result.get("throughput_batches_sec", ""),
                "throughput_keys_sec": result.get("throughput_keys_sec", ""),
                "data_path": str(data_path),
                "log_path": str(log_path),
                "error_tail": error_tail(stdout, returncode),
            }
        )
    if not args.keep_data:
        cleanup_generated_data(data_path)
    return rows


def aggregate_run_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "alias",
        "mode",
        "read_ratio",
        "record_count",
        "threads",
        "batch_size",
        "value_size",
        "distribution",
        "zipfian_alpha",
    ]
    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("phase")) != "run" or int(row.get("exit_code", 1)) != 0:
            continue
        if row.get("throughput_keys_sec", "") == "":
            continue
        groups.setdefault(tuple(str(row[key]) for key in keys), []).append(row)

    out: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        key_values = [float(row["throughput_keys_sec"]) for row in group_rows]
        batch_values = [float(row["throughput_batches_sec"]) for row in group_rows]
        misses = [float(row.get("misses", 0) or 0) for row in group_rows]
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(
            {
                "count": len(key_values),
                "throughput_keys_sec_mean": statistics.fmean(key_values),
                "throughput_keys_sec_std": statistics.stdev(key_values) if len(key_values) > 1 else 0.0,
                "throughput_keys_sec_min": min(key_values),
                "throughput_keys_sec_max": max(key_values),
                "throughput_batches_sec_mean": statistics.fmean(batch_values),
                "throughput_batches_sec_std": statistics.stdev(batch_values) if len(batch_values) > 1 else 0.0,
                "misses_mean": statistics.fmean(misses),
            }
        )
        out.append(entry)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backend_benchmark for one parameter point.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="point")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--backends", nargs="+", default=["fasterkv", "dram_pet_dram", "dram_eh_dram", "hps_rocksdb"])
    parser.add_argument("--mode", choices=["fetch", "insert", "mixed", "fetch_insert"], default="fetch")
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--record-count", type=int, default=1_000_000)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--distribution", choices=["uniform", "zipfian"], default="uniform")
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--dram-capacity-bytes", type=int, default=0)
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--ssd-capacity-bytes", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.benchmark.exists():
        raise FileNotFoundError(args.benchmark)
    unknown = sorted(set(args.backends) - set(BACKENDS))
    if unknown:
        raise ValueError(f"unknown backends: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    manifest_path = args.output_dir / "manifest.jsonl"
    for repeat in range(args.repeat):
        for alias in args.backends:
            run_rows = run_one(BACKENDS[alias], repeat, args)
            rows.extend(run_rows)
            write_csv(args.output_dir / "summary.csv", rows, SUMMARY_FIELDS)
            with manifest_path.open("a", encoding="utf-8") as f:
                for row in run_rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            run_metric = next(
                (row.get("throughput_keys_sec", "") for row in run_rows if row.get("phase") == "run"), ""
            )
            print(f"{args.label} {alias} r{repeat}: exit={run_rows[0]['exit_code']} run_keys_sec={run_metric}")

    write_csv(args.output_dir / "aggregate.csv", aggregate_run_rows(rows))
    if args.allow_failures:
        return 0
    return 0 if all(int(row["exit_code"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
