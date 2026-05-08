from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_BIN = ROOT / "build/bin/hps_backend_benchmark"


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    index_type: str
    value_store_type: str


BACKEND_ALIASES: dict[str, BackendSpec] = {
    "hps_hash_map": BackendSpec("hps_hash_map", "", ""),
    "hps_rocksdb": BackendSpec("hps_rocksdb", "", ""),
    "dram_eh_dram": BackendSpec("recstore", "DRAM_EXTENDIBLE_HASH", "DRAM_VALUE_STORE"),
    "dram_map_dram": BackendSpec("recstore", "DRAM_UNORDERED_MAP", "DRAM_VALUE_STORE"),
    "dram_pet_dram": BackendSpec("recstore", "DRAM_PET_HASH", "DRAM_VALUE_STORE"),
}

SUMMARY_FIELDS = [
    "alias",
    "backend",
    "index_type",
    "value_store_type",
    "mode",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HugeCTR HPS native DB backend with RecStore BaseKV backends."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["hps_hash_map", "hps_rocksdb", "dram_eh_dram", "dram_map_dram", "dram_pet_dram"],
    )
    parser.add_argument("--mode", choices=["fetch", "insert", "mixed", "fetch_insert"], default="fetch")
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--record-count", type=int, default=1_000_000)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument(
        "--hps-rocksdb-load-threads",
        type=int,
        default=1,
        help=(
            "Load thread count used only for hps_rocksdb when --load-threads is 0. "
            "HugeCTR RocksDB can crash on large concurrent insert loads; fetch "
            "transactions still use --threads."
        ),
    )
    parser.add_argument(
        "--hps-rocksdb-db-threads",
        type=int,
        default=1,
        help=(
            "RocksDB internal thread count used only for hps_rocksdb. "
            "0 passes --threads through to RocksDB."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--distribution", choices=["uniform", "zipfian"], default="uniform")
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--dram-capacity-bytes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def ensure_build(build_jobs: int) -> None:
    if build_jobs <= 0:
        raise ValueError("--build-jobs must be greater than zero")
    subprocess.run(
        [
            "cmake",
            "--build",
            "build",
            "--target",
            "hps_backend_benchmark",
            "--",
            f"-j{build_jobs}",
        ],
        cwd=ROOT,
        check=True,
    )


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def gflag(name: str, value: object) -> str:
    return f"--{name}={value}"


def command_for(alias: str, spec: BackendSpec, data_path: Path, args: argparse.Namespace) -> list[str]:
    load_threads = args.load_threads
    if alias == "hps_rocksdb" and load_threads == 0:
        load_threads = args.hps_rocksdb_load_threads
    cmd = [
        str(BENCHMARK_BIN),
        gflag("backend", spec.backend),
        gflag("path", data_path),
        gflag("mode", args.mode),
        gflag("read_ratio", args.read_ratio),
        gflag("record_count", args.record_count),
        gflag("running_seconds", args.runtime_seconds),
        gflag("thread_num", args.threads),
        gflag("load_thread_num", load_threads),
        gflag("batch_size", args.batch_size),
        gflag("value_size", args.value_size),
        gflag("distribution", args.distribution),
        gflag("zipfian_alpha", args.zipfian_alpha),
        gflag("dram_allocator", args.dram_allocator),
    ]
    if spec.index_type:
        cmd.append(gflag("index_type", spec.index_type))
    if spec.value_store_type:
        cmd.append(gflag("value_store_type", spec.value_store_type))
    if alias == "hps_rocksdb":
        cmd.append(gflag("hps_rocksdb_thread_num", args.hps_rocksdb_db_threads))
    if args.dram_capacity_bytes:
        cmd.append(gflag("dram_capacity_bytes", args.dram_capacity_bytes))
    cmd.extend(args.extra_arg)
    return cmd


def parse_result_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.strip().split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def extract_results(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if line.startswith("HPS_BACKEND_RESULT "):
            rows.append(parse_result_line(line))
    return rows


def error_tail(output: str, exit_code: int) -> str:
    if exit_code == 0:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-5:])[:1000]


def run_one(alias: str, repeat: int, args: argparse.Namespace) -> list[dict[str, object]]:
    if alias not in BACKEND_ALIASES:
        raise ValueError(f"unknown backend alias '{alias}'")
    spec = BACKEND_ALIASES[alias]
    run_name = f"{sanitize(alias)}_{sanitize(args.mode)}_{sanitize(args.distribution)}_r{repeat}"
    data_path = args.output_dir / "data" / run_name
    log_path = args.output_dir / "logs" / f"{run_name}.log"
    if data_path.exists() and not args.keep_data:
        shutil.rmtree(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = command_for(alias, spec, data_path, args)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = " ".join(cmd) + "\n\n" + proc.stdout
    log_path.write_text(output, encoding="utf-8")
    parsed = extract_results(proc.stdout)
    rows: list[dict[str, object]] = []
    if not parsed:
        parsed = [{"phase": "missing"}]
    for result in parsed:
        rows.append(
            {
                "alias": alias,
                "backend": spec.backend,
                "index_type": spec.index_type,
                "value_store_type": spec.value_store_type,
                "mode": args.mode,
                "repeat": repeat,
                "record_count": args.record_count,
                "threads": args.threads,
                "batch_size": args.batch_size,
                "value_size": args.value_size,
                "distribution": args.distribution,
                "zipfian_alpha": args.zipfian_alpha,
                "phase": result.get("phase", ""),
                "exit_code": proc.returncode,
                "runtime_s": result.get("runtime_s", ""),
                "batches": result.get("batches", ""),
                "key_ops": result.get("key_ops", ""),
                "misses": result.get("misses", ""),
                "throughput_batches_sec": result.get("throughput_batches_sec", ""),
                "throughput_keys_sec": result.get("throughput_keys_sec", ""),
                "data_path": str(data_path),
                "log_path": str(log_path),
                "error_tail": error_tail(proc.stdout, proc.returncode),
            }
        )
    if not args.keep_data:
        shutil.rmtree(data_path, ignore_errors=True)
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.build:
        ensure_build(args.build_jobs)
    if not BENCHMARK_BIN.exists():
        raise FileNotFoundError(f"{BENCHMARK_BIN} does not exist")
    rows: list[dict[str, object]] = []
    for repeat in range(args.repeat):
        for alias in args.backends:
            new_rows = run_one(alias, repeat, args)
            rows.extend(new_rows)
            run_rows = [row for row in new_rows if row["phase"] == "run"]
            metric = run_rows[0]["throughput_keys_sec"] if run_rows else ""
            print(f"{alias} r{repeat}: exit={new_rows[0]['exit_code']} run_keys_sec={metric}")
            write_summary(args.output_dir / "summary.csv", rows)
    return 0 if all(int(row["exit_code"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
