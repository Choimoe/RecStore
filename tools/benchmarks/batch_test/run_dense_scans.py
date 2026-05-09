#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().with_name("storage_runner.py")
DEFAULT_BACKENDS = ["fasterkv", "dram_pet_dram", "dram_eh_dram", "hps_rocksdb"]


def csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.replace(",", " ").split() if item]


def csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item]


def successful_run_count(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sum(
        1
        for row in rows
        if row.get("phase") == "run"
        and row.get("exit_code") == "0"
        and row.get("throughput_keys_sec", "") != ""
    )


def should_skip(out_dir: Path, expected_successful_runs: int, force: bool) -> bool:
    if force:
        return False
    return (out_dir / "aggregate.csv").exists() and successful_run_count(out_dir / "summary.csv") >= expected_successful_runs


def run_point(out_dir: Path, label: str, args: argparse.Namespace, extra: list[str]) -> int:
    expected = len(args.backends) * args.repeat
    if should_skip(out_dir, expected, args.force):
        print(f"skip complete {out_dir}")
        return 0
    cmd = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(out_dir),
        "--label",
        label,
        "--backends",
        *args.backends,
        "--record-count",
        str(args.record_count),
        "--runtime-seconds",
        str(args.runtime_seconds),
        "--repeat",
        str(args.repeat),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--load-threads",
        str(args.load_threads),
        "--dram-allocator",
        args.dram_allocator,
        "--ssd-io-backend",
        args.ssd_io_backend,
        "--ssd-queue-depth",
        str(args.ssd_queue_depth),
        "--allow-failures",
        *extra,
    ]
    if args.keep_data:
        cmd.append("--keep-data")
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def tag_float(value: float) -> str:
    return str(value).replace(".", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense one-dimensional backend_benchmark scans.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory that will contain raw/<scan> outputs.")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--record-count", type=int, default=1_000_000)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--base-threads", type=int, default=16)
    parser.add_argument("--base-batch-size", type=int, default=1024)
    parser.add_argument("--base-value-size", type=int, default=512)
    parser.add_argument("--thread-values", default="1 2 4 8 12 16 24 32 48 64")
    parser.add_argument("--value-sizes", default="64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096")
    parser.add_argument("--batch-sizes", default="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192")
    parser.add_argument("--zipf-alphas", default="0.3 0.6 0.8 0.9 1.0 1.2 1.4")
    parser.add_argument("--read-ratios", default="50 70 80 90 95 98 100")
    parser.add_argument(
        "--scans",
        nargs="+",
        default=["thread", "value", "batch", "zipf", "read"],
        choices=["thread", "value", "batch", "zipf", "read"],
    )
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run points even if aggregate.csv already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.output_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    failures = 0

    if "thread" in args.scans:
        for threads in csv_ints(args.thread_values):
            label = f"dense_thread_t{threads}"
            failures += run_point(
                raw / label,
                label,
                args,
                [
                    "--mode",
                    "fetch",
                    "--read-ratio",
                    "100",
                    "--threads",
                    str(threads),
                    "--batch-size",
                    str(args.base_batch_size),
                    "--value-size",
                    str(args.base_value_size),
                ],
            )

    if "value" in args.scans:
        for value_size in csv_ints(args.value_sizes):
            label = f"dense_value_v{value_size}"
            failures += run_point(
                raw / label,
                label,
                args,
                [
                    "--mode",
                    "fetch",
                    "--read-ratio",
                    "100",
                    "--threads",
                    str(args.base_threads),
                    "--batch-size",
                    str(args.base_batch_size),
                    "--value-size",
                    str(value_size),
                ],
            )

    if "batch" in args.scans:
        for batch_size in csv_ints(args.batch_sizes):
            label = f"dense_batch_b{batch_size}"
            failures += run_point(
                raw / label,
                label,
                args,
                [
                    "--mode",
                    "fetch",
                    "--read-ratio",
                    "100",
                    "--threads",
                    str(args.base_threads),
                    "--batch-size",
                    str(batch_size),
                    "--value-size",
                    str(args.base_value_size),
                ],
            )

    if "zipf" in args.scans:
        for alpha in csv_floats(args.zipf_alphas):
            label = f"dense_zipf_a{tag_float(alpha)}"
            failures += run_point(
                raw / label,
                label,
                args,
                [
                    "--mode",
                    "fetch",
                    "--read-ratio",
                    "100",
                    "--threads",
                    str(args.base_threads),
                    "--batch-size",
                    str(args.base_batch_size),
                    "--value-size",
                    str(args.base_value_size),
                    "--distribution",
                    "zipfian",
                    "--zipfian-alpha",
                    str(alpha),
                ],
            )

    if "read" in args.scans:
        for read_ratio in csv_ints(args.read_ratios):
            label = f"dense_read_r{read_ratio}"
            failures += run_point(
                raw / label,
                label,
                args,
                [
                    "--mode",
                    "mixed",
                    "--read-ratio",
                    str(read_ratio),
                    "--threads",
                    str(args.base_threads),
                    "--batch-size",
                    str(args.base_batch_size),
                    "--value-size",
                    str(args.base_value_size),
                ],
            )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
