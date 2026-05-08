from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from run_hps_backend_compare import (
    BACKEND_ALIASES,
    SUMMARY_FIELDS as HPS_SUMMARY_FIELDS,
    run_one as run_hps_one,
)

ROOT = Path(__file__).resolve().parents[2]
HIERKV_BIN = ROOT / "third_party/HierarchicalKV/build/hierkv_backend_benchmark"

HIERKV_ALIASES = {
    "hierkv_hbm": None,
    "hierkv_0hbm": 0,
}

SUMMARY_FIELDS = HPS_SUMMARY_FIELDS

RESULT_RE = re.compile(
    r"HIERKV_BACKEND_RESULT "
    r"phase=(?P<phase>\S+) "
    r"backend=(?P<backend>\S+) "
    r"mode=(?P<mode>\S+) "
    r"distribution=(?P<distribution>\S+) "
    r"zipfian_alpha=(?P<zipfian_alpha>[0-9.eE+-]+) "
    r"threads=(?P<threads>\d+) "
    r"batch_size=(?P<batch_size>\d+) "
    r"records=(?P<records>\d+) "
    r"runtime_s=(?P<runtime_s>[0-9.eE+-]+) "
    r"batches=(?P<batches>\d+) "
    r"key_ops=(?P<key_ops>\d+) "
    r"misses=(?P<misses>\d+) "
    r"throughput_batches_sec=(?P<throughput_batches_sec>[0-9.eE+-]+) "
    r"throughput_keys_sec=(?P<throughput_keys_sec>[0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RecStore, HPS, and HierKV storage backends."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[
            "hps_hash_map",
            "hps_rocksdb",
            "dram_map_dram",
            "dram_eh_dram",
            "dram_pet_dram",
            "hierkv_hbm",
            "hierkv_0hbm",
        ],
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
            "Fetch transactions still use --threads."
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
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--ssd-capacity-bytes", type=int, default=0)
    parser.add_argument("--hierkv-binary", type=Path, default=HIERKV_BIN)
    parser.add_argument(
        "--hierkv-max-hbm-for-vectors",
        type=int,
        default=1 << 30,
        help="HBM vector budget for hierkv_hbm; hierkv_0hbm always uses 0.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def ensure_build() -> None:
    subprocess.run(
        ["cmake", "--build", "build", "--target", "backend_benchmark", "-j"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "-S", ".", "-B", "build", "-Dsm=80"],
        cwd=ROOT / "third_party/HierarchicalKV",
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--target", "hierkv_backend_benchmark", "-j"],
        cwd=ROOT / "third_party/HierarchicalKV",
        check=True,
    )


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def gflag(name: str, value: object) -> str:
    return f"--{name}={value}"


def build_hierkv_command(
    *,
    binary: Path,
    alias: str,
    mode: str,
    record_count: int,
    runtime_seconds: int,
    threads: int,
    batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    max_hbm_for_vectors: int,
) -> list[str]:
    return [
        str(binary),
        gflag("backend", alias),
        gflag("mode", mode),
        gflag("record_count", record_count),
        gflag("running_seconds", runtime_seconds),
        gflag("thread_num", threads),
        gflag("batch_size", batch_size),
        gflag("value_size", value_size),
        gflag("distribution", distribution),
        gflag("zipfian_alpha", zipfian_alpha),
        gflag("max_hbm_for_vectors", max_hbm_for_vectors),
    ]


def collect_hierkv_rows(
    text: str,
    *,
    alias: str,
    repeat: int,
    record_count: int,
    threads: int,
    batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    log_path: Path,
    exit_code: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for match in RESULT_RE.finditer(text):
        result = match.groupdict()
        rows.append(
            {
                "alias": alias,
                "backend": "hierkv",
                "index_type": "",
                "value_store_type": "HBM_HOST_VALUE_STORE",
                "mode": result.get("mode", ""),
                "repeat": repeat,
                "record_count": record_count,
                "threads": threads,
                "batch_size": batch_size,
                "value_size": value_size,
                "distribution": distribution,
                "zipfian_alpha": zipfian_alpha,
                "phase": result.get("phase", ""),
                "exit_code": exit_code,
                "runtime_s": result.get("runtime_s", ""),
                "batches": result.get("batches", ""),
                "key_ops": result.get("key_ops", ""),
                "misses": result.get("misses", ""),
                "throughput_batches_sec": result.get("throughput_batches_sec", ""),
                "throughput_keys_sec": result.get("throughput_keys_sec", ""),
                "data_path": "",
                "log_path": str(log_path),
                "error_tail": "" if exit_code == 0 else error_tail(text, exit_code),
            }
        )
    if not rows:
        rows.append(
            {
                "alias": alias,
                "backend": "hierkv",
                "index_type": "",
                "value_store_type": "HBM_HOST_VALUE_STORE",
                "mode": "",
                "repeat": repeat,
                "record_count": record_count,
                "threads": threads,
                "batch_size": batch_size,
                "value_size": value_size,
                "distribution": distribution,
                "zipfian_alpha": zipfian_alpha,
                "phase": "missing",
                "exit_code": exit_code,
                "runtime_s": "",
                "batches": "",
                "key_ops": "",
                "misses": "",
                "throughput_batches_sec": "",
                "throughput_keys_sec": "",
                "data_path": "",
                "log_path": str(log_path),
                "error_tail": error_tail(text, exit_code),
            }
        )
    return rows


def error_tail(output: str, exit_code: int) -> str:
    if exit_code == 0:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-5:])[:1000]


def run_hierkv_one(alias: str, repeat: int, args: argparse.Namespace) -> list[dict[str, object]]:
    if alias not in HIERKV_ALIASES:
        raise ValueError(f"unknown HierKV alias '{alias}'")
    max_hbm = (
        HIERKV_ALIASES[alias]
        if HIERKV_ALIASES[alias] is not None
        else args.hierkv_max_hbm_for_vectors
    )
    log_path = args.output_dir / "logs" / f"{sanitize(alias)}_{sanitize(args.mode)}_{sanitize(args.distribution)}_r{repeat}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_hierkv_command(
        binary=args.hierkv_binary,
        alias=alias,
        mode=args.mode,
        record_count=args.record_count,
        runtime_seconds=args.runtime_seconds,
        threads=args.threads,
        batch_size=args.batch_size,
        value_size=args.value_size,
        distribution=args.distribution,
        zipfian_alpha=args.zipfian_alpha,
        max_hbm_for_vectors=max_hbm,
    )
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = " ".join(cmd) + "\n\n" + proc.stdout
    log_path.write_text(output, encoding="utf-8")
    return collect_hierkv_rows(
        proc.stdout,
        alias=alias,
        repeat=repeat,
        record_count=args.record_count,
        threads=args.threads,
        batch_size=args.batch_size,
        value_size=args.value_size,
        distribution=args.distribution,
        zipfian_alpha=args.zipfian_alpha,
        log_path=log_path,
        exit_code=proc.returncode,
    )


def merge_rows(
    hps_rows: list[dict[str, object]],
    hierkv_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [*hps_rows, *hierkv_rows]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.build:
        ensure_build()
    rows: list[dict[str, object]] = []
    hps_backends = [backend for backend in args.backends if backend in BACKEND_ALIASES]
    hierkv_backends = [backend for backend in args.backends if backend in HIERKV_ALIASES]
    unknown = set(args.backends) - set(hps_backends) - set(hierkv_backends)
    if unknown:
        raise ValueError(f"unknown backend aliases: {sorted(unknown)}")
    if hierkv_backends and not args.hierkv_binary.exists():
        raise FileNotFoundError(f"HierKV benchmark binary not found: {args.hierkv_binary}")

    for repeat in range(args.repeat):
        for alias in hps_backends:
            new_rows = run_hps_one(alias, repeat, argparse.Namespace(**vars(args)))
            rows.extend(new_rows)
            write_summary(args.output_dir / "summary.csv", rows)
            if not args.keep_data:
                for row in new_rows:
                    data_path = row.get("data_path", "")
                    if data_path:
                        shutil.rmtree(str(data_path), ignore_errors=True)

        for alias in hierkv_backends:
            new_rows = run_hierkv_one(alias, repeat, args)
            rows.extend(new_rows)
            write_summary(args.output_dir / "summary.csv", rows)

    return 0 if all(int(row["exit_code"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
