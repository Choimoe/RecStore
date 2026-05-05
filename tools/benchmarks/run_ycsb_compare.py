#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
YCSB_ROOT = REPO_ROOT / "tools" / "ycsb"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    db: str
    property_files: tuple[Path, ...] = field(default_factory=tuple)
    path_property: str | None = None
    thread_property: str | None = None
    default_props: tuple[str, ...] = field(default_factory=tuple)


ENGINE_SPECS: dict[str, EngineSpec] = {
    "kvdb": EngineSpec(
        name="kvdb",
        db="kvdb",
        property_files=(YCSB_ROOT / "db" / "kv_db.properties",),
        path_property="hybridkv.path",
        thread_property="hybridkv.threadcount",
        default_props=(
            "hybridkv.mode=perf",
            "hybridkv.read_return=none",
        ),
    ),
    "kvdb_batch": EngineSpec(
        name="kvdb_batch",
        db="kvdb",
        property_files=(YCSB_ROOT / "db" / "kv_db.properties",),
        path_property="hybridkv.path",
        thread_property="hybridkv.threadcount",
        default_props=(
            "hybridkv.mode=perf",
            "hybridkv.read_return=none",
            "ycsb.batch=true",
        ),
    ),
    "kvdb_embedding": EngineSpec(
        name="kvdb_embedding",
        db="kvdb",
        property_files=(YCSB_ROOT / "db" / "kv_db.properties",),
        path_property="hybridkv.path",
        thread_property="hybridkv.threadcount",
        default_props=(
            "hybridkv.mode=embedding",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "kvdb_embedding_batch": EngineSpec(
        name="kvdb_embedding_batch",
        db="kvdb",
        property_files=(YCSB_ROOT / "db" / "kv_db.properties",),
        path_property="hybridkv.path",
        thread_property="hybridkv.threadcount",
        default_props=(
            "hybridkv.mode=embedding",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
            "ycsb.batch=true",
        ),
    ),
    "cceh": EngineSpec(
        name="cceh",
        db="cceh",
        property_files=(YCSB_ROOT / "db" / "cceh.properties",),
        path_property="cceh.path",
        thread_property="cceh.threadcount",
    ),
    "cceh_batch": EngineSpec(
        name="cceh_batch",
        db="cceh",
        property_files=(YCSB_ROOT / "db" / "cceh.properties",),
        path_property="cceh.path",
        thread_property="cceh.threadcount",
        default_props=("ycsb.batch=true",),
    ),
    "cceh_embedding": EngineSpec(
        name="cceh_embedding",
        db="cceh",
        property_files=(YCSB_ROOT / "db" / "cceh.properties",),
        path_property="cceh.path",
        thread_property="cceh.threadcount",
        default_props=(
            "cceh.mode=embedding",
            "cceh.value_size=128",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "cceh_embedding_batch": EngineSpec(
        name="cceh_embedding_batch",
        db="cceh",
        property_files=(YCSB_ROOT / "db" / "cceh.properties",),
        path_property="cceh.path",
        thread_property="cceh.threadcount",
        default_props=(
            "cceh.mode=embedding",
            "cceh.value_size=128",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
            "ycsb.batch=true",
        ),
    ),
    "basic": EngineSpec(name="basic", db="basic", default_props=("basic.silent=true",)),
    "rocksdb": EngineSpec(
        name="rocksdb",
        db="rocksdb",
        property_files=(YCSB_ROOT / "rocksdb" / "rocksdb.properties",),
        path_property="rocksdb.dbname",
        default_props=("rocksdb.destroy=true",),
    ),
    "rocksdb_embedding": EngineSpec(
        name="rocksdb_embedding",
        db="rocksdb",
        property_files=(YCSB_ROOT / "rocksdb" / "rocksdb.properties",),
        path_property="rocksdb.dbname",
        default_props=(
            "rocksdb.destroy=true",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "leveldb": EngineSpec(
        name="leveldb",
        db="leveldb",
        property_files=(YCSB_ROOT / "leveldb" / "leveldb.properties",),
        path_property="leveldb.dbname",
        default_props=("leveldb.destroy=true",),
    ),
    "leveldb_embedding": EngineSpec(
        name="leveldb_embedding",
        db="leveldb",
        property_files=(YCSB_ROOT / "leveldb" / "leveldb.properties",),
        path_property="leveldb.dbname",
        default_props=(
            "leveldb.destroy=true",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "lmdb": EngineSpec(
        name="lmdb",
        db="lmdb",
        property_files=(YCSB_ROOT / "lmdb" / "lmdb.properties",),
        path_property="lmdb.dbpath",
    ),
    "sqlite": EngineSpec(
        name="sqlite",
        db="sqlite",
        property_files=(YCSB_ROOT / "sqlite" / "sqlite.properties",),
        path_property="sqlite.dbpath",
    ),
    "sqlite_embedding": EngineSpec(
        name="sqlite_embedding",
        db="sqlite",
        property_files=(YCSB_ROOT / "sqlite" / "sqlite.properties",),
        path_property="sqlite.dbpath",
        default_props=(
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "wiredtiger": EngineSpec(
        name="wiredtiger",
        db="wiredtiger",
        property_files=(YCSB_ROOT / "wiredtiger" / "wiredtiger.properties",),
        path_property="wiredtiger.home",
    ),
}

METRIC_RE = re.compile(r"^(Load|Run) (runtime\(sec\)|operations\(ops\)|throughput\(ops/sec\)): (.+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YCSB across RecStore engines and optional external stores."
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["kvdb", "cceh"],
        choices=sorted(ENGINE_SPECS),
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["workloada"],
        help="YCSB workload filenames under tools/ycsb/workloads, or explicit paths.",
    )
    parser.add_argument("--record-count", type=int, default=1000)
    parser.add_argument("--operation-count", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/recstore_ycsb_compare"))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--ycsb-bin", type=Path, default=None)
    parser.add_argument(
        "--engine-ycsb-bin",
        action="append",
        default=[],
        metavar="ENGINE:PATH",
        help="Use a different YCSB binary for one engine. Can be repeated.",
    )
    parser.add_argument("--build", action="store_true", help="Build the ycsb target first.")
    parser.add_argument(
        "--phase",
        choices=["load-run", "load", "run"],
        default="load-run",
        help="YCSB phase selection. load-run is the normal cold-start path.",
    )
    parser.add_argument("--measurement-type", default="basic")
    parser.add_argument(
        "--extra-prop",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Append a YCSB -p property for all engines. Can be repeated.",
    )
    parser.add_argument(
        "--engine-prop",
        action="append",
        default=[],
        metavar="ENGINE:KEY=VALUE",
        help="Append a YCSB -p property only for one engine. Can be repeated.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep per-run engine data directories after each case. Logs and summaries are always kept.",
    )
    parser.add_argument(
        "--append-summary",
        action="store_true",
        help="Append rows to existing summary.csv and summary.jsonl instead of replacing them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def workload_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = YCSB_ROOT / "workloads" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"workload not found: {value}")


def parse_engine_props(values: list[str]) -> dict[str, list[str]]:
    by_engine: dict[str, list[str]] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"--engine-prop must be ENGINE:KEY=VALUE, got {value!r}")
        engine, prop = value.split(":", 1)
        if engine not in ENGINE_SPECS:
            raise ValueError(f"unknown engine in --engine-prop: {engine}")
        if "=" not in prop:
            raise ValueError(f"--engine-prop must be ENGINE:KEY=VALUE, got {value!r}")
        by_engine.setdefault(engine, []).append(prop)
    return by_engine


def parse_engine_ycsb_bins(values: list[str]) -> dict[str, Path]:
    by_engine: dict[str, Path] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"--engine-ycsb-bin must be ENGINE:PATH, got {value!r}")
        engine, path = value.split(":", 1)
        if engine not in ENGINE_SPECS:
            raise ValueError(f"unknown engine in --engine-ycsb-bin: {engine}")
        if not path:
            raise ValueError(f"--engine-ycsb-bin path is empty for engine {engine}")
        by_engine[engine] = Path(path)
    return by_engine


def parse_metrics(output: str) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    for line in output.splitlines():
        match = METRIC_RE.match(line.strip())
        if not match:
            continue
        phase = match.group(1).lower()
        name = match.group(2)
        value = match.group(3).strip()
        key = {
            "runtime(sec)": f"{phase}_runtime_sec",
            "operations(ops)": f"{phase}_operations",
            "throughput(ops/sec)": f"{phase}_throughput_ops_sec",
        }[name]
        try:
            parsed: float | int
            parsed = int(value) if key.endswith("_operations") else float(value)
            metrics[key] = parsed
        except ValueError:
            metrics[key] = value
    return metrics


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> tuple[int, str, str]:
    printable = " ".join(cmd)
    if dry_run:
        print(printable)
        return 0, "", ""
    start = time.monotonic()
    env = os.environ.copy()
    env.setdefault("CPUPROFILE_FREQUENCY", "0")
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    elapsed = time.monotonic() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {printable}\n"
        f"# exit_code={proc.returncode} elapsed_sec={elapsed:.6f}\n\n"
        "## stdout\n"
        f"{proc.stdout}\n"
        "## stderr\n"
        f"{proc.stderr}\n",
        encoding="utf-8",
    )
    return proc.returncode, proc.stdout, proc.stderr


def path_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["du", "-sh", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.split()[0]


def build_command(
    *,
    ycsb_bin: Path,
    spec: EngineSpec,
    workload: Path,
    data_path: Path,
    phase: str,
    record_count: int,
    operation_count: int,
    threads: int,
    batch_size: int,
    measurement_type: str,
    common_props: list[str],
    engine_props: list[str],
) -> list[str]:
    cmd = [str(ycsb_bin)]
    if phase in ("load-run", "load"):
        cmd.append("-load")
    if phase in ("load-run", "run"):
        cmd.append("-run")
    cmd.extend(["-db", spec.db, "-threads", str(threads), "-P", str(workload)])
    for prop_file in spec.property_files:
        cmd.extend(["-P", str(prop_file)])

    props = [
        f"recordcount={record_count}",
        f"operationcount={operation_count}",
        f"measurementtype={measurement_type}",
    ]
    props.extend(spec.default_props)
    if "ycsb.batch=true" in spec.default_props:
        props.append(f"ycsb.batch_size={batch_size}")
    if spec.path_property is not None:
        path_value = data_path / "ycsb.sqlite3" if spec.db == "sqlite" else data_path
        props.append(f"{spec.path_property}={path_value}")
    if spec.thread_property is not None:
        props.append(f"{spec.thread_property}={threads}")
    props.extend(common_props)
    props.extend(engine_props)
    for prop in props:
        cmd.extend(["-p", prop])
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    run_root = output_dir / "data"
    log_root = output_dir / "logs"
    summary_csv = output_dir / "summary.csv"
    summary_jsonl = output_dir / "summary.jsonl"
    ycsb_bin = args.ycsb_bin or args.build_dir / "bin" / "ycsb"
    engine_props = parse_engine_props(args.engine_prop)
    engine_ycsb_bins = parse_engine_ycsb_bins(args.engine_ycsb_bin)

    if args.build and not args.dry_run:
        subprocess.run(
            ["cmake", "--build", str(args.build_dir), "--target", "ycsb", "-j"],
            cwd=REPO_ROOT,
            check=True,
        )
    if not args.dry_run and not ycsb_bin.exists():
        raise FileNotFoundError(f"ycsb binary not found: {ycsb_bin}")
    if not args.dry_run:
        for engine, engine_bin in engine_ycsb_bins.items():
            if not engine_bin.exists():
                raise FileNotFoundError(f"ycsb binary not found for engine {engine}: {engine_bin}")

    workloads = [workload_path(value) for value in args.workloads]
    rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for workload in workloads:
        workload_name = workload.name
        for engine in args.engines:
            spec = ENGINE_SPECS[engine]
            if "ycsb.batch=true" in spec.default_props and workload_name != "workloadc":
                raise ValueError("batch YCSB engines currently support workloadc only")
            for repeat_idx in range(args.repeat):
                run_id = f"{workload_name}_{engine}_r{repeat_idx}"
                data_path = run_root / run_id
                log_path = log_root / f"{run_id}.log"
                if not args.keep_data and data_path.exists():
                    shutil.rmtree(data_path)
                data_path.mkdir(parents=True, exist_ok=True)
                cmd = build_command(
                    ycsb_bin=engine_ycsb_bins.get(engine, ycsb_bin),
                    spec=spec,
                    workload=workload,
                    data_path=data_path,
                    phase=args.phase,
                    record_count=args.record_count,
                    operation_count=args.operation_count,
                    threads=args.threads,
                    batch_size=args.batch_size,
                    measurement_type=args.measurement_type,
                    common_props=args.extra_prop,
                    engine_props=engine_props.get(engine, []),
                )
                exit_code, stdout, stderr = run_cmd(cmd, REPO_ROOT, log_path, args.dry_run)
                row: dict[str, object] = {
                    "workload": workload_name,
                    "engine": engine,
                    "db": spec.db,
                    "repeat": repeat_idx,
                    "record_count": args.record_count,
                    "operation_count": args.operation_count,
                    "threads": args.threads,
                    "phase": args.phase,
                    "exit_code": exit_code,
                    "data_path": str(data_path),
                    "log_path": str(log_path),
                }
                row.update(parse_metrics(stdout))
                if exit_code != 0:
                    row["error_tail"] = (stderr or stdout)[-1000:]
                rows.append(row)
                print(
                    f"{run_id}: exit={exit_code} "
                    f"load={row.get('load_throughput_ops_sec', '')} "
                    f"run={row.get('run_throughput_ops_sec', '')} "
                    f"data_size={path_size(data_path)}"
                )
                if not args.keep_data and data_path.exists():
                    shutil.rmtree(data_path)

    if rows and not args.dry_run:
        fieldnames = [
            "workload",
            "engine",
            "db",
            "repeat",
            "record_count",
            "operation_count",
            "threads",
            "phase",
            "exit_code",
            "load_runtime_sec",
            "load_operations",
            "load_throughput_ops_sec",
            "run_runtime_sec",
            "run_operations",
            "run_throughput_ops_sec",
            "data_path",
            "log_path",
            "error_tail",
        ]
        csv_mode = "a" if args.append_summary and summary_csv.exists() else "w"
        with summary_csv.open(csv_mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if csv_mode == "w":
                writer.writeheader()
            writer.writerows(rows)
        jsonl_mode = "a" if args.append_summary and summary_jsonl.exists() else "w"
        with summary_jsonl.open(jsonl_mode, encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return 1 if any(row["exit_code"] != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
