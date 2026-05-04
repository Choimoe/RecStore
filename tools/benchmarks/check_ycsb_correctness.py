#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
YCSB_ROOT = REPO_ROOT / "tools" / "ycsb"
FAILED_RE = re.compile(r"\[(INSERT|READ|UPDATE|SCAN|READMODIFYWRITE|DELETE)-FAILED: Count=(\d+)")


@dataclass(frozen=True)
class Case:
    name: str
    db: str
    property_file: Path
    path_property: str
    extra_props: tuple[str, ...] = field(default_factory=tuple)


CASES = {
    "kvdb": Case(
        name="kvdb",
        db="kvdb",
        property_file=YCSB_ROOT / "db" / "kv_db.properties",
        path_property="hybridkv.path",
        extra_props=(
            "hybridkv.threadcount=1",
            "hybridkv.mode=compat",
            "hybridkv.read_return=parse",
            "hybridkv.shmcapacity=67108864",
            "hybridkv.ssdcapacity=134217728",
        ),
    ),
    "kvdb_embedding": Case(
        name="kvdb_embedding",
        db="kvdb",
        property_file=YCSB_ROOT / "db" / "kv_db.properties",
        path_property="hybridkv.path",
        extra_props=(
            "hybridkv.threadcount=1",
            "hybridkv.mode=embedding",
            "hybridkv.shmcapacity=67108864",
            "hybridkv.ssdcapacity=134217728",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
    "cceh": Case(
        name="cceh",
        db="cceh",
        property_file=YCSB_ROOT / "db" / "cceh.properties",
        path_property="cceh.path",
        extra_props=(
            "cceh.threadcount=1",
            "cceh.value_size=2048",
        ),
    ),
    "cceh_embedding": Case(
        name="cceh_embedding",
        db="cceh",
        property_file=YCSB_ROOT / "db" / "cceh.properties",
        path_property="cceh.path",
        extra_props=(
            "cceh.threadcount=1",
            "cceh.mode=embedding",
            "cceh.value_size=128",
            "fieldcount=1",
            "fieldlength=128",
            "readallfields=true",
            "writeallfields=true",
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run small YCSB read/update correctness checks.")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=["kvdb", "cceh"])
    parser.add_argument("--workload", default="workloada")
    parser.add_argument("--record-count", type=int, default=128)
    parser.add_argument("--operation-count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/recstore_ycsb_correctness"))
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--ycsb-bin", type=Path, default=None)
    parser.add_argument("--build", action="store_true")
    return parser


def workload_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = YCSB_ROOT / "workloads" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"workload not found: {value}")


def failed_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op, count in FAILED_RE.findall(output):
        counts[op] = counts.get(op, 0) + int(count)
    return counts


def command_for_case(
    *,
    ycsb_bin: Path,
    case: Case,
    workload: Path,
    data_path: Path,
    record_count: int,
    operation_count: int,
    threads: int,
) -> list[str]:
    cmd = [
        str(ycsb_bin),
        "-load",
        "-run",
        "-s",
        "-db",
        case.db,
        "-threads",
        str(threads),
        "-P",
        str(workload),
        "-P",
        str(case.property_file),
        "-p",
        f"{case.path_property}={data_path}",
        "-p",
        f"recordcount={record_count}",
        "-p",
        f"operationcount={operation_count}",
        "-p",
        "measurementtype=basic",
        "-p",
        "status.interval=1",
    ]
    for prop in case.extra_props:
        cmd.extend(["-p", prop])
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ycsb_bin = args.ycsb_bin or args.build_dir / "bin" / "ycsb"
    workload = workload_path(args.workload)

    if args.build:
        subprocess.run(
            ["cmake", "--build", str(args.build_dir), "--target", "ycsb", "-j"],
            cwd=REPO_ROOT,
            check=True,
        )
    if not ycsb_bin.exists():
        raise FileNotFoundError(f"ycsb binary not found: {ycsb_bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for case_name in args.cases:
        case = CASES[case_name]
        data_path = args.output_dir / "data" / case.name
        log_path = args.output_dir / f"{case.name}.log"
        if data_path.exists():
            shutil.rmtree(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        cmd = command_for_case(
            ycsb_bin=ycsb_bin,
            case=case,
            workload=workload,
            data_path=data_path,
            record_count=args.record_count,
            operation_count=args.operation_count,
            threads=args.threads,
        )
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        output = proc.stdout + proc.stderr
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\n" + output,
            encoding="utf-8",
        )
        counts = failed_counts(output)
        case_failed = proc.returncode != 0 or any(value != 0 for value in counts.values())
        failed = failed or case_failed
        status = "FAIL" if case_failed else "PASS"
        print(f"{status} {case.name}: exit={proc.returncode} failed_counts={counts} log={log_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
