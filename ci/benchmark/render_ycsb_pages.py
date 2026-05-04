#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAGE_SOURCE_DIR = Path(__file__).resolve().parents[1] / "pages" / "ycsb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render YCSB CI Pages artifacts.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-history", type=Path, default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--workflow-url", default="")
    parser.add_argument("--record-count", type=int, default=1000)
    parser.add_argument("--operation-count", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=200)
    return parser


def _coerce_value(value: str) -> int | float | str:
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: _coerce_value(value) for key, value in row.items()})
    return rows


def read_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_history(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def copy_page_sources(source_dir: Path, output_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"YCSB page source directory not found: {source_dir}")
    for source in source_dir.iterdir():
        if source.is_file():
            shutil.copyfile(source, output_dir / source.name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_summary(args.summary_csv)
    output_dir = args.output_dir
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    copy_page_sources(PAGE_SOURCE_DIR, output_dir)

    run = {
        "run_id": args.run_id,
        "sha": args.sha,
        "branch": args.branch,
        "workflow_url": args.workflow_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": args.record_count,
        "operation_count": args.operation_count,
        "threads": args.threads,
    }
    latest = {"run": run, "rows": rows}
    history = read_history(args.existing_history)
    history.append(latest)
    history = history[-args.max_history :]

    shutil.copyfile(args.summary_csv, latest_dir / "summary.csv")
    write_json(latest_dir / "run.json", latest)
    write_history(output_dir / "history.jsonl", history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
