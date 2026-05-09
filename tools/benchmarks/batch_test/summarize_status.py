#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PREFIXES = ["dense_thread_t", "dense_value_v", "dense_batch_b", "dense_zipf_a", "dense_read_r"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_dir(scan_dir: Path) -> dict[str, object]:
    rows = read_rows(scan_dir / "summary.csv")
    run = [row for row in rows if row.get("phase") == "run"]
    ok = [
        row
        for row in run
        if row.get("exit_code") == "0" and row.get("throughput_keys_sec", "") != ""
    ]
    failed = [row for row in rows if row.get("exit_code") not in ("", "0")]
    by_failure = Counter((row.get("alias", ""), row.get("exit_code", "")) for row in failed)
    return {
        "name": scan_dir.name,
        "run": len(run),
        "ok": len(ok),
        "fail": len(failed),
        "aggregate": (scan_dir / "aggregate.csv").exists(),
        "by_failure": by_failure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dense scan completion and failures.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--include-aborted", action="store_true")
    parser.add_argument("--prefixes", nargs="+", default=PREFIXES)
    parser.add_argument("--show-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.output_root / "raw"
    exit_code = 0
    for prefix in args.prefixes:
        scan_dirs = sorted(raw.glob(prefix + "*"))
        if not args.include_aborted:
            scan_dirs = [path for path in scan_dirs if "_aborted_" not in path.name]
        print(f"\n{prefix} dirs={len(scan_dirs)}")
        total_ok = 0
        total_fail = 0
        for scan_dir in scan_dirs:
            item = summarize_dir(scan_dir)
            total_ok += int(item["ok"])
            total_fail += int(item["fail"])
            incomplete = not item["aggregate"] or (int(item["ok"]) == 0 and int(item["fail"]) == 0)
            interesting = args.show_complete or incomplete or int(item["fail"]) > 0
            if interesting:
                failure_text = ""
                if item["by_failure"]:
                    failure_text = " " + ", ".join(
                        f"{alias}:{code}={count}"
                        for (alias, code), count in sorted(item["by_failure"].items())
                    )
                print(
                    f"{item['name']} run={item['run']} ok={item['ok']} "
                    f"fail={item['fail']} aggregate={item['aggregate']}{failure_text}"
                )
            if incomplete:
                exit_code = 1
        print(f"total_ok={total_ok} total_fail={total_fail}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
