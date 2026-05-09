#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


BACKEND_LABELS = {
    "fasterkv": "FasterKV",
    "dram_pet_dram": "PetHash",
    "dram_eh_dram": "ExtendibleHash",
    "hps_rocksdb": "RocksDB",
}
BACKEND_ORDER = ["fasterkv", "dram_pet_dram", "dram_eh_dram", "hps_rocksdb"]

SCAN_FIELDS = [
    "scan",
    "x_value",
    "alias",
    "backend_label",
    "count",
    "keys_M_per_s_mean",
    "keys_M_per_s_std",
    "batches_per_s_mean",
    "misses_mean",
]


@dataclass(frozen=True)
class ScanSpec:
    name: str
    glob: str
    x_field: str
    xlabel: str
    title: str
    output_stem: str
    log_x: bool = False


SCAN_SPECS = [
    ScanSpec("thread", "dense_thread_t*", "threads", "Threads", "Throughput vs. thread count", "dense_thread_throughput", True),
    ScanSpec("value_size", "dense_value_v*", "value_size", "Value size (bytes)", "Throughput vs. value size", "dense_value_size_throughput", True),
    ScanSpec("batch_size", "dense_batch_b*", "batch_size", "Batch size (keys)", "Throughput vs. batch size", "dense_batch_size_throughput", True),
    ScanSpec("zipf_alpha", "dense_zipf_a*", "zipfian_alpha", "Zipfian alpha", "Throughput vs. Zipfian skew", "dense_zipf_throughput"),
    ScanSpec("read_ratio", "dense_read_r*", "read_ratio", "Read ratio (%)", "Throughput vs. read ratio", "dense_read_ratio_throughput"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def format_mkeys(value: float | str) -> str:
    return f"{float(value) / 1_000_000:.2f}"


def format_number(value: float | str) -> str:
    return f"{float(value):.2f}"


def load_scan_rows(output_root: Path, spec: ScanSpec) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scan_dir in sorted((output_root / "raw").glob(spec.glob)):
        if "_aborted_" in scan_dir.name:
            continue
        aggregate = scan_dir / "aggregate.csv"
        if not aggregate.exists():
            continue
        for row in read_csv(aggregate):
            alias = row.get("alias", "")
            if alias not in BACKEND_LABELS:
                continue
            rows.append(
                {
                    "scan": spec.name,
                    "x_value": row.get(spec.x_field, ""),
                    "alias": alias,
                    "backend_label": BACKEND_LABELS[alias],
                    "count": row.get("count", ""),
                    "keys_M_per_s_mean": format_mkeys(row["throughput_keys_sec_mean"]),
                    "keys_M_per_s_std": format_mkeys(row["throughput_keys_sec_std"]),
                    "batches_per_s_mean": format_number(row["throughput_batches_sec_mean"]),
                    "misses_mean": format_number(row.get("misses_mean", "0")),
                }
            )
    return sorted(rows, key=lambda row: (float(row["x_value"]), BACKEND_ORDER.index(row["alias"])))


def write_scan_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_scan(rows: list[dict[str, str]], spec: ScanSpec, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    colors = {
        "fasterkv": "#0072B2",
        "dram_pet_dram": "#D55E00",
        "dram_eh_dram": "#009E73",
        "hps_rocksdb": "#666666",
    }
    markers = {
        "fasterkv": "o",
        "dram_pet_dram": "s",
        "dram_eh_dram": "^",
        "hps_rocksdb": "D",
    }
    for alias in BACKEND_ORDER:
        series = [row for row in rows if row["alias"] == alias]
        if not series:
            continue
        x = [float(row["x_value"]) for row in series]
        y = [float(row["keys_M_per_s_mean"]) for row in series]
        yerr = [float(row["keys_M_per_s_std"]) for row in series]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=BACKEND_LABELS[alias],
            color=colors[alias],
            marker=markers[alias],
            linewidth=1.8,
            markersize=4.8,
            capsize=2.5,
        )
    if spec.log_x:
        ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel("Throughput (M keys/s, log scale)")
    ax.set_title(spec.title, fontsize=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dense scan aggregate CSVs and PDF line charts.")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate_dir = args.aggregate_dir or (args.output_root / "aggregate")
    figure_dir = args.figure_dir or (args.output_root / "figures")
    for spec in SCAN_SPECS:
        rows = load_scan_rows(args.output_root, spec)
        output_stem = f"{args.prefix}{spec.output_stem}"
        write_scan_csv(aggregate_dir / f"{output_stem}.csv", rows)
        if rows:
            plot_scan(rows, spec, figure_dir / f"{output_stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
