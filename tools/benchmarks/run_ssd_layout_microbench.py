#!/usr/bin/env python3
"""Compare plain, Bandana, and MaxEmbed page reads on one co-access log."""

import argparse
import subprocess
from pathlib import Path


def write_log(path: Path, vertices: int, width: int) -> None:
    pins = vertices * width
    lines = [f"0 {vertices} {vertices} {pins}"]
    for query in range(vertices):
        group = [(query + offset) % vertices for offset in range(width)]
        lines.append(" ".join(str(vertex) for vertex in group))
    path.write_text("\n".join(lines) + "\n")


def run(cmd, log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def parse_layout_result(log_path: Path) -> dict[str, int | float]:
    result_line = next(
        (line for line in log_path.read_text().splitlines()
         if line.startswith("LAYOUT_RESULT ")),
        None,
    )
    if result_line is None:
        raise ValueError(f"{log_path} has no LAYOUT_RESULT")
    fields = dict(
        field.split("=", 1) for field in result_line.split()[1:] if "=" in field
    )
    required = ("seconds", "layout_page_reads", "layout_bytes_read")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"{log_path} missing layout metric: {', '.join(missing)}")
    seconds = float(fields["seconds"])
    bytes_read = int(fields["layout_bytes_read"])
    return {
        "page_io_count": int(fields["layout_page_reads"]),
        "bytes_read": bytes_read,
        "seconds": seconds,
        "bandwidth_bytes_sec": bytes_read / seconds if seconds > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-bin", required=True)
    parser.add_argument("--benchmark-bin", required=True)
    parser.add_argument("--ssd-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vertices", type=int, default=1024)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--value-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True)
    query = out / "query.txt"
    write_log(query, args.vertices, args.width)
    bandana = out / "bandana.bin"
    maxembed = out / "maxembed.bin"
    run(
        [
            args.layout_bin, "partition", "--query", str(query),
            "--output", str(bandana), "--row-size", str(args.value_size),
        ],
        out / "partition.log",
    )
    run(
        [
            args.layout_bin, "replicate", "--query", str(query),
            "--mapping", str(bandana), "--output", str(maxembed),
            "--rep-ratio", "1.0",
        ],
        out / "replicate.log",
    )

    rows = [
        "strategy\tpage_io_count\tbytes_read\tseconds"
        "\tbandwidth_bytes_sec\tlog"
    ]
    for strategy, mapping in (
        ("plain", ""),
        ("bandana", str(bandana)),
        ("maxembed", str(maxembed)),
    ):
        data = Path(args.ssd_path) / strategy
        data.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.benchmark_bin,
            f"--ssd_path={data}",
            "--value_store_type=SSD_VALUE_STORE",
            "--index_type=DRAM_UNORDERED_MAP",
            f"--record_count={args.vertices}",
            f"--value_size={args.value_size}",
            f"--thread_num={args.threads}",
            "--load_thread_num=1",
            "--run=false",
            "--running_seconds=1",
            f"--ssd_layout_strategy={strategy}",
            f"--ssd_layout_query={query}",
        ]
        if mapping:
            cmd.append(f"--ssd_layout_mapping={mapping}")
        log = out / f"{strategy}.log"
        run(cmd, log)
        metrics = parse_layout_result(log)
        rows.append(
            f"{strategy}\t{metrics['page_io_count']}\t{metrics['bytes_read']}"
            f"\t{metrics['seconds']:.6f}"
            f"\t{metrics['bandwidth_bytes_sec']:.6f}\t{log.name}"
        )
    (out / "summary.tsv").write_text("\n".join(rows) + "\n")
    print(out)


if __name__ == "__main__":
    main()
