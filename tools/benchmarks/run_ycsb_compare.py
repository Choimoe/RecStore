from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_BIN = ROOT / "build/bin/benchmark_kv_engine"
DEFAULT_DRAM_ROOT = Path("/dev/shm/recstore")
DEFAULT_SSD_ROOT = Path("/mnt/nvme5n1_recstore")
ENGINE_LABELS = {
    "dram_eh_dram": "DRAM_EH+DRAM",
    "dram_map_dram": "DRAM_MAP+DRAM",
    "dram_pet_dram": "DRAM_PET+DRAM",
    "dram_pet_ssd": "DRAM_PET+SSD",
    "dram_eh_ssd": "DRAM_EH+SSD",
    "dram_eh_tiered": "DRAM_EH+TIERED",
    "ssd_eh_ssd": "SSD_EH+SSD",
    "KVEngineExtendibleHash": "KVEngineExtendibleHash",
    "KVEngineCCEH": "KVEngineCCEH",
}
DEFAULT_ENGINE_ORDER = list(ENGINE_LABELS.keys())
DISTRIBUTION_LABELS = {
    "uniform": "uniform",
    "zipfian": "zipfian(alpha=0.9)",
}


@dataclass(frozen=True)
class EngineSpec:
    index_type: str
    value_store_type: str
    engine_class: str = "KVEngine"


KVENGINE_ALIASES: dict[str, EngineSpec] = {
    "dram_eh_dram": EngineSpec("DRAM_EXTENDIBLE_HASH", "DRAM_VALUE_STORE"),
    # "dram_map_dram": EngineSpec("DRAM_UNORDERED_MAP", "DRAM_VALUE_STORE"),
    "dram_pet_dram": EngineSpec("DRAM_PET_HASH", "DRAM_VALUE_STORE"),
    "dram_eh_ssd": EngineSpec("DRAM_EXTENDIBLE_HASH", "SSD_VALUE_STORE"),
    "dram_pet_ssd": EngineSpec("DRAM_PET_HASH", "SSD_VALUE_STORE"),
    "dram_eh_tiered": EngineSpec("DRAM_EXTENDIBLE_HASH", "TIERED_VALUE_STORE"),
    # "ssd_eh_ssd": EngineSpec("SSD_EXTENDIBLE_HASH", "SSD_VALUE_STORE"),
    # Run specific engine implementations directly.
    "KVEngineExtendibleHash": EngineSpec(
        "DRAM_EXTENDIBLE_HASH",
        "DRAM_VALUE_STORE",
        "KVEngineExtendibleHash",
    ),
    "KVEngineCCEH": EngineSpec(
        "SSD_EXTENDIBLE_HASH",
        "SSD_VALUE_STORE",
        "KVEngineCCEH",
    ),
}

SUMMARY_FIELDS = [
    "workload",
    "engine",
    "index_type",
    "value_store_type",
    "repeat",
    "record_count",
    "operation_count",
    "threads",
    "distribution",
    "zipfian_alpha",
    "read_mode",
    "phase",
    "exit_code",
    "load_runtime_sec",
    "load_operations",
    "load_throughput_ops_sec",
    "run_runtime_sec",
    "run_operations",
    "run_throughput_ops_sec",
    "run_read_operations",
    "run_update_operations",
    "data_path",
    "log_path",
    "raw_log_path",
    "error_tail",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native benchmark_kv_engine YCSB comparisons."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(KVENGINE_ALIASES.keys()),
        help="KV engines to run. Defaults to all known engines.",
    )
    parser.add_argument("--workloads", nargs="+", default=["a","b", "c"])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "zipfian"],
        default=["uniform"],
        help="Run one or more distributions in one command.",
    )
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--record-count", type=int, default=10*1_000_000)
    parser.add_argument(
        "--operation-count",
        type=int,
        default=0,
        help="Kept for CSV compatibility; timed benchmark ignores this value.",
    )
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--value-size", type=int, default=128)
    parser.add_argument("--read-mode", choices=["exists", "get"], default="get")
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--dram-capacity-bytes", type=int, default=0)
    parser.add_argument("--ssd-capacity-bytes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True, default="../../results/ycsb_kvengine")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra benchmark_kv_engine argument, for example --extra-arg=--print_util=true.",
    )
    parser.add_argument(
        "--engine-prop",
        action="append",
        default=[],
        help="engine:gflag=value override, for example dram_eh_dram:dram_capacity_bytes=...",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw-only mode: only render aggregate CSV/chart from existing summary.csv.",
    )
    return parser.parse_args()


def engine_spec(name: str) -> EngineSpec:
    if name in KVENGINE_ALIASES:
        return KVENGINE_ALIASES[name]
    raise ValueError(f"unknown KVEngine alias '{name}'")


def split_engine_props(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--engine-prop must use engine:key=value: {item}")
        engine, prop = item.split(":", 1)
        if "=" not in prop:
            raise ValueError(f"--engine-prop must include key=value: {item}")
        out.setdefault(engine, []).append(prop)
    return out


def ensure_build() -> None:
    subprocess.run(
        ["cmake", "--build", "build", "--target", "benchmark_kv_engine", "-j"],
        cwd=ROOT,
        check=True,
    )


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def normalize_workload(workload: str) -> str:
    lower = workload.lower()
    if lower == "workloada":
        return "a"
    if lower == "workloadb":
        return "b"
    if lower == "workloadc":
        return "c"
    if lower in {"a", "b", "c"}:
        return lower
    raise ValueError(f"unknown workload '{workload}', expected a/b/c")


def gflag(name: str, value: object) -> str:
    return f"--{name}={value}"


def benchmark_command(
    *,
    spec: EngineSpec,
    workload: str,
    dram_path: Path,
    ssd_path: Path,
    args: argparse.Namespace,
    distribution: str,
    engine_props: list[str],
) -> list[str]:
    cmd = [
        str(BENCHMARK_BIN),
        gflag("dram_path", dram_path),
        gflag("ssd_path", ssd_path),
        gflag("index_type", spec.index_type),
        gflag("value_store_type", spec.value_store_type),
        gflag("engine_class", spec.engine_class),
        gflag("record_count", args.record_count),
        gflag("workload", workload),
        gflag("distribution", distribution),
        gflag("zipfian_alpha", args.zipfian_alpha),
        gflag("thread_num", args.threads),
        gflag("load_thread_num", args.load_threads),
        gflag("running_seconds", args.runtime_seconds),
        gflag("value_size", args.value_size),
        gflag("read_mode", args.read_mode),
        gflag("load", str(not args.skip_load).lower()),
        gflag("run", str(not args.skip_run).lower()),
        gflag("dram_allocator", args.dram_allocator),
        gflag("ssd_io_backend", args.ssd_io_backend),
        gflag("ssd_queue_depth", args.ssd_queue_depth),
    ]
    if args.dram_capacity_bytes:
        cmd.append(gflag("dram_capacity_bytes", args.dram_capacity_bytes))
    if args.ssd_capacity_bytes:
        cmd.append(gflag("ssd_capacity_bytes", args.ssd_capacity_bytes))
    for prop in engine_props:
        key, value = prop.split("=", 1)
        cmd.append(gflag(key, value))
    cmd.extend(args.extra_arg)
    return cmd


def parse_result_line(line: str) -> dict[str, str]:
    parts = line.strip().split()
    out: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def extract_metrics(output: str) -> tuple[dict[str, str], dict[str, str]]:
    load: dict[str, str] = {}
    run: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("YCSB_LOAD_RESULT "):
            load = parse_result_line(line)
        elif line.startswith("YCSB_RESULT "):
            run = parse_result_line(line)
    return load, run


def error_tail(output: str, exit_code: int) -> str:
    if exit_code == 0:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-5:])[:1000]


def run_one(
    *,
    engine: str,
    spec: EngineSpec,
    workload: str,
    repeat: int,
    distribution: str,
    args: argparse.Namespace,
    engine_props: list[str],
) -> dict[str, object]:
    run_name = f"{sanitize(workload)}_{sanitize(distribution)}_{sanitize(engine)}_r{repeat}"
    data_path = args.output_dir / "data" / run_name
    dram_path = DEFAULT_DRAM_ROOT / run_name
    ssd_path = DEFAULT_SSD_ROOT / run_name
    log_path = args.output_dir / "logs" / f"{run_name}.log"
    raw_log_path = args.output_dir / "logs" / f"{run_name}.raw.log"
    if not args.keep_data:
        shutil.rmtree(dram_path, ignore_errors=True)
        shutil.rmtree(ssd_path, ignore_errors=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DRAM_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_SSD_ROOT.mkdir(parents=True, exist_ok=True)
    dram_path.mkdir(parents=True, exist_ok=True)
    ssd_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = benchmark_command(
        spec=spec,
        workload=workload,
        dram_path=dram_path,
        ssd_path=ssd_path,
        args=args,
        distribution=distribution,
        engine_props=engine_props,
    )
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    output = " ".join(cmd) + "\n\n" + proc.stdout + f"\nwall_runtime_sec={elapsed}\n"
    log_path.write_text(output, encoding="utf-8")
    # Keep an unmodified benchmark output file for post-run inspection/parsing.
    raw_log_path.write_text(proc.stdout, encoding="utf-8")
    load, run = extract_metrics(proc.stdout)

    row = {
        "workload": workload,
        "engine": engine,
        "index_type": spec.index_type,
        "value_store_type": spec.value_store_type,
        "repeat": repeat,
        "record_count": args.record_count,
        "operation_count": args.operation_count,
        "threads": args.threads,
        "distribution": distribution,
        "zipfian_alpha": args.zipfian_alpha,
        "read_mode": args.read_mode,
        "phase": "load-run",
        "exit_code": proc.returncode,
        "load_runtime_sec": load.get("seconds", ""),
        "load_operations": load.get("ops", ""),
        "load_throughput_ops_sec": load.get("throughput_ops_sec", ""),
        "run_runtime_sec": run.get("runtime_s", ""),
        "run_operations": run.get("ops", ""),
        "run_throughput_ops_sec": run.get("throughput_ops_sec", ""),
        "run_read_operations": run.get("read_ops", ""),
        "run_update_operations": run.get("update_ops", ""),
        "data_path": str(data_path),
        "log_path": str(log_path),
        "raw_log_path": str(raw_log_path),
        "error_tail": error_tail(proc.stdout, proc.returncode),
    }
    if not args.keep_data:
        shutil.rmtree(dram_path, ignore_errors=True)
        shutil.rmtree(ssd_path, ignore_errors=True)
    return row


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_aggregate(rows: list[dict[str, object]], path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["distribution"]), str(row["workload"]), str(row["engine"])), []).append(row)

    out: list[dict[str, object]] = []
    for (distribution, workload, engine), group in sorted(grouped.items()):
        ok = [r for r in group if str(r["exit_code"]) == "0"]
        run_values = [float(r["run_throughput_ops_sec"]) for r in ok if r["run_throughput_ops_sec"]]
        load_values = [float(r["load_throughput_ops_sec"]) for r in ok if r["load_throughput_ops_sec"]]
        run_ops = [float(r["run_operations"]) for r in ok if r["run_operations"]]
        out.append(
            {
                "distribution": distribution,
                "distribution_label": DISTRIBUTION_LABELS.get(distribution, distribution),
                "workload": workload,
                "engine": engine,
                "engine_label": ENGINE_LABELS.get(engine, engine),
                "runs": len(group),
                "successes": len(ok),
                "avg_load_ops_sec": sum(load_values) / len(load_values) if load_values else "",
                "avg_run_ops_sec": sum(run_values) / len(run_values) if run_values else "",
                "avg_run_operations": sum(run_ops) / len(run_ops) if run_ops else "",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "distribution",
            "distribution_label",
            "workload",
            "engine",
            "engine_label",
            "runs",
            "successes",
            "avg_load_ops_sec",
            "avg_run_ops_sec",
            "avg_run_operations",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
    return out


def format_ops(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def render_chart(rows: list[dict[str, object]], svg_path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to render the YCSB chart. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc

    categories = sorted({(str(r["distribution"]), str(r["workload"])) for r in rows})
    engines = [e for e in DEFAULT_ENGINE_ORDER if any(str(r["engine"]) == e for r in rows)]
    engines.extend(sorted({str(r["engine"]) for r in rows} - set(engines)))
    values = {
        (str(r["distribution"]), str(r["workload"]), str(r["engine"])): float(r["avg_run_ops_sec"] or 0)
        for r in rows
    }

    x = list(range(len(categories)))
    width = 0.82 / max(len(engines), 1)
    fig_w = max(14, 2.4 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_w, 7), constrained_layout=True)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]

    for idx, engine in enumerate(engines):
        offset = (idx - (len(engines) - 1) / 2) * width
        heights = [values.get((distribution, workload, engine), 0.0) for distribution, workload in categories]
        bars = ax.bar(
            [pos + offset for pos in x],
            heights,
            width=width,
            label=ENGINE_LABELS.get(engine, engine),
            color=colors[idx % len(colors)],
        )
        ax.bar_label(
            bars,
            labels=[format_ops(v) if v > 0 else "" for v in heights],
            rotation=75,
            padding=3,
            fontsize=8,
        )

    ax.set_title("YCSB timed-run throughput by KVEngine and key distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("YCSB workload / key distribution")
    ax.set_ylabel("Run throughput (ops/sec)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{workload}\n{DISTRIBUTION_LABELS.get(distribution, distribution)}" for distribution, workload in categories],
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_ops(v)))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, fontsize=9)

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)


def run_suite(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    distribution: str,
    engine_props: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    original_output_dir = args.output_dir
    args.output_dir = output_dir
    try:
        for repeat in range(args.repeat):
            for workload_arg in args.workloads:
                workload = normalize_workload(workload_arg)
                for engine in args.engines:
                    spec = engine_spec(engine)
                    props = engine_props.get(engine, [])
                    row = run_one(
                        engine=engine,
                        spec=spec,
                        workload=workload,
                        repeat=repeat,
                        distribution=distribution,
                        args=args,
                        engine_props=props,
                    )
                    rows.append(row)
                    print(
                        f"{workload} {distribution} {engine} r{repeat}: "
                        f"exit={row['exit_code']} load={row['load_throughput_ops_sec']} "
                        f"run={row['run_throughput_ops_sec']}"
                    )
                    write_summary(args.output_dir / "summary.csv", rows)
    finally:
        args.output_dir = original_output_dir
    return rows


def main() -> int:
    args = parse_args()
    summary_path = args.output_dir / "summary.csv"
    if args.draw:
        if not summary_path.exists():
            raise FileNotFoundError(f"{summary_path} does not exist; run benchmark first")
        all_rows = load_summary(summary_path)
        aggregate_path = args.output_dir / "kvengine_workload_summary.csv"
        chart_svg_path = args.output_dir / "kvengine_ycsb_run_throughput.svg"
        aggregate_rows = write_aggregate(all_rows, aggregate_path)
        render_chart(aggregate_rows, chart_svg_path)
        print(f"summary: {summary_path}")
        print(f"aggregate: {aggregate_path}")
        print(f"chart: {chart_svg_path}")
        return 0

    if args.build:
        ensure_build()
    if not BENCHMARK_BIN.exists():
        raise FileNotFoundError(
            f"{BENCHMARK_BIN} does not exist; build target benchmark_kv_engine first"
        )
    engine_props = split_engine_props(args.engine_prop)
    distributions = args.distributions
    all_rows: list[dict[str, object]] = []
    for distribution in distributions:
        suite_output_dir = args.output_dir
        if len(distributions) > 1:
            suite_output_dir = args.output_dir / distribution
        rows = run_suite(
            args=args,
            output_dir=suite_output_dir,
            distribution=distribution,
            engine_props=engine_props,
        )
        all_rows.extend(rows)
    if len(distributions) > 1:
        write_summary(summary_path, all_rows)
    aggregate_path = args.output_dir / "kvengine_workload_summary.csv"
    chart_svg_path = args.output_dir / "kvengine_ycsb_run_throughput.svg"
    aggregate_rows = write_aggregate(all_rows, aggregate_path)
    render_chart(aggregate_rows, chart_svg_path)
    print(f"summary: {summary_path}")
    print(f"aggregate: {aggregate_path}")
    print(f"chart: {chart_svg_path}")
    return 0 if all(int(row["exit_code"]) == 0 for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
