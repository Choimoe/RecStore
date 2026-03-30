#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from launch_config import SingleDayLaunchConfig, build_config_from_sources
except ImportError:
    from .launch_config import SingleDayLaunchConfig, build_config_from_sources


def resolve_run_name(config: SingleDayLaunchConfig) -> str:
    if config.run_name:
        return config.run_name
    mode = "torchrec" if config.use_torchrec else "recstore"
    fusion = 1 if config.fuse_emb_tables else 0
    return f"{mode}-bs{config.batch_size}-pf{config.prefetch_depth}-f{fusion}-{int(time.time())}"


def detect_dataset_size(processed_dataset_path: str) -> int:
    labels_path = os.path.join(processed_dataset_path, "day_0_labels.npy")
    labels = np.load(labels_path, mmap_mode="r")
    return int(labels.shape[0])


def _require_dataset_files(processed_dataset_path: str) -> None:
    required = ("day_0_dense.npy", "day_0_sparse.npy", "day_0_labels.npy")
    if not os.path.isdir(processed_dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {processed_dataset_path}")
    for filename in required:
        full_path = os.path.join(processed_dataset_path, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Required file missing: {full_path}")


def _script_path(repo_root: str, use_torchrec: bool) -> str:
    rel = "tests/dlrm_main_torchrec_single.py" if use_torchrec else "tests/dlrm_main_single_day.py"
    return os.path.join(repo_root, rel)


def build_torchrun_command(
    python_bin: str, repo_root: str, config: SingleDayLaunchConfig, run_id: str
) -> List[str]:
    script_to_run = _script_path(repo_root, config.use_torchrec)
    cmd = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(config.nproc_per_node),
        "--rdzv_backend",
        config.rdzv_backend,
        "--rdzv_endpoint",
        config.rdzv_endpoint,
        "--rdzv_id",
        run_id,
        "--role",
        "trainer",
        script_to_run,
        "--single_day_mode",
        "--in_memory_binary_criteo_path",
        config.processed_dataset_path,
        "--batch_size",
        str(config.batch_size),
        "--learning_rate",
        str(config.learning_rate),
        "--epochs",
        str(config.epochs),
        "--pin_memory",
        "--mmap_mode",
        "--embedding_dim",
        "128",
        "--adagrad",
    ]

    if config.allow_tf32:
        cmd.append("--allow_tf32")
    if config.trace_file:
        cmd.extend(["--trace_file", config.trace_file])

    if config.use_torchrec:
        cmd.extend(["--embedding_storage", config.embedding_storage])
    else:
        if config.enable_prefetch:
            cmd.extend(["--enable_prefetch", "--prefetch_depth", str(config.prefetch_depth)])
        if config.fuse_emb_tables:
            cmd.append("--fuse-emb-tables")
        else:
            cmd.append("--no-fuse-emb-tables")
        cmd.extend(["--fuse-k", str(config.fuse_k)])
    return cmd


def _serialize_config(path: str, config: SingleDayLaunchConfig) -> None:
    payload = asdict(config)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _to_cli_overrides(args: argparse.Namespace) -> Dict[str, object]:
    mapping = {
        "use_torchrec": args.use_torchrec,
        "processed_dataset_path": args.dataset_path,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "enable_prefetch": args.enable_prefetch,
        "prefetch_depth": args.prefetch_depth,
        "fuse_emb_tables": args.fuse_emb_tables,
        "fuse_k": args.fuse_k,
        "trace_file": args.trace_file,
        "allow_tf32": args.allow_tf32,
        "embedding_storage": args.embedding_storage,
        "log_dir": args.log_dir,
        "run_name": args.run_name,
        "nproc_per_node": args.nproc_per_node,
        "rdzv_backend": args.rdzv_backend,
        "rdzv_endpoint": args.rdzv_endpoint,
        "rdzv_id": args.rdzv_id,
    }
    overrides: Dict[str, object] = {}
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value
    return overrides


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-day launcher with layered gin configs")
    parser.add_argument("--gin_config", action="append", default=[], help="Path to gin config (repeatable)")
    parser.add_argument("--gin_binding", dest="gin_bindings", action="append", default=[])
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prefetch-depth", type=int, default=None)
    parser.add_argument("--fuse-k", type=int, default=None)
    parser.add_argument("--trace-file", type=str, default=None)
    parser.add_argument("--embedding-storage", type=str, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--rdzv-backend", type=str, default=None)
    parser.add_argument("--rdzv-endpoint", type=str, default=None)
    parser.add_argument("--rdzv-id", type=str, default=None)
    parser.add_argument("--enable-prefetch", action="store_true", default=None)
    parser.add_argument("--disable-prefetch", dest="enable_prefetch", action="store_false")
    parser.add_argument("--enable-fuse-emb", dest="fuse_emb_tables", action="store_true", default=None)
    parser.add_argument("--disable-fuse-emb", dest="fuse_emb_tables", action="store_false")
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=None)
    parser.add_argument("--torchrec", dest="use_torchrec", action="store_true", default=None)
    parser.add_argument("--custom", dest="use_torchrec", action="store_false")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = os.path.dirname(__file__)
    cli_overrides = _to_cli_overrides(args)
    config = build_config_from_sources(
        gin_configs=args.gin_config,
        gin_bindings=args.gin_bindings,
        cli_overrides=cli_overrides,
    )

    mode = "TorchRec" if config.use_torchrec else "RecStore"
    run_id = config.rdzv_id if config.rdzv_id else f"run-{int(time.time())}"
    cmd = build_torchrun_command(
        python_bin=sys.executable,
        repo_root=repo_root,
        config=config,
        run_id=run_id,
    )

    run_name = resolve_run_name(config)
    run_dir = os.path.join(repo_root, config.log_dir, run_name)
    print(f"Mode: {mode}")
    print(f"Dataset path: {config.processed_dataset_path}")
    print(f"Run dir: {run_dir}")
    print("Command:", " ".join(cmd))

    if args.print_config:
        print(json.dumps(asdict(config), indent=2, sort_keys=True))
    if args.print_config or args.dry_run:
        return 0

    _require_dataset_files(config.processed_dataset_path)
    dataset_size = detect_dataset_size(config.processed_dataset_path)
    os.makedirs(run_dir, exist_ok=True)
    _serialize_config(os.path.join(run_dir, "effective_config.json"), config)
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"dataset_size": dataset_size, "mode": "torchrec" if config.use_torchrec else "recstore"}, handle)
    log_path = os.path.join(run_dir, f"training_output.{dataset_size}.{mode}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
