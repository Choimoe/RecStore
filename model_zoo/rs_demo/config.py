from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    num_embeddings: int = 200000
    embedding_dim: int = 128
    batch_size: int = 4096
    steps: int = 80
    warmup_steps: int = 5
    seed: int = 20260330
    table_name: str = "mock_perf_table"
    init_rows: int = 50000
    read_before_update: bool = True
    read_mode: str = "prefetch"
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port0: int | None = None
    server_port1: int | None = None
    server_wait_seconds: float = 20.0
    allocator: str = "R2ShmMalloc"
    jsonl: str = "/tmp/rs_demo_events.jsonl"
    csv: str = "/tmp/rs_demo_embupdate.csv"
    library_path: str = ""
    server_log: str = "/tmp/rs_demo_ps_server.log"
    data_dir: str = "model_zoo/torchrec_dlrm/processed_day_0_data"
    train_ratio: float = 0.8
    fuse_k: int = 30
    backend: str = "recstore"
    ps_type: str = "BRPC"
    torchrec_profiler: bool = False
    torchrec_profiler_warmup: int = 0
    torchrec_profiler_active: int = 2
    torchrec_profiler_repeat: int = 1
    torchrec_trace_dir: str = "/tmp/rs_demo_torchrec_traces"
    torchrec_main_csv: str = "/tmp/rs_demo_torchrec_main.csv"
    torchrec_trace_csv: str = "/tmp/rs_demo_torchrec_trace.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular benchmark demo based on DLRM-style data path."
    )
    parser.add_argument(
        "--backend", type=str, default="recstore", choices=["recstore", "torchrec"]
    )
    parser.add_argument("--ps-type", type=str, default="BRPC", choices=["BRPC", "GRPC"])
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--table-name", type=str, default="mock_perf_table")
    parser.add_argument("--init-rows", type=int, default=50000)
    parser.add_argument("--read-before-update", action="store_true", default=True)
    parser.add_argument("--no-read-before-update", action="store_true")
    parser.add_argument(
        "--read-mode",
        type=str,
        default="prefetch",
        choices=["prefetch", "direct"],
        help="read path mode when read-before-update is enabled",
    )
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port0", type=int, default=None)
    parser.add_argument("--server-port1", type=int, default=None)
    parser.add_argument("--server-wait-seconds", type=float, default=20.0)
    parser.add_argument("--allocator", type=str, default="R2ShmMalloc")
    parser.add_argument("--jsonl", type=str, default="/tmp/rs_demo_events.jsonl")
    parser.add_argument("--csv", type=str, default="/tmp/rs_demo_embupdate.csv")
    parser.add_argument("--library-path", type=str, default="")
    parser.add_argument("--server-log", type=str, default="/tmp/rs_demo_ps_server.log")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="model_zoo/torchrec_dlrm/processed_day_0_data",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--fuse-k", type=int, default=30)
    parser.add_argument("--torchrec-profiler", action="store_true", default=False)
    parser.add_argument("--torchrec-profiler-warmup", type=int, default=0)
    parser.add_argument("--torchrec-profiler-active", type=int, default=2)
    parser.add_argument("--torchrec-profiler-repeat", type=int, default=1)
    parser.add_argument("--torchrec-trace-dir", type=str, default="/tmp/rs_demo_torchrec_traces")
    parser.add_argument("--torchrec-main-csv", type=str, default="/tmp/rs_demo_torchrec_main.csv")
    parser.add_argument("--torchrec-trace-csv", type=str, default="/tmp/rs_demo_torchrec_trace.csv")
    return parser


def parse_config(argv: list[str] | None = None) -> RunConfig:
    ns = build_parser().parse_args(argv)
    cfg_kwargs = vars(ns).copy()
    cfg_kwargs.pop("no_read_before_update", None)
    cfg_kwargs.pop("no_start_server", None)
    cfg = RunConfig(**cfg_kwargs)
    if ns.no_read_before_update:
        cfg.read_before_update = False
    if ns.no_start_server:
        cfg.start_server = False
    return cfg


def ensure_parent_dirs(cfg: RunConfig) -> None:
    Path(cfg.jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.server_log).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.torchrec_trace_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.torchrec_main_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.torchrec_trace_csv).parent.mkdir(parents=True, exist_ok=True)
