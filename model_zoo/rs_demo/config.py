from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def ensure_shared_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o777)
    except OSError:
        pass


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
    prefetch_depth: int = 0
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port0: int | None = None
    server_port1: int | None = None
    server_wait_seconds: float = 20.0
    allocator: str = "R2ShmMalloc"
    output_root: str = "/nas/home/shq/docker/rs_demo"
    run_id: str = ""
    jsonl: str = ""
    csv: str = ""
    local_shm_server_csv: str = ""
    recstore_main_csv: str = ""
    recstore_main_agg_csv: str = ""
    library_path: str = ""
    recstore_runtime_dir: str = ""
    server_log: str = ""
    data_dir: str = "model_zoo/torchrec_dlrm/processed_day_0_data"
    train_ratio: float = 0.8
    fuse_k: int = 30
    dense_arch_layer_sizes: str = "512,256,128"
    over_arch_layer_sizes: str = "1024,1024,512,256,1"
    backend: str = "recstore"
    nproc: int = 1
    nnodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    enable_single_node_distributed_fast_path: bool = False
    single_node_ps_backend: str = "local_shm"
    single_node_owner_policy: str = "hash_mod_world_size"
    enable_gpu_cache: bool = False
    gpu_cache_capacity: int = 0
    disable_gpu_cache_lookup_bypass: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    rdzv_backend: str = "c10d"
    rdzv_id: str = ""
    ps_type: str = "BRPC"
    torchrec_profiler: bool = False
    torchrec_dist_mode: str = "replicated"
    torchrec_memory_mode: str = "hbm"
    torchrec_profiler_warmup: int = 0
    torchrec_profiler_active: int = 2
    torchrec_profiler_repeat: int = 1
    torchrec_trace_dir: str = ""
    torchrec_main_csv: str = ""
    torchrec_main_agg_csv: str = ""
    torchrec_trace_csv: str = ""
    torchrec_compare_recstore_csv: str = ""
    torchrec_compare_csv: str = ""
    rdma_thread_num: int = 1
    rdma_max_keys_per_request: int = 4096
    rdma_use_local_memcached: str = "auto"
    rdma_memcached_host: str = "127.0.0.1"
    rdma_memcached_port: int = 21211
    rdma_transport_mode: str = "raw_message"
    rdma_put_protocol_version: int = 2
    rdma_put_v2_transfer_mode: str = "push"
    rdma_wait_timeout_ms: int = 30000
    rdma_server_ready_timeout_sec: int = 30
    rdma_put_v2_push_slot_bytes: int = 1048576
    rdma_put_v2_push_slots_per_client: int = 8

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular benchmark demo based on DLRM-style data path."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="recstore",
        choices=["recstore", "torchrec"],
    )
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument(
        "--enable-single-node-distributed-fast-path",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--single-node-ps-backend",
        type=str,
        default="local_shm",
        choices=["local_shm", "hierkv"],
    )
    parser.add_argument(
        "--single-node-owner-policy",
        type=str,
        default="hash_mod_world_size",
        choices=["hash_mod_world_size"],
    )
    parser.add_argument(
        "--enable-gpu-cache",
        action="store_true",
        default=False,
        help="Enable RecStore GPU read/write training cache for local fast path.",
    )
    parser.add_argument(
        "--gpu-cache-capacity",
        type=int,
        default=0,
        help="Number of embedding rows to keep in the RecStore GPU cache.",
    )
    parser.add_argument(
        "--disable-gpu-cache-lookup-bypass",
        action="store_true",
        default=False,
        help=(
            "Keep querying the RecStore GPU cache for large low-hit lookups. "
            "Useful for planned/lookahead cache experiments."
        ),
    )
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rdzv-backend", type=str, default="c10d")
    parser.add_argument("--rdzv-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default="/nas/home/shq/docker/rs_demo")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--ps-type",
        type=str,
        default="BRPC",
        choices=["BRPC", "GRPC", "LOCAL_SHM", "RDMA"],
    )
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
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=0,
        help=(
            "Number of future batches to issue fused embedding prefetches ahead. "
            "0 keeps the legacy issue-and-immediate-wait path."
        ),
    )
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port0", type=int, default=None)
    parser.add_argument("--server-port1", type=int, default=None)
    parser.add_argument("--server-wait-seconds", type=float, default=20.0)
    parser.add_argument("--allocator", type=str, default="R2ShmMalloc")
    parser.add_argument("--jsonl", type=str, default="")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--local-shm-server-csv", type=str, default="")
    parser.add_argument("--recstore-main-csv", type=str, default="")
    parser.add_argument("--recstore-main-agg-csv", type=str, default="")
    parser.add_argument("--library-path", type=str, default="")
    parser.add_argument("--recstore-runtime-dir", type=str, default="")
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="model_zoo/torchrec_dlrm/processed_day_0_data",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--fuse-k", type=int, default=30)
    parser.add_argument(
        "--dense-arch-layer-sizes",
        type=str,
        default="512,256,128",
    )
    parser.add_argument(
        "--over-arch-layer-sizes",
        type=str,
        default="1024,1024,512,256,1",
    )
    parser.add_argument("--torchrec-profiler", action="store_true", default=False)
    parser.add_argument(
        "--torchrec-dist-mode",
        type=str,
        default="replicated",
        choices=["replicated", "fair_remote"],
    )
    parser.add_argument(
        "--torchrec-memory-mode",
        type=str,
        default="hbm",
        choices=["hbm", "uvm_caching"],
        help="TorchRec embedding memory mode. hbm keeps the current GPU-resident baseline; uvm_caching uses TorchRec/FBGEMM fused UVM caching when available.",
    )
    parser.add_argument("--torchrec-profiler-warmup", type=int, default=0)
    parser.add_argument("--torchrec-profiler-active", type=int, default=2)
    parser.add_argument("--torchrec-profiler-repeat", type=int, default=1)
    parser.add_argument("--torchrec-trace-dir", type=str, default="")
    parser.add_argument("--torchrec-main-csv", type=str, default="")
    parser.add_argument(
        "--torchrec-main-agg-csv",
        type=str,
        default="",
    )
    parser.add_argument("--torchrec-trace-csv", type=str, default="")
    parser.add_argument(
        "--torchrec-compare-recstore-csv",
        type=str,
        default="",
        help="If provided, generate RecStore vs TorchRec comparison csv from this RecStore csv.",
    )
    parser.add_argument(
        "--torchrec-compare-csv",
        type=str,
        default="",
    )
    parser.add_argument("--rdma-thread-num", type=int, default=1)
    parser.add_argument("--rdma-max-keys-per-request", type=int, default=4096)
    parser.add_argument(
        "--rdma-use-local-memcached",
        type=str,
        default="auto",
        choices=["always", "auto", "never"],
    )
    parser.add_argument("--rdma-memcached-host", type=str, default="127.0.0.1")
    parser.add_argument("--rdma-memcached-port", type=int, default=21211)
    parser.add_argument(
        "--rdma-transport-mode",
        type=str,
        default="raw_message",
        choices=["raw_message", "descriptor_doorbell"],
    )
    parser.add_argument("--rdma-put-protocol-version", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--rdma-put-v2-transfer-mode",
        type=str,
        default="push",
        choices=["read", "push"],
    )
    parser.add_argument("--rdma-wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--rdma-server-ready-timeout-sec", type=int, default=30)
    parser.add_argument("--rdma-put-v2-push-slot-bytes", type=int, default=1048576)
    parser.add_argument("--rdma-put-v2-push-slots-per-client", type=int, default=8)
    return parser


def parse_config(argv: list[str] | None = None) -> RunConfig:
    ns = build_parser().parse_args(argv)
    cfg_kwargs = vars(ns).copy()
    cfg_kwargs.pop("no_read_before_update", None)
    cfg_kwargs.pop("no_start_server", None)
    if cfg_kwargs["nproc_per_node"] is None:
        cfg_kwargs["nproc_per_node"] = cfg_kwargs.get("nproc", 1)
    cfg = RunConfig(**cfg_kwargs)
    if ns.no_read_before_update:
        cfg.read_before_update = False
    if ns.no_start_server:
        cfg.start_server = False
    return cfg


def validate_torchrec_config(cfg: RunConfig) -> None:
    if cfg.backend != "torchrec":
        return

    if cfg.nnodes <= 0:
        raise RuntimeError("--nnodes must be greater than 0.")
    if cfg.nproc_per_node <= 0:
        raise RuntimeError("--nproc-per-node must be greater than 0.")
    if cfg.node_rank < 0 or cfg.node_rank >= cfg.nnodes:
        raise RuntimeError("--node-rank must be within [0, nnodes).")

    profiler_subargs_nondefault = any(
        [
            cfg.torchrec_profiler_warmup != 0,
            cfg.torchrec_profiler_active != 2,
            cfg.torchrec_profiler_repeat != 1,
        ]
    )

    if profiler_subargs_nondefault and not cfg.torchrec_profiler:
        raise RuntimeError(
            "TorchRec profiler sub-arguments require --torchrec-profiler."
        )
    if cfg.torchrec_dist_mode == "fair_remote":
        world_size = cfg.nnodes * cfg.nproc_per_node
        if world_size <= 1:
            raise RuntimeError("fair_remote requires world_size greater than 1.")


def validate_recstore_config(cfg: RunConfig) -> None:
    if cfg.backend != "recstore":
        return

    if cfg.nnodes <= 0:
        raise RuntimeError("--nnodes must be greater than 0.")
    if cfg.nproc_per_node <= 0:
        raise RuntimeError("--nproc-per-node must be greater than 0.")
    if cfg.node_rank < 0 or cfg.node_rank >= cfg.nnodes:
        raise RuntimeError("--node-rank must be within [0, nnodes).")
    if cfg.enable_gpu_cache and cfg.gpu_cache_capacity <= 0:
        raise RuntimeError(
            "--gpu-cache-capacity must be positive when --enable-gpu-cache is set"
        )
    if cfg.prefetch_depth < 0:
        raise RuntimeError("--prefetch-depth must be non-negative")
    if cfg.ps_type == "RDMA":
        if cfg.nnodes != 1 or cfg.nproc_per_node != 1:
            raise RuntimeError("RecStore RDMA rs_demo path currently supports single-process runs only.")
        if cfg.rdma_thread_num <= 0:
            raise RuntimeError("--rdma-thread-num must be positive")
        if cfg.rdma_max_keys_per_request <= 0:
            raise RuntimeError("--rdma-max-keys-per-request must be positive")
        if cfg.rdma_memcached_port <= 0:
            raise RuntimeError("--rdma-memcached-port must be positive")
        if cfg.rdma_put_v2_push_slot_bytes <= 0:
            raise RuntimeError("--rdma-put-v2-push-slot-bytes must be positive")
        if cfg.rdma_put_v2_push_slots_per_client <= 0:
            raise RuntimeError("--rdma-put-v2-push-slots-per-client must be positive")
    if cfg.enable_single_node_distributed_fast_path:
        if cfg.nnodes != 1:
            raise RuntimeError(
                "RecStore single-node distributed fast path requires --nnodes=1."
            )
        if cfg.nproc_per_node <= 1:
            raise RuntimeError(
                "RecStore single-node distributed fast path requires --nproc-per-node greater than 1."
            )
        if cfg.single_node_ps_backend not in {"local_shm", "hierkv"}:
            raise RuntimeError(
                "RecStore single-node distributed fast path only supports --single-node-ps-backend=local_shm or hierkv."
            )
        if cfg.single_node_owner_policy != "hash_mod_world_size":
            raise RuntimeError(
                "RecStore single-node distributed fast path only supports --single-node-owner-policy=hash_mod_world_size."
            )
    if cfg.nnodes > 1 and not cfg.recstore_runtime_dir:
        raise RuntimeError(
            "RecStore multi-node requires --recstore-runtime-dir pointing to a shared runtime directory."
        )


def ensure_run_id(cfg: RunConfig) -> None:
    if cfg.run_id:
        return
    cfg.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def populate_default_paths(cfg: RunConfig) -> None:
    ensure_run_id(cfg)
    outputs_base = Path(cfg.output_root) / "outputs" / cfg.run_id
    logs_base = Path(cfg.output_root) / "logs" / cfg.run_id

    if not cfg.jsonl:
        cfg.jsonl = str(outputs_base / "recstore_events.jsonl")
    if not cfg.csv:
        cfg.csv = str(outputs_base / "recstore_embupdate.csv")
    if not cfg.local_shm_server_csv:
        cfg.local_shm_server_csv = str(outputs_base / "recstore_local_shm_server.csv")
    if not cfg.recstore_main_csv:
        cfg.recstore_main_csv = str(outputs_base / "recstore_main.csv")
    if not cfg.recstore_main_agg_csv:
        cfg.recstore_main_agg_csv = str(outputs_base / "recstore_main_agg.csv")
    if not cfg.server_log:
        cfg.server_log = str(logs_base / "ps_server.log")
    if not cfg.torchrec_trace_dir:
        cfg.torchrec_trace_dir = str(outputs_base / "torchrec_traces")
    if not cfg.torchrec_main_csv:
        cfg.torchrec_main_csv = str(outputs_base / "torchrec_main.csv")
    if not cfg.torchrec_main_agg_csv:
        cfg.torchrec_main_agg_csv = str(outputs_base / "torchrec_main_agg.csv")
    if not cfg.torchrec_trace_csv:
        cfg.torchrec_trace_csv = str(outputs_base / "torchrec_trace.csv")
    if not cfg.torchrec_compare_csv:
        cfg.torchrec_compare_csv = str(outputs_base / "recstore_torchrec_compare.csv")


def ensure_parent_dirs(cfg: RunConfig) -> None:
    ensure_shared_dir(Path(cfg.jsonl).parent)
    ensure_shared_dir(Path(cfg.csv).parent)
    ensure_shared_dir(Path(cfg.local_shm_server_csv).parent)
    ensure_shared_dir(Path(cfg.recstore_main_csv).parent)
    ensure_shared_dir(Path(cfg.recstore_main_agg_csv).parent)
    ensure_shared_dir(Path(cfg.server_log).parent)
    ensure_shared_dir(Path(cfg.torchrec_trace_dir))
    ensure_shared_dir(Path(cfg.torchrec_main_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_main_agg_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_trace_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_compare_csv).parent)
