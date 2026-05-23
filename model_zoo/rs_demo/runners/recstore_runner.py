from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from collections import deque
from pathlib import Path
from typing import Any

import torch

from ..config import RunConfig, ensure_shared_dir, validate_recstore_config
from ..data.dlrm_source import (
    build_kjt_batch_from_dense_sparse_labels,
    build_train_dataloader,
    get_default_cat_names,
    inject_project_paths,
)
from ..runtime.hybrid_dlrm import (
    build_hybrid_dense_arch,
    parse_layer_sizes,
    prepare_hybrid_dlrm_input,
    reshape_torchrec_embeddings_for_dlrm,
    run_hybrid_backward,
    sync_device,
)
from ..runtime.recstore_distributed import ShardedRecstoreClient
from ..runtime.report import finalize_recstore_row, summarize_us, write_stage_csv
from .base import BenchmarkRunner

FAST_PATH_LOOKUP_PROFILE_KEYS = (
    "lookup_exchange_ids_ms",
    "lookup_local_lookup_ms",
    "lookup_exchange_responses_ms",
    "lookup_rebuild_ms",
    "lookup_post_rebuild_h2d_ms",
    "lookup_cpp_total_ms",
    "lookup_keys_stage_ms",
    "lookup_submit_ms",
    "lookup_wait_ms",
    "lookup_payload_pin_ms",
    "lookup_fallback_copy_ms",
    "lookup_values_h2d_enqueue_ms",
)

FAST_PATH_UPDATE_PROFILE_KEYS = (
    "trace_collect_ms",
    "trace_aggregate_ms",
    "exchange_ms",
    "owner_aggregate_ms",
    "local_update_ms",
    "local_update_cpp_total_ms",
    "local_update_keys_stage_ms",
    "local_update_grads_stage_ms",
    "local_update_shm_call_ms",
    "local_update_backend_call_ms",
    "local_update_stage_wait_ms",
)

GPU_CACHE_PROFILE_KEYS = (
    "gpu_cache_query_ms",
    "gpu_cache_backend_lookup_ms",
    "gpu_cache_fill_ms",
    "gpu_cache_update_ms",
    "gpu_cache_hit_count",
    "gpu_cache_invalidate_ms",
    "gpu_cache_request_count",
    "gpu_cache_miss_count",
)


class LookaheadPrefetcher:
    """Issue fused prefetches ahead of consumption by a fixed batch depth."""

    def __init__(self, embedding_module: Any, depth: int) -> None:
        self._embedding_module = embedding_module
        self._depth = max(0, int(depth))
        self._pending: deque[tuple[int, int, float, Any, Any]] = deque()
        self._ready: deque[tuple[int, int, float, Any, Any]] = deque()
        self._stats: dict[str, float] = {}
        self.reset_stats()

    @property
    def depth(self) -> int:
        return self._depth

    def reset_stats(self) -> None:
        self._stats = {
            "prefetch_depth": float(self._depth),
            "prefetch_issued_batches": 0.0,
            "prefetch_consumed_batches": 0.0,
            "prefetch_pending_batches": float(len(self._pending)),
            "prefetch_ready_batches": float(len(self._ready)),
            "prefetch_total_ids": 0.0,
            "prefetch_consumed_total_ids": 0.0,
            "prefetch_issue_to_consume_ms": 0.0,
        }

    def enqueue(self, sparse_features: Any) -> None:
        if self._depth <= 0:
            return
        result = self._embedding_module.issue_fused_prefetch(
            sparse_features,
            record_handle=False,
        )
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result
        self._pending.append(
            (int(handle), int(num_ids), float(issue_ts), fused_ids_cpu, fused_inverse)
        )
        self._stats["prefetch_issued_batches"] += 1.0
        self._stats["prefetch_total_ids"] += float(num_ids)
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))

    def advance(self) -> bool:
        if self._depth <= 0 or len(self._pending) <= self._depth:
            self._stats["prefetch_pending_batches"] = float(len(self._pending))
            self._stats["prefetch_ready_batches"] = float(len(self._ready))
            return False
        self._ready.append(self._pending.popleft())
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))
        return True

    def advance_all(self) -> int:
        moved = 0
        while self._pending:
            self._ready.append(self._pending.popleft())
            moved += 1
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))
        return moved

    def attach_next(self) -> bool:
        if self._depth <= 0 or not self._ready:
            self._stats["prefetch_pending_batches"] = float(len(self._pending))
            self._stats["prefetch_ready_batches"] = float(len(self._ready))
            return False
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = self._ready.popleft()
        self._embedding_module.set_fused_prefetch_handle(
            handle,
            num_ids=num_ids,
            issue_ts=issue_ts,
            fused_ids_cpu=fused_ids_cpu,
            fused_inverse=fused_inverse,
        )
        self._stats["prefetch_consumed_batches"] += 1.0
        self._stats["prefetch_consumed_total_ids"] += float(num_ids)
        self._stats["prefetch_issue_to_consume_ms"] += max(
            0.0,
            (time.perf_counter() - issue_ts) * 1e3,
        )
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))
        return True

    def consume_stats(self, *, reset: bool = True) -> dict[str, float]:
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))
        stats = dict(self._stats)
        if reset:
            self.reset_stats()
        return stats


@contextmanager
def stage_timer(row: dict[str, Any], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        row[key] = (time.perf_counter() - start) * 1e3


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


def _consume_perf_stats(obj: Any) -> dict[str, float]:
    if obj is None:
        return {}
    consume = getattr(obj, "consume_perf_stats", None)
    if consume is None:
        return {}
    stats = consume(reset=True)
    return stats if isinstance(stats, dict) else {}


def _merge_consumed_perf_stats(row: dict[str, Any], stats: dict[str, float]) -> None:
    for key, value in stats.items():
        if key in row and row[key] not in (0, 0.0, "", None):
            continue
        row[key] = value


def _reset_perf_stats(obj: Any) -> None:
    if obj is None:
        return
    reset = getattr(obj, "reset_perf_stats", None)
    if reset is not None:
        reset()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_stage_csv(path, rows)


def _merge_numeric_fields(
    row: dict[str, Any],
    profile: Any,
    keys: tuple[str, ...],
) -> None:
    if not isinstance(profile, dict):
        for key in keys:
            row.setdefault(key, 0.0)
        return
    for key in keys:
        value = profile.get(key, 0.0)
        row[key] = float(value) if isinstance(value, (int, float)) else 0.0


def _merge_gpu_cache_profile(
    row: dict[str, Any],
    kv_client: Any,
    prefix: str,
) -> None:
    getter = getattr(kv_client, "get_last_gpu_cache_profile", None)
    profile = getter() if callable(getter) else {}
    if not isinstance(profile, dict):
        profile = {}
    for key in GPU_CACHE_PROFILE_KEYS:
        value = profile.get(key, 0.0)
        row[f"{prefix}_{key}"] = float(value) if isinstance(value, (int, float)) else 0.0


def _maybe_warmup_gpu_local_shm_fast_path(
    cfg: RunConfig,
    client: Any,
    device: torch.device,
) -> bool:
    if not cfg.enable_single_node_distributed_fast_path:
        return False
    if cfg.single_node_ps_backend != "local_shm":
        return False
    if device.type != "cuda":
        return False
    is_shared_local_shm_table = getattr(client, "is_shared_local_shm_table", None)
    if not callable(is_shared_local_shm_table) or not is_shared_local_shm_table():
        return False
    activate_shard = getattr(client, "activate_shard", None)
    if callable(activate_shard):
        activate_shard(0)
    current_ps_backend = getattr(client, "current_ps_backend", None)
    set_ps_backend = getattr(client, "set_ps_backend", None)
    if callable(current_ps_backend) and callable(set_ps_backend):
        if current_ps_backend() != "local_shm":
            set_ps_backend("local_shm")
    warmup = getattr(client, "warmup_local_lookup_flat_cuda_region", None)
    if not callable(warmup):
        return False
    return bool(warmup())


def _configure_gpu_cache(
    embedding_module: Any,
    cfg: RunConfig,
    *,
    embedding_dim: int,
) -> None:
    if not getattr(cfg, "enable_gpu_cache", False):
        return
    kv_client = getattr(embedding_module, "kv_client", None)
    if kv_client is None:
        raise RuntimeError("GPU cache requires embedding module kv_client")
    enable = getattr(kv_client, "enable_gpu_cache", None)
    if not callable(enable):
        raise RuntimeError("GPU cache requires kv_client.enable_gpu_cache support")
    enabled = bool(enable(int(cfg.gpu_cache_capacity), int(embedding_dim)))
    if not enabled:
        raise RuntimeError("GPU cache enable request returned False")
    if getattr(cfg, "disable_gpu_cache_lookup_bypass", False):
        setter = getattr(kv_client, "set_gpu_cache_lookup_bypass_enabled", None)
        if not callable(setter):
            raise RuntimeError(
                "GPU cache lookup bypass control requires "
                "kv_client.set_gpu_cache_lookup_bypass_enabled support"
            )
        setter(False)


def _pick_socket_ifname() -> str | None:
    preferred = ("eno1", "eno8303")
    try:
        available = set(os.listdir("/sys/class/net"))
    except OSError:
        return None
    for name in preferred:
        if name in available:
            return name
    return None


def _debug_log_path(cfg: RunConfig, rank: int) -> Path:
    return Path(cfg.output_root) / "outputs" / cfg.run_id / f"recstore_worker_rank{rank}.log"


def _append_worker_debug(cfg: RunConfig, rank: int, message: str) -> None:
    debug_path = _debug_log_path(cfg, rank)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} rank={rank} {message}\n")


def _build_worker_fingerprint(repo_root: Path) -> dict[str, dict[str, str]]:
    fallback_repo_root = Path(__file__).resolve().parents[3]
    rel_paths = [
        "model_zoo/rs_demo/cli.py",
        "model_zoo/rs_demo/config.py",
        "model_zoo/rs_demo/runners/recstore_runner.py",
        "model_zoo/rs_demo/runtime/hybrid_dlrm.py",
    ]
    files: dict[str, str] = {}
    for rel_path in rel_paths:
        path = repo_root / rel_path
        if not path.exists():
            path = fallback_repo_root / rel_path
        files[rel_path] = hashlib.md5(path.read_bytes()).hexdigest()
    return {"files": files}


def _write_or_verify_worker_fingerprint(
    rank: int,
    world_size: int,
    fingerprint: dict[str, dict[str, str]],
    fingerprint_path: Path,
) -> None:
    del world_size
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = fingerprint_path.with_suffix(fingerprint_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if fingerprint_path.exists():
            content = fingerprint_path.read_text(encoding="utf-8")
            all_fingerprints = json.loads(content) if content.strip() else {}
        else:
            all_fingerprints = {}

        all_fingerprints[str(rank)] = fingerprint
        fingerprint_path.write_text(
            json.dumps(all_fingerprints, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        if rank != 0:
            baseline = all_fingerprints.get("0")
            if baseline is not None and baseline != fingerprint:
                raise RuntimeError(
                    f"worker fingerprint mismatch: rank0={baseline} rank{rank}={fingerprint}"
                )


def _merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                    continue
                if key in {"backend", "dist_mode"}:
                    normalized[key] = value
                    continue
                try:
                    if "." in value:
                        normalized[key] = float(value)
                    else:
                        normalized[key] = int(value)
                except (TypeError, ValueError):
                    normalized[key] = value
            merged.append(normalized)
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    _write_rows(out_path, merged)
    return merged


def _barrier_for_step_alignment(dist, device, local_rank: int, use_dist: bool) -> None:
    if not use_dist:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _build_train_dataloader_for_mode(
    repo_root: Path,
    cfg: RunConfig,
    rank: int,
):
    world_size = cfg.nnodes * cfg.nproc_per_node
    return build_train_dataloader(
        repo_root=repo_root,
        data_dir_rel=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        num_embeddings=cfg.num_embeddings,
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
        rank=rank if world_size > 1 else None,
        world_size=world_size if world_size > 1 else None,
    )


def _maybe_wrap_dense_module_for_dist(
    dense_module: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    use_dist: bool,
) -> torch.nn.Module:
    if not use_dist:
        return dense_module
    if device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            dense_module,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    return torch.nn.parallel.DistributedDataParallel(dense_module)


def detect_library_path(repo_root: Path, user_path: str) -> Path:
    if user_path:
        p = Path(user_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"library_path not found: {p}")
        return p

    candidates = [
        repo_root / "build/lib/lib_recstore_ops.so",
        repo_root / "build/lib/_recstore_ops.so",
        repo_root / "build/_recstore_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find RecStore ops library. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


class RecStoreRunner(BenchmarkRunner):
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def _rank_output_dir(self, cfg: RunConfig) -> Path:
        return Path(cfg.output_root) / "outputs" / cfg.run_id / "recstore_ranks"

    def _build_torchrun_cmd(self, repo_root: Path, cfg: RunConfig) -> list[str]:
        rdzv_endpoint = f"{cfg.master_addr}:{cfg.master_port}"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(cfg.nnodes),
            "--node_rank",
            str(cfg.node_rank),
            "--nproc_per_node",
            str(cfg.nproc_per_node),
            "--rdzv_backend",
            str(cfg.rdzv_backend),
            "--rdzv_endpoint",
            rdzv_endpoint,
            "--rdzv_id",
            str(cfg.rdzv_id),
            "--tee",
            "3",
            str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
            "--backend",
            "recstore",
            "--nnodes",
            str(cfg.nnodes),
            "--node-rank",
            str(cfg.node_rank),
            "--nproc-per-node",
            str(cfg.nproc_per_node),
            "--master-addr",
            str(cfg.master_addr),
            "--master-port",
            str(cfg.master_port),
            "--rdzv-backend",
            str(cfg.rdzv_backend),
            "--rdzv-id",
            str(cfg.rdzv_id),
            "--run-id",
            str(cfg.run_id),
            "--output-root",
            str(cfg.output_root),
            "--steps",
            str(cfg.steps),
            "--warmup-steps",
            str(cfg.warmup_steps),
            "--batch-size",
            str(cfg.batch_size),
            "--num-embeddings",
            str(cfg.num_embeddings),
            "--embedding-dim",
            str(cfg.embedding_dim),
            "--fuse-k",
            str(cfg.fuse_k),
            "--recstore-index-type",
            str(cfg.recstore_index_type),
            "--dense-arch-layer-sizes",
            str(cfg.dense_arch_layer_sizes),
            "--over-arch-layer-sizes",
            str(cfg.over_arch_layer_sizes),
            "--seed",
            str(cfg.seed),
            "--data-dir",
            cfg.data_dir,
            "--train-ratio",
            str(cfg.train_ratio),
            "--recstore-main-csv",
            str(Path(cfg.recstore_main_csv)),
            "--recstore-main-agg-csv",
            str(Path(cfg.recstore_main_agg_csv)),
            "--recstore-runtime-dir",
            str(cfg.recstore_runtime_dir),
            "--library-path",
            str(cfg.library_path),
            "--read-mode",
            str(cfg.read_mode),
            "--prefetch-depth",
            str(cfg.prefetch_depth),
            "--no-start-server",
        ]
        if cfg.enable_single_node_distributed_fast_path:
            cmd.extend(
                [
                    "--enable-single-node-distributed-fast-path",
                    "--single-node-ps-backend",
                    str(cfg.single_node_ps_backend),
                    "--single-node-owner-policy",
                    str(cfg.single_node_owner_policy),
                ]
            )
        if cfg.enable_gpu_cache:
            cmd.extend(
                [
                    "--enable-gpu-cache",
                    "--gpu-cache-capacity",
                    str(cfg.gpu_cache_capacity),
                ]
            )
            if cfg.disable_gpu_cache_lookup_bypass:
                cmd.append("--disable-gpu-cache-lookup-bypass")
        if not cfg.read_before_update:
            cmd.append("--no-read-before-update")
        return cmd

    def _run_single_process(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        return self._run_local_worker(
            repo_root=repo_root,
            cfg=cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            out_csv=Path(cfg.recstore_main_csv),
        )

    def _run_distributed(self, repo_root: Path, cfg: RunConfig) -> dict[str, Any]:
        rank_dir = self._rank_output_dir(cfg)
        ensure_shared_dir(rank_dir)

        cmd = self._build_torchrun_cmd(repo_root, cfg)
        env = os.environ.copy()
        env["RS_DEMO_RECSTORE_WORKER"] = "1"
        env["RS_DEMO_RECSTORE_WORKER_DIR"] = str(rank_dir)
        socket_ifname = _pick_socket_ifname()
        if socket_ifname:
            env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
            env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        env.setdefault("NCCL_DEBUG", "WARN")
        res = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "recstore torchrun worker failed\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )

        world_size = cfg.nnodes * cfg.nproc_per_node
        rank_csvs = [rank_dir / f"rank{rank}.csv" for rank in range(world_size)]
        missing = [str(path) for path in rank_csvs if not path.exists()]
        if missing:
            raise RuntimeError(f"missing rank csv outputs: {missing}")
        rows = _merge_rank_outputs(rank_csvs, Path(cfg.recstore_main_csv))
        return {"backend": "recstore", "rows": rows}

    def _run_local_worker(
        self,
        repo_root: Path,
        cfg: RunConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        out_csv: Path,
    ) -> dict[str, Any]:
        inject_project_paths(repo_root)
        from client import RecstoreClient  # type: ignore
        from python.pytorch.recstore.optimizer import SparseSGD  # type: ignore
        from python.pytorch.torchrec_kv.EmbeddingBag import (  # type: ignore
            RecStoreEmbeddingBagCollection,
        )
        from torch import distributed as dist
        default_cat_names = get_default_cat_names()

        library_path = detect_library_path(repo_root, cfg.library_path)
        print(f"[rs_demo] repo_root={repo_root}")
        print(f"[rs_demo] backend=recstore")
        print(f"[rs_demo] library={library_path}")

        orig_cwd = Path.cwd()
        try:
            os.chdir(str(self.runtime_dir))
            torch.manual_seed(cfg.seed)
            use_dist = world_size > 1
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            _append_worker_debug(
                cfg,
                rank,
                f"worker_start world_size={world_size} local_rank={local_rank} backend={backend}",
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            if use_dist and not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    device_id=device if device.type == "cuda" else None,
                )
            if use_dist:
                fingerprint_path = (
                    Path(cfg.output_root)
                    / "outputs"
                    / cfg.run_id
                    / "recstore_worker_fingerprints.json"
                )
                fingerprint = _build_worker_fingerprint(repo_root)
                _write_or_verify_worker_fingerprint(
                    rank=rank,
                    world_size=world_size,
                    fingerprint=fingerprint,
                    fingerprint_path=fingerprint_path,
                )
                _append_worker_debug(cfg, rank, f"worker_fingerprint {fingerprint}")
            raw_client = RecstoreClient(library_path=str(library_path))
            client = ShardedRecstoreClient(raw_client, self.runtime_dir)
            if cfg.enable_single_node_distributed_fast_path:
                client.set_ps_backend(cfg.single_node_ps_backend)
                client.activate_shard(rank)
            if cfg.read_before_update and cfg.read_mode == "prefetch":
                print("[rs_demo] sharded recstore path uses prefetch read mode")
            elif cfg.read_mode != "direct":
                print("[rs_demo] unknown read mode, fallback to direct read mode")

            dataset, dataloader = _build_train_dataloader_for_mode(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
            )

            eb_configs = [
                {
                    "name": f"t_{feature_name}",
                    "num_embeddings": int(cfg.num_embeddings),
                    "embedding_dim": int(cfg.embedding_dim),
                    "feature_names": [feature_name],
                }
                for feature_name in default_cat_names
            ]

            embedding_module = RecStoreEmbeddingBagCollection(
                embedding_bag_configs=eb_configs,
                enable_fusion=True,
                fusion_k=cfg.fuse_k,
                kv_client=client,
                initialize_tables=(rank == 0),
            )
            if cfg.enable_single_node_distributed_fast_path:
                embedding_module.enable_single_node_distributed_fast_path = True
                embedding_module.single_node_distributed_mode = "single_node"
                embedding_module.single_node_ps_backend = cfg.single_node_ps_backend
                embedding_module.single_node_owner_policy = cfg.single_node_owner_policy
            _configure_gpu_cache(
                embedding_module,
                cfg,
                embedding_dim=cfg.embedding_dim,
            )
            _append_worker_debug(
                cfg,
                rank,
                "fast_path_state "
                f"enabled={getattr(embedding_module, 'enable_single_node_distributed_fast_path', False)} "
                f"mode={getattr(embedding_module, 'single_node_distributed_mode', None)} "
                f"backend={getattr(embedding_module, 'single_node_ps_backend', None)} "
                f"owner_policy={getattr(embedding_module, 'single_node_owner_policy', None)} "
                f"dist_initialized={dist.is_initialized()} "
                f"dist_world_size={dist.get_world_size() if dist.is_initialized() else 'na'} "
                f"can_use={embedding_module._can_use_single_node_distributed_fast_path()}",
            )
            _barrier_for_step_alignment(
                dist=dist,
                device=device,
                local_rank=local_rank,
                use_dist=use_dist,
            )
            dense_module = build_hybrid_dense_arch(
                torch=torch,
                dense_in_features=13,
                embedding_dim=cfg.embedding_dim,
                num_sparse_features=len(default_cat_names),
                dense_arch_layer_sizes=parse_layer_sizes(cfg.dense_arch_layer_sizes),
                over_arch_layer_sizes=parse_layer_sizes(cfg.over_arch_layer_sizes),
                device=device,
            )
            dense_module = _maybe_wrap_dense_module_for_dist(
                dense_module=dense_module,
                device=device,
                local_rank=local_rank,
                use_dist=use_dist,
            )
            criterion = torch.nn.BCEWithLogitsLoss()
            dense_optimizer = torch.optim.SGD(dense_module.parameters(), lr=0.01)
            sparse_optimizer = SparseSGD([embedding_module], lr=0.01)
            fast_path_region_warmed = _maybe_warmup_gpu_local_shm_fast_path(
                cfg=cfg,
                client=client,
                device=device,
            )
            if fast_path_region_warmed:
                print("[rs_demo] warmed local_shm lookup payload region for GPU fast path")
                _barrier_for_step_alignment(
                    dist=dist,
                    device=device,
                    local_rank=local_rank,
                    use_dist=use_dist,
                )

            read_lat_us: list[float] = []
            update_lat_us: list[float] = []
            rows: list[dict[str, Any]] = []
            data_iter = iter(dataloader)
            lookahead_prefetcher = LookaheadPrefetcher(
                embedding_module,
                depth=cfg.prefetch_depth
                if cfg.read_before_update and cfg.read_mode == "prefetch"
                else 0,
            )
            prepared_batches: deque[
                tuple[dict[str, Any], float, Any, Any, Any, Any]
            ] = deque()

            def prepare_next_batch(batch_step: int):
                row: dict[str, Any] = {
                    "backend": "recstore",
                    "nproc": world_size,
                    "rank": rank,
                    "batch_size": cfg.batch_size,
                    "step": batch_step,
                    "warmup_excluded": _bool_int(batch_step < cfg.warmup_steps),
                    "nnodes": cfg.nnodes,
                    "nproc_per_node": cfg.nproc_per_node,
                    "world_size": cfg.nnodes * cfg.nproc_per_node,
                    "dist_mode": "multi_node" if cfg.nnodes > 1 else "single_node",
                }
                batch_prepare_start = time.perf_counter()
                try:
                    raw_dense_batch, sparse_batch, labels_batch = next(
                        data_iter_state["iter"]
                    )
                except StopIteration:
                    data_iter_state["iter"] = iter(dataloader)
                    raw_dense_batch, sparse_batch, labels_batch = next(
                        data_iter_state["iter"]
                    )
                dense_batch = raw_dense_batch
                row["batch_prepare_ms"] = (
                    time.perf_counter() - batch_prepare_start
                ) * 1e3

                input_pack_start = time.perf_counter()
                _, sparse_features = build_kjt_batch_from_dense_sparse_labels(
                    dense_batch,
                    sparse_batch,
                    labels_batch,
                    device=device,
                )
                row["input_pack_ms"] = (time.perf_counter() - input_pack_start) * 1e3

                if cfg.read_before_update and cfg.read_mode == "prefetch":
                    lookahead_prefetcher.enqueue(sparse_features)
                    lookahead_prefetcher.advance()

                return row, time.perf_counter(), dense_batch, sparse_features, labels_batch

            data_iter_state = {"iter": data_iter}
            for step in range(cfg.steps):
                while (
                    len(prepared_batches) <= lookahead_prefetcher.depth
                    and step + len(prepared_batches) < cfg.steps
                ):
                    prepared_batches.append(prepare_next_batch(step + len(prepared_batches)))
                if step + len(prepared_batches) >= cfg.steps:
                    lookahead_prefetcher.advance_all()

                row, step_start, dense_batch, sparse_features, labels_batch = (
                    prepared_batches.popleft()
                )

                _reset_perf_stats(embedding_module)
                _reset_perf_stats(sparse_optimizer)
                lookahead_prefetcher.reset_stats()
                sparse_optimizer.zero_grad()
                embeddings = None
                with stage_timer(row, "embed_lookup_local_ms"):
                    sync_device(torch, device)
                    if cfg.read_before_update and cfg.read_mode == "prefetch":
                        if cfg.prefetch_depth > 0:
                            lookahead_prefetcher.attach_next()
                        else:
                            embedding_module.issue_fused_prefetch(sparse_features)
                    embeddings = embedding_module(sparse_features)
                    sync_device(torch, device)
                row.update(lookahead_prefetcher.consume_stats(reset=False))
                _merge_numeric_fields(
                    row,
                    getattr(embedding_module, "_single_node_forward_profile", None),
                    FAST_PATH_LOOKUP_PROFILE_KEYS,
                )
                _merge_gpu_cache_profile(row, client, "lookup")
                if embeddings is None:
                    raise RuntimeError("recstore embedding module returned no embeddings")
                if step >= cfg.warmup_steps:
                    read_lat_us.append(row["embed_lookup_local_ms"] * 1e3)

                with stage_timer(row, "embed_pool_local_ms"):
                    embedded_sparse_source = reshape_torchrec_embeddings_for_dlrm(
                        embeddings=embeddings,
                        feature_names=default_cat_names,
                        torch=torch,
                    )
                    sync_device(torch, device)

                with stage_timer(row, "output_unpack_ms"):
                    dense_features, embedded_sparse, labels = prepare_hybrid_dlrm_input(
                        dense_batch=dense_batch,
                        embedded_sparse_source=embedded_sparse_source,
                        labels_batch=labels_batch,
                        torch=torch,
                        device=device,
                        detach_sparse=True,
                    )

                with stage_timer(row, "dense_fwd_ms"):
                    sync_device(torch, device)
                    logits = dense_module(dense_features, embedded_sparse)
                    loss = criterion(logits, labels)
                    sync_device(torch, device)

                with stage_timer(row, "backward_ms"):
                    embedded_sparse_grad = run_hybrid_backward(
                        loss=loss,
                        embedded_sparse=embedded_sparse,
                        dense_module=dense_module,
                        torch=torch,
                        device=device,
                    )

                with stage_timer(row, "optimizer_ms"):
                    sync_device(torch, device)
                    dense_optimizer.step()
                    dense_optimizer.zero_grad(set_to_none=True)
                    sync_device(torch, device)

                with stage_timer(row, "sparse_update_ms"):
                    sync_device(torch, device)
                    replay_start = time.perf_counter()
                    embedded_sparse_source.backward(
                        embedded_sparse_grad.to(embedded_sparse_source.device)
                    )
                    sync_device(torch, device)
                    row["sparse_backward_replay_ms"] = (
                        time.perf_counter() - replay_start
                    ) * 1e3

                    optimizer_step_start = time.perf_counter()
                    sparse_optimizer.step()
                    sync_device(torch, device)
                    row["sparse_optimizer_step_ms"] = (
                        time.perf_counter() - optimizer_step_start
                    ) * 1e3

                    flush_start = time.perf_counter()
                    sparse_optimizer.flush()
                    sync_device(torch, device)
                    row["sparse_optimizer_flush_ms"] = (
                        time.perf_counter() - flush_start
                    ) * 1e3

                    zero_grad_start = time.perf_counter()
                    sparse_optimizer.zero_grad()
                    row["sparse_zero_grad_ms"] = (
                        time.perf_counter() - zero_grad_start
                    ) * 1e3
                    sync_device(torch, device)
                _merge_numeric_fields(
                    row,
                    getattr(sparse_optimizer, "_last_step_profile", None),
                    FAST_PATH_UPDATE_PROFILE_KEYS,
                )
                _merge_gpu_cache_profile(row, client, "update")

                _merge_consumed_perf_stats(row, _consume_perf_stats(embedding_module))
                _merge_consumed_perf_stats(row, _consume_perf_stats(sparse_optimizer))
                if step >= cfg.warmup_steps:
                    update_lat_us.append(row["sparse_update_ms"] * 1e3)
                row["step_total_ms"] = (time.perf_counter() - step_start) * 1e3
                rows.append(finalize_recstore_row(row))
                _barrier_for_step_alignment(
                    dist=dist,
                    device=device,
                    local_rank=local_rank,
                    use_dist=use_dist,
                )

                if (step + 1) % 10 == 0:
                    print(
                        f"[rs_demo] step {step + 1}/{cfg.steps} "
                        f"emb={rows[-1]['emb_stage_ms'] if rows else 0:.2f}ms "
                        f"step={rows[-1]['step_total_ms'] if rows else 0:.2f}ms"
                    )

            print("[rs_demo] workload finished")
            print(f"[rs_demo] emb_read latency: {summarize_us(read_lat_us)}")
            print(f"[rs_demo] emb_update latency: {summarize_us(update_lat_us)}")
            _write_rows(out_csv, rows)
            if use_dist and dist.is_initialized():
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
                dist.destroy_process_group()
            return {
                "backend": "recstore",
                "read_lat_us": read_lat_us,
                "update_lat_us": update_lat_us,
                "rows": rows,
            }
        finally:
            os.chdir(str(orig_cwd))

    def run(self, repo_root: Path, cfg: RunConfig) -> dict:
        if cfg.backend != "recstore":
            raise ValueError("RecStoreRunner requires cfg.backend to be 'recstore'.")
        validate_recstore_config(cfg)

        if os.environ.get("RS_DEMO_RECSTORE_WORKER") == "1":
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(
                os.environ.get("WORLD_SIZE", str(cfg.nnodes * cfg.nproc_per_node))
            )
            worker_dir = Path(os.environ["RS_DEMO_RECSTORE_WORKER_DIR"])
            ensure_shared_dir(worker_dir)
            out_csv = worker_dir / f"rank{rank}.csv"
            return self._run_local_worker(
                repo_root=repo_root,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                out_csv=out_csv,
            )

        if cfg.nnodes * cfg.nproc_per_node <= 1:
            return self._run_single_process(repo_root, cfg)
        return self._run_distributed(repo_root, cfg)
