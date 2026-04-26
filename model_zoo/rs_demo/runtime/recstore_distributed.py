from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ShardServer:
    shard: int
    host: str
    port: int


def _load_runtime_config(runtime_dir: Path) -> dict:
    cfg_path = runtime_dir / "recstore_config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_client_configs(cfg: dict) -> tuple[dict, dict]:
    distributed = cfg.get("distributed_client")
    cache_ps = cfg.get("cache_ps")
    return (
        distributed if isinstance(distributed, dict) else {},
        cache_ps if isinstance(cache_ps, dict) else {},
    )


def _load_shard_servers(distributed_cfg: dict, cache_cfg: dict, cfg_path: Path) -> list[ShardServer]:
    if "servers" in distributed_cfg:
        raw_servers = distributed_cfg.get("servers") or []
    else:
        raw_servers = cache_cfg.get("servers") or []
    if not raw_servers:
        raise RuntimeError(f"missing shard servers in {cfg_path}")

    servers: list[ShardServer] = [
        ShardServer(
            shard=int(server["shard"]),
            host=str(server["host"]),
            port=int(server["port"]),
        )
        for server in raw_servers
    ]
    servers.sort(key=lambda s: s.shard)
    return servers


def _load_num_shards(distributed_cfg: dict, cache_cfg: dict, servers: list[ShardServer]) -> int:
    if "num_shards" in distributed_cfg:
        return int(distributed_cfg["num_shards"])
    if "num_shards" in cache_cfg:
        return int(cache_cfg["num_shards"])
    return len(servers)


def _load_cache_servers(cache_cfg: dict) -> list[ShardServer]:
    raw_servers = cache_cfg.get("servers") or []
    return [
        ShardServer(
            shard=int(server["shard"]),
            host=str(server["host"]),
            port=int(server["port"]),
        )
        for server in raw_servers
    ]


def _load_hash_method(distributed_cfg: dict) -> str:
    return str(distributed_cfg.get("hash_method", "city_hash"))


_CITYHASH_LIB_HANDLE: ctypes.CDLL | None = None
_CITYHASH64_FUNC = None


def _get_cityhash64_func():
    global _CITYHASH_LIB_HANDLE, _CITYHASH64_FUNC
    if _CITYHASH64_FUNC is not None:
        return _CITYHASH64_FUNC
    lib_path = Path(__file__).resolve().parents[3] / "third_party/cityhash/src/.libs/libcityhash.so.0.0.0"
    try:
        _CITYHASH_LIB_HANDLE = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise RuntimeError(f"failed to load cityhash library: {lib_path}") from exc
    try:
        func = _CITYHASH_LIB_HANDLE._Z10CityHash64PKcm
    except AttributeError as exc:
        raise RuntimeError(f"missing CityHash64 symbol in library: {lib_path}") from exc
    func.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    func.restype = ctypes.c_uint64
    _CITYHASH64_FUNC = func
    return _CITYHASH64_FUNC


def _city_hash64_of_uint64(key: int) -> int:
    raw = struct.pack("<Q", int(key) & 0xFFFFFFFFFFFFFFFF)
    city_hash64 = _get_cityhash64_func()
    return int(city_hash64(raw, len(raw)))


class ShardedRecstoreClient:
    def __init__(self, client, runtime_dir: Path) -> None:
        cfg = _load_runtime_config(runtime_dir)
        distributed_cfg, cache_cfg = _load_client_configs(cfg)
        self._client = client
        self._cache_ps_type = str(cache_cfg.get("ps_type", "")).upper()
        self._cache_servers = _load_cache_servers(cache_cfg)
        self._cache_num_shards = _load_num_shards({}, cache_cfg, self._cache_servers)
        self._servers = _load_shard_servers(
            distributed_cfg, cache_cfg, runtime_dir / "recstore_config.json"
        )
        self._servers_by_shard = {server.shard: server for server in self._servers}
        self._num_shards = _load_num_shards(distributed_cfg, cache_cfg, self._servers)
        self._hash_method = _load_hash_method(distributed_cfg)
        self._active_shard: int | None = None
        self._next_prefetch_id = 1
        self._prefetch_contexts: dict[int, tuple[int, list[tuple[int, torch.Tensor, torch.Tensor]]]] = {}
        self._kv_prefetch_next_id = 1
        self._kv_prefetch_contexts: dict[int, tuple[int, list[tuple[int, torch.Tensor, torch.Tensor]]]] = {}
        self._tensor_meta: Dict[str, Dict[str, Any]] = {}
        self._next_async_handle = 1
        self._pending_async_ops: Dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}

    def register_tensor_meta(
        self,
        name: str,
        shape: tuple[int, int],
        dtype: torch.dtype,
        base_offset: int = 0,
    ) -> None:
        if name in self._tensor_meta:
            return
        num_embeddings, embedding_dim = shape
        self._tensor_meta[name] = {
            "shape": (int(num_embeddings), int(embedding_dim)),
            "dtype": dtype,
            "base_offset": int(base_offset),
        }

    def _shard_for_key(self, key: int) -> int:
        if self._hash_method == "simple_mod":
            return int(key) % self._num_shards
        return _city_hash64_of_uint64(int(key)) % self._num_shards

    def _activate_shard(self, shard: int) -> None:
        shard = int(shard)
        if self._active_shard == shard:
            return
        if self.is_shared_local_shm_table():
            self._active_shard = shard
            return
        server = self._servers_by_shard.get(shard)
        if server is None:
            raise RuntimeError(f"no server configured for shard {shard}")
        self._client.ops.set_ps_config(server.host, server.port)
        self._active_shard = shard

    def _group_indices(self, keys: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
        shard_to_indices: dict[int, list[int]] = {}
        shard_order: list[int] = []
        for idx, key in enumerate(keys.tolist()):
            shard = self._shard_for_key(int(key))
            if shard not in shard_to_indices:
                shard_to_indices[shard] = []
                shard_order.append(shard)
            shard_to_indices[shard].append(idx)
        return [
            (shard, torch.tensor(indices, dtype=torch.long))
            for shard, indices in ((shard, shard_to_indices[shard]) for shard in shard_order)
        ]

    def _normalize_ids(self, ids: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        normalized_ids = ids.to(dtype=torch.int64)
        if not normalized_ids.is_contiguous():
            normalized_ids = normalized_ids.contiguous()
        if not keep_device and normalized_ids.device.type != "cpu":
            normalized_ids = normalized_ids.to("cpu")
        return normalized_ids

    def _normalize_grads(self, grads: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
        if not isinstance(grads, torch.Tensor):
            raise TypeError("grads must be a torch.Tensor")
        normalized_grads = grads.to(dtype=torch.float32)
        if not normalized_grads.is_contiguous():
            normalized_grads = normalized_grads.contiguous()
        if not keep_device and normalized_grads.device.type != "cpu":
            normalized_grads = normalized_grads.to("cpu")
        return normalized_grads

    def _require_active_shard(self, api_name: str) -> None:
        if self._active_shard is None:
            raise RuntimeError(f"{api_name} requires an active shard to be selected first.")

    def _require_local_shm_backend(self, api_name: str) -> None:
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "current_ps_backend"):
            raise RuntimeError(
                f"{api_name} requires a RecStore ops library exposing current_ps_backend()."
            )
        backend = self._client.ops.current_ps_backend()
        if backend != "local_shm":
            raise RuntimeError(
                f"{api_name} requires local_shm backend, but current backend is {backend}."
            )

    def _group_ids_by_shard(self, keys: torch.Tensor) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        ids_cpu = self._normalize_ids(keys)
        shard_to_indices: dict[int, list[int]] = {}
        shard_order: list[int] = []
        for idx, key in enumerate(ids_cpu.tolist()):
            shard = self._shard_for_key(int(key))
            if shard not in shard_to_indices:
                shard_to_indices[shard] = []
                shard_order.append(shard)
            shard_to_indices[shard].append(idx)
        grouped: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for shard in shard_order:
            indices = shard_to_indices[shard]
            index_tensor = torch.tensor(indices, dtype=torch.long)
            shard_keys = ids_cpu.index_select(0, index_tensor)
            if not shard_keys.is_contiguous():
                shard_keys = shard_keys.contiguous()
            grouped.append((shard, index_tensor, shard_keys))
        return grouped

    def init_embedding_table(self, table_name: str, num_embeddings: int, embedding_dim: int) -> bool:
        ok = True
        for server in self._servers:
            self._activate_shard(server.shard)
            ok = self._client.init_embedding_table(table_name, num_embeddings, embedding_dim) and ok
        return ok

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_vals = values.index_select(0, index_tensor).contiguous()
            self._client.emb_write(shard_keys, shard_vals)

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        if keys.numel() == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        out = torch.empty((keys.shape[0], embedding_dim), dtype=torch.float32)
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_values = self._client.emb_read(shard_keys, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        req_id = self._next_prefetch_id
        self._next_prefetch_id += 1
        if keys.numel() == 0:
            self._prefetch_contexts[req_id] = (0, [])
            return req_id

        shard_requests: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for shard, index_tensor in self._group_indices(keys):
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_requests.append((shard, index_tensor, shard_keys))
        self._prefetch_contexts[req_id] = (int(keys.shape[0]), shard_requests)
        return req_id

    def emb_wait_result(self, prefetch_id: int, embedding_dim: int) -> torch.Tensor:
        context = self._prefetch_contexts.pop(int(prefetch_id), None)
        if context is None:
            raise RuntimeError(f"unknown prefetch_id: {prefetch_id}")
        total_rows, shard_requests = context
        if total_rows == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32)
        out = torch.empty((total_rows, embedding_dim), dtype=torch.float32)
        for shard, index_tensor, shard_keys in shard_requests:
            self._activate_shard(shard)
            shard_prefetch_id = int(self._client.emb_prefetch(shard_keys))
            shard_values = self._client.emb_wait_result(shard_prefetch_id, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        return out

    def emb_read_prefetch(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        prefetch_id = self.emb_prefetch(keys)
        return self.emb_wait_result(prefetch_id, embedding_dim)

    def current_ps_backend(self) -> str:
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "current_ps_backend"):
            raise RuntimeError(
                "current_ps_backend requires a RecStore ops library exposing current_ps_backend()."
            )
        return str(self._client.ops.current_ps_backend())

    def activate_shard(self, shard: int) -> None:
        self._activate_shard(int(shard))

    def set_ps_backend(self, backend: str) -> None:
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        if not hasattr(self._client, "ops") or not hasattr(self._client.ops, "set_ps_backend"):
            raise RuntimeError(
                "set_ps_backend requires a RecStore ops library exposing set_ps_backend()."
            )
        self._client.ops.set_ps_backend(backend)

    def is_shared_local_shm_table(self) -> bool:
        if self._cache_ps_type != "LOCAL_SHM":
            return False
        if self._cache_num_shards != 1:
            return False
        if len(self._cache_servers) != 1:
            return False
        return True

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_active_shard("local_lookup_flat")
        self._require_local_shm_backend("local_lookup_flat")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        return self._client.ops.local_lookup_flat(normalized_ids, embedding_dim)

    def local_update_flat(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> None:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self._require_active_shard("local_update_flat")
        self._require_local_shm_backend("local_update_flat")
        normalized_ids = self._normalize_ids(ids, keep_device=True)
        normalized_grads = self._normalize_grads(grads, keep_device=True)
        if normalized_grads.dim() != 2:
            raise ValueError("grads must be a 2-dimensional tensor")
        if normalized_ids.size(0) != normalized_grads.size(0):
            raise ValueError("ids and grads must have the same number of rows")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        if normalized_grads.size(1) != embedding_dim:
            raise ValueError(
                f"grads second dimension must match embedding dim {embedding_dim} for tensor '{name}'"
            )
        self._client.ops.local_update_flat(name, normalized_ids, normalized_grads)

    def emb_update_table(self, table_name: str, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        for shard, index_tensor in self._group_indices(keys):
            self._activate_shard(shard)
            shard_keys = keys.index_select(0, index_tensor).contiguous()
            shard_grads = grads.index_select(0, index_tensor).contiguous()
            self._client.emb_update_table(table_name, shard_keys, shard_grads)

    def init_data(
        self,
        name: str,
        shape: tuple[int, int],
        dtype: torch.dtype,
        part_policy: Any = None,
        init_func: Any | None = None,
        is_gdata: bool = True,
        base_offset: int = 0,
    ) -> None:
        if name in self._tensor_meta:
            return
        num_embeddings, embedding_dim = shape
        self.register_tensor_meta(name=name, shape=shape, dtype=dtype, base_offset=base_offset)
        self.init_embedding_table(name, num_embeddings, embedding_dim)

        if init_func:
            initial_data = init_func(shape, dtype)
        else:
            initial_data = torch.zeros(shape, dtype=dtype)
        if not isinstance(initial_data, torch.Tensor):
            initial_data = torch.tensor(initial_data, dtype=dtype)
        initial_data = initial_data.to(dtype=dtype)
        if initial_data.device.type != "cpu":
            initial_data = initial_data.to("cpu")
        if not initial_data.is_contiguous():
            initial_data = initial_data.contiguous()

        keys = torch.arange(num_embeddings, dtype=torch.int64)
        if base_offset != 0:
            keys = keys + int(base_offset)

        for shard, index_tensor, shard_keys in self._group_ids_by_shard(keys):
            self._activate_shard(shard)
            shard_values = initial_data.index_select(0, index_tensor).contiguous()
            self._client.emb_write(shard_keys, shard_values)

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        return self.emb_read(ids, embedding_dim)

    def prefetch(self, ids: torch.Tensor) -> int:
        normalized_ids = self._normalize_ids(ids)
        req_id = self._kv_prefetch_next_id
        self._kv_prefetch_next_id += 1
        if normalized_ids.numel() == 0:
            self._kv_prefetch_contexts[req_id] = (0, [])
            return req_id
        shard_requests: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(normalized_ids):
            shard_requests.append((shard, index_tensor, shard_keys))
        self._kv_prefetch_contexts[req_id] = (int(normalized_ids.numel()), shard_requests)
        return req_id

    def wait_and_get(
        self,
        prefetch_id: int,
        embedding_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        context = self._kv_prefetch_contexts.pop(int(prefetch_id), None)
        if context is None:
            raise RuntimeError(f"unknown prefetch_id: {prefetch_id}")
        total_rows, shard_requests = context
        if total_rows == 0:
            return torch.empty((0, embedding_dim), dtype=torch.float32, device=device)
        out = torch.empty((total_rows, embedding_dim), dtype=torch.float32)
        for shard, index_tensor, shard_keys in shard_requests:
            self._activate_shard(shard)
            shard_prefetch_id = int(self._client.emb_prefetch(shard_keys))
            shard_values = self._client.emb_wait_result(shard_prefetch_id, embedding_dim)
            out.index_copy_(0, index_tensor, shard_values)
        if device.type == "cuda":
            out = out.to(device)
        return out

    def update_async(self, name: str, ids: torch.Tensor, grads: torch.Tensor) -> int:
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor {name} has not been initialized.")
        normalized_ids = self._normalize_ids(ids)
        normalized_grads = self._normalize_grads(grads)
        if normalized_ids.size(0) != normalized_grads.size(0):
            raise RuntimeError("ids and grads must have the same number of rows for update_async")
        handle = self._next_async_handle
        self._next_async_handle += 1
        self._pending_async_ops[handle] = (name, normalized_ids, normalized_grads)
        return handle

    def wait(self, handle: int) -> None:
        pending = self._pending_async_ops.pop(int(handle), None)
        if pending is None:
            return
        name, ids, grads = pending
        if ids.numel() == 0:
            return
        for shard, index_tensor, shard_keys in self._group_ids_by_shard(ids):
            shard_grads = grads.index_select(0, index_tensor).contiguous()
            self._activate_shard(shard)
            self._client.emb_update_table(name, shard_keys, shard_grads)
