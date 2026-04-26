from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import torch


_CPU_EXCHANGE_GROUP: Any | None = None


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    return tensor.clone()


def _require_int64_vector(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1-D tensor, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.int64:
        raise ValueError(f"{name} must use torch.int64, got {tensor.dtype}")
    return tensor


def _require_float32_matrix(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2-D tensor, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32, got {tensor.dtype}")
    return tensor


@dataclass(frozen=True)
class LookupIdsPayload:
    rank: int
    destination_ranks: torch.Tensor
    source_ranks: torch.Tensor
    row_positions: torch.Tensor
    fused_ids: torch.Tensor

    def __post_init__(self) -> None:
        _require_int64_vector("destination_ranks", self.destination_ranks)
        _require_int64_vector("source_ranks", self.source_ranks)
        _require_int64_vector("row_positions", self.row_positions)
        _require_int64_vector("fused_ids", self.fused_ids)
        _validate_parallel_lengths(
            "lookup id payload",
            self.destination_ranks,
            self.source_ranks,
            self.row_positions,
            self.fused_ids,
        )

    def clone(self) -> "LookupIdsPayload":
        return LookupIdsPayload(
            rank=int(self.rank),
            destination_ranks=_clone_tensor(self.destination_ranks),
            source_ranks=_clone_tensor(self.source_ranks),
            row_positions=_clone_tensor(self.row_positions),
            fused_ids=_clone_tensor(self.fused_ids),
        )


@dataclass(frozen=True)
class LookupEmbeddingResponsePayload:
    rank: int
    requestor_ranks: torch.Tensor
    row_positions: torch.Tensor
    embeddings: torch.Tensor

    def __post_init__(self) -> None:
        _require_int64_vector("requestor_ranks", self.requestor_ranks)
        _require_int64_vector("row_positions", self.row_positions)
        _require_float32_matrix("embeddings", self.embeddings)
        _validate_parallel_lengths(
            "lookup embedding response payload",
            self.requestor_ranks,
            self.row_positions,
            self.embeddings,
        )

    def clone(self) -> "LookupEmbeddingResponsePayload":
        return LookupEmbeddingResponsePayload(
            rank=int(self.rank),
            requestor_ranks=_clone_tensor(self.requestor_ranks),
            row_positions=_clone_tensor(self.row_positions),
            embeddings=_clone_tensor(self.embeddings),
        )


@dataclass(frozen=True)
class SparseGradPayload:
    rank: int
    destination_ranks: torch.Tensor
    source_ranks: torch.Tensor
    row_positions: torch.Tensor
    fused_ids: torch.Tensor
    grads: torch.Tensor

    def __post_init__(self) -> None:
        _require_int64_vector("destination_ranks", self.destination_ranks)
        _require_int64_vector("source_ranks", self.source_ranks)
        _require_int64_vector("row_positions", self.row_positions)
        _require_int64_vector("fused_ids", self.fused_ids)
        _require_float32_matrix("grads", self.grads)
        _validate_parallel_lengths(
            "sparse grad payload",
            self.destination_ranks,
            self.source_ranks,
            self.row_positions,
            self.fused_ids,
            self.grads,
        )

    def clone(self) -> "SparseGradPayload":
        return SparseGradPayload(
            rank=int(self.rank),
            destination_ranks=_clone_tensor(self.destination_ranks),
            source_ranks=_clone_tensor(self.source_ranks),
            row_positions=_clone_tensor(self.row_positions),
            fused_ids=_clone_tensor(self.fused_ids),
            grads=_clone_tensor(self.grads),
        )


def _validate_parallel_lengths(label: str, *tensors: torch.Tensor) -> None:
    if not tensors:
        return
    expected_rows = int(tensors[0].shape[0])
    for tensor in tensors[1:]:
        if int(tensor.shape[0]) != expected_rows:
            raise ValueError(
                f"{label} fields must have the same leading dimension, "
                f"got {expected_rows} and {int(tensor.shape[0])}"
            )


PayloadT = TypeVar(
    "PayloadT",
    LookupIdsPayload,
    LookupEmbeddingResponsePayload,
    SparseGradPayload,
)


def _resolve_exchange_backend(backend: Any) -> tuple[Any, Any | None, torch.device]:
    get_backend = getattr(backend, "get_backend", None)
    backend_name = ""
    if callable(get_backend):
        resolved = get_backend()
        backend_name = str(resolved)
    if backend_name.lower() == "nccl":
        new_group = getattr(backend, "new_group", None)
        if callable(new_group):
            global _CPU_EXCHANGE_GROUP
            if _CPU_EXCHANGE_GROUP is None:
                _CPU_EXCHANGE_GROUP = new_group(backend="gloo")
            return backend, _CPU_EXCHANGE_GROUP, torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL exchange requires CUDA to be available")
        return backend, None, torch.device("cuda", torch.cuda.current_device())
    return backend, None, torch.device("cpu")


def _resolve_all_to_all_backend(backend: Any) -> tuple[Any, Any | None, torch.device]:
    get_backend = getattr(backend, "get_backend", None)
    backend_name = ""
    if callable(get_backend):
        backend_name = str(get_backend())
    if backend_name.lower() == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL exchange requires CUDA to be available")
        return backend, None, torch.device("cuda", torch.cuda.current_device())
    return _resolve_exchange_backend(backend)


def _backend_all_gather(
    backend: Any,
    group: Any | None,
    output_tensors: list[torch.Tensor],
    input_tensor: torch.Tensor,
) -> None:
    if group is None:
        backend.all_gather(output_tensors, input_tensor)
    else:
        backend.all_gather(output_tensors, input_tensor, group=group)


def _backend_all_to_all_single(
    backend: Any,
    group: Any | None,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
) -> None:
    if group is None:
        backend.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )
    else:
        backend.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )


def _gather_metadata(
    *,
    rank: int,
    rows: int,
    cols: int,
    world_size: int,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> list[tuple[int, int, int]]:
    local_meta = torch.tensor([rank, rows, cols], dtype=torch.int64, device=device)
    gathered_meta = [torch.empty_like(local_meta) for _ in range(world_size)]
    _backend_all_gather(backend, group, gathered_meta, local_meta)
    return [
        (
            int(meta[0].item()),
            int(meta[1].item()),
            int(meta[2].item()),
        )
        for meta in gathered_meta
    ]


def _gather_padded_vector(
    tensor: torch.Tensor,
    *,
    max_rows: int,
    world_size: int,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> list[torch.Tensor]:
    local = tensor.detach().to(device=device, dtype=torch.int64)
    padded = torch.empty((max_rows,), dtype=torch.int64, device=device)
    padded.zero_()
    if local.numel() > 0:
        padded[: local.numel()] = local
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    _backend_all_gather(backend, group, gathered, padded)
    return [item.cpu() for item in gathered]


def _gather_padded_matrix(
    tensor: torch.Tensor,
    *,
    max_rows: int,
    max_cols: int,
    world_size: int,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> list[torch.Tensor]:
    local = tensor.detach().to(device=device, dtype=torch.float32)
    padded = torch.empty((max_rows, max_cols), dtype=torch.float32, device=device)
    padded.zero_()
    if local.numel() > 0:
        rows = int(local.shape[0])
        cols = int(local.shape[1])
        padded[:rows, :cols] = local
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    _backend_all_gather(backend, group, gathered, padded)
    return [item.cpu() for item in gathered]


def _gather_lookup_ids_tensor(
    payload: LookupIdsPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupIdsPayload]:
    exchange_backend, exchange_group, device = _resolve_exchange_backend(backend)
    metadata = _gather_metadata(
        rank=int(payload.rank),
        rows=int(payload.fused_ids.numel()),
        cols=0,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    max_rows = max(rows for _, rows, _ in metadata)
    destination_ranks = _gather_padded_vector(
        payload.destination_ranks,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    source_ranks = _gather_padded_vector(
        payload.source_ranks,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    row_positions = _gather_padded_vector(
        payload.row_positions,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    fused_ids = _gather_padded_vector(
        payload.fused_ids,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[LookupIdsPayload] = []
    for idx, (rank, rows, _) in enumerate(metadata):
        gathered.append(
            LookupIdsPayload(
                rank=rank,
                destination_ranks=destination_ranks[idx][:rows].clone(),
                source_ranks=source_ranks[idx][:rows].clone(),
                row_positions=row_positions[idx][:rows].clone(),
                fused_ids=fused_ids[idx][:rows].clone(),
            )
        )
    return gathered


def _gather_lookup_embedding_responses_tensor(
    payload: LookupEmbeddingResponsePayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupEmbeddingResponsePayload]:
    exchange_backend, exchange_group, device = _resolve_exchange_backend(backend)
    metadata = _gather_metadata(
        rank=int(payload.rank),
        rows=int(payload.requestor_ranks.numel()),
        cols=int(payload.embeddings.shape[1]),
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    max_rows = max(rows for _, rows, _ in metadata)
    max_cols = max(cols for _, _, cols in metadata)
    nonzero_cols = {cols for _, rows, cols in metadata if rows > 0}
    if len(nonzero_cols) > 1:
        raise ValueError("lookup embedding response payloads must share the same embedding dimension")
    requestor_ranks = _gather_padded_vector(
        payload.requestor_ranks,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    row_positions = _gather_padded_vector(
        payload.row_positions,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    embeddings = _gather_padded_matrix(
        payload.embeddings,
        max_rows=max_rows,
        max_cols=max_cols,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[LookupEmbeddingResponsePayload] = []
    for idx, (rank, rows, cols) in enumerate(metadata):
        gathered.append(
            LookupEmbeddingResponsePayload(
                rank=rank,
                requestor_ranks=requestor_ranks[idx][:rows].clone(),
                row_positions=row_positions[idx][:rows].clone(),
                embeddings=embeddings[idx][:rows, :cols].clone(),
            )
        )
    return gathered


def _gather_sparse_grads_tensor(
    payload: SparseGradPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[SparseGradPayload]:
    exchange_backend, exchange_group, device = _resolve_exchange_backend(backend)
    metadata = _gather_metadata(
        rank=int(payload.rank),
        rows=int(payload.fused_ids.numel()),
        cols=int(payload.grads.shape[1]),
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    max_rows = max(rows for _, rows, _ in metadata)
    max_cols = max(cols for _, _, cols in metadata)
    nonzero_cols = {cols for _, rows, cols in metadata if rows > 0}
    if len(nonzero_cols) > 1:
        raise ValueError("sparse grad payloads must share the same embedding dimension")
    destination_ranks = _gather_padded_vector(
        payload.destination_ranks,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    source_ranks = _gather_padded_vector(
        payload.source_ranks,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    row_positions = _gather_padded_vector(
        payload.row_positions,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    fused_ids = _gather_padded_vector(
        payload.fused_ids,
        max_rows=max_rows,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    grads = _gather_padded_matrix(
        payload.grads,
        max_rows=max_rows,
        max_cols=max_cols,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[SparseGradPayload] = []
    for idx, (rank, rows, cols) in enumerate(metadata):
        gathered.append(
            SparseGradPayload(
                rank=rank,
                destination_ranks=destination_ranks[idx][:rows].clone(),
                source_ranks=source_ranks[idx][:rows].clone(),
                row_positions=row_positions[idx][:rows].clone(),
                fused_ids=fused_ids[idx][:rows].clone(),
                grads=grads[idx][:rows, :cols].clone(),
            )
        )
    return gathered


def _exchange_counts(
    send_counts: list[int],
    *,
    world_size: int,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> list[int]:
    input_counts = torch.tensor(send_counts, dtype=torch.int64, device=device)
    output_counts = torch.empty_like(input_counts)
    _backend_all_to_all_single(
        backend,
        group,
        output_counts,
        input_counts,
        output_split_sizes=[1] * world_size,
        input_split_sizes=[1] * world_size,
    )
    return [int(v) for v in output_counts.cpu().tolist()]


def _group_row_indices_by_destination(
    destination_ranks: torch.Tensor,
    *,
    world_size: int,
) -> tuple[list[int], torch.Tensor]:
    valid_mask = (destination_ranks >= 0) & (destination_ranks < int(world_size))
    valid_destinations = destination_ranks[valid_mask]
    send_counts_tensor = torch.bincount(valid_destinations, minlength=world_size)
    send_counts = [int(v) for v in send_counts_tensor.cpu().tolist()[:world_size]]
    if valid_destinations.numel() == 0:
        return send_counts, torch.empty((0,), dtype=torch.int64, device=destination_ranks.device)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    permutation = torch.argsort(valid_destinations, stable=True)
    grouped_indices = valid_positions.index_select(0, permutation)
    return send_counts, grouped_indices


def _permute_vector_rows(tensor: torch.Tensor, grouped_indices: torch.Tensor) -> torch.Tensor:
    if grouped_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=tensor.device)
    return tensor.index_select(0, grouped_indices.to(device=tensor.device)).to(torch.int64).contiguous()


def _permute_matrix_rows(tensor: torch.Tensor, grouped_indices: torch.Tensor) -> torch.Tensor:
    if grouped_indices.numel() == 0:
        return torch.empty((0, int(tensor.shape[1])), dtype=torch.float32, device=tensor.device)
    return tensor.index_select(0, grouped_indices.to(device=tensor.device)).to(torch.float32).contiguous()


def _permute_int_matrix_rows(tensor: torch.Tensor, grouped_indices: torch.Tensor) -> torch.Tensor:
    if grouped_indices.numel() == 0:
        return torch.empty((0, int(tensor.shape[1])), dtype=torch.int64, device=tensor.device)
    return tensor.index_select(0, grouped_indices.to(device=tensor.device)).to(torch.int64).contiguous()


def _build_exchange_plan(
    *,
    destination_ranks: torch.Tensor,
    world_size: int,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> tuple[list[int], list[int], torch.Tensor]:
    send_counts, grouped_indices = _group_row_indices_by_destination(
        destination_ranks,
        world_size=world_size,
    )
    recv_counts = _exchange_counts(
        send_counts,
        world_size=world_size,
        backend=backend,
        group=group,
        device=device,
    )
    return send_counts, recv_counts, grouped_indices


def _exchange_vector_by_plan(
    tensor: torch.Tensor,
    *,
    send_counts: list[int],
    recv_counts: list[int],
    grouped_indices: torch.Tensor,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> torch.Tensor:
    send_tensor = _permute_vector_rows(tensor, grouped_indices).to(device=device, dtype=torch.int64)
    recv_tensor = torch.empty((sum(recv_counts),), dtype=torch.int64, device=device)
    _backend_all_to_all_single(
        backend,
        group,
        recv_tensor,
        send_tensor,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return recv_tensor


def _exchange_matrix_by_plan(
    tensor: torch.Tensor,
    *,
    send_counts: list[int],
    recv_counts: list[int],
    grouped_indices: torch.Tensor,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> torch.Tensor:
    cols = int(tensor.shape[1])
    send_tensor = _permute_matrix_rows(tensor, grouped_indices).to(device=device, dtype=torch.float32)
    recv_tensor = torch.empty((sum(recv_counts), cols), dtype=torch.float32, device=device)
    _backend_all_to_all_single(
        backend,
        group,
        recv_tensor.view(-1),
        send_tensor.view(-1),
        output_split_sizes=[count * cols for count in recv_counts],
        input_split_sizes=[count * cols for count in send_counts],
    )
    return recv_tensor


def _exchange_int_matrix_by_plan(
    tensor: torch.Tensor,
    *,
    send_counts: list[int],
    recv_counts: list[int],
    grouped_indices: torch.Tensor,
    backend: Any,
    group: Any | None,
    device: torch.device,
) -> torch.Tensor:
    cols = int(tensor.shape[1])
    send_tensor = _permute_int_matrix_rows(tensor, grouped_indices).to(device=device, dtype=torch.int64)
    recv_tensor = torch.empty((sum(recv_counts), cols), dtype=torch.int64, device=device)
    _backend_all_to_all_single(
        backend,
        group,
        recv_tensor.view(-1),
        send_tensor.view(-1),
        output_split_sizes=[count * cols for count in recv_counts],
        input_split_sizes=[count * cols for count in send_counts],
    )
    return recv_tensor


def _slice_offsets(counts: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for count in counts:
        end = start + int(count)
        offsets.append((start, end))
        start = end
    return offsets


def _exchange_lookup_ids_all_to_all(
    payload: LookupIdsPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupIdsPayload]:
    exchange_backend, exchange_group, device = _resolve_all_to_all_backend(backend)
    send_counts, recv_counts, grouped_indices = _build_exchange_plan(
        destination_ranks=payload.destination_ranks,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    packed_int = torch.stack(
        [
            payload.source_ranks.to(torch.int64),
            payload.row_positions.to(torch.int64),
            payload.fused_ids.to(torch.int64),
        ],
        dim=1,
    )
    received_int = _exchange_int_matrix_by_plan(
        packed_int,
        send_counts=send_counts,
        recv_counts=recv_counts,
        grouped_indices=grouped_indices,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[LookupIdsPayload] = []
    for source_rank, (start, end) in enumerate(_slice_offsets(recv_counts)):
        rows = end - start
        gathered.append(
            LookupIdsPayload(
                rank=source_rank,
                destination_ranks=torch.full(
                    (rows,),
                    int(payload.rank),
                    dtype=torch.int64,
                    device=received_int.device,
                ),
                source_ranks=received_int[start:end, 0].clone(),
                row_positions=received_int[start:end, 1].clone(),
                fused_ids=received_int[start:end, 2].clone(),
            )
        )
    return gathered


def _exchange_lookup_embedding_responses_all_to_all(
    payload: LookupEmbeddingResponsePayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupEmbeddingResponsePayload]:
    exchange_backend, exchange_group, device = _resolve_all_to_all_backend(backend)
    send_counts, recv_counts, grouped_indices = _build_exchange_plan(
        destination_ranks=payload.requestor_ranks,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    packed_int = torch.stack(
        [
            payload.requestor_ranks.to(torch.int64),
            payload.row_positions.to(torch.int64),
        ],
        dim=1,
    )
    received_int = _exchange_int_matrix_by_plan(
        packed_int,
        send_counts=send_counts,
        recv_counts=recv_counts,
        grouped_indices=grouped_indices,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    embeddings = _exchange_matrix_by_plan(
        payload.embeddings,
        send_counts=send_counts,
        recv_counts=recv_counts,
        grouped_indices=grouped_indices,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[LookupEmbeddingResponsePayload] = []
    for source_rank, (start, end) in enumerate(_slice_offsets(recv_counts)):
        gathered.append(
            LookupEmbeddingResponsePayload(
                rank=source_rank,
                requestor_ranks=received_int[start:end, 0].clone(),
                row_positions=received_int[start:end, 1].clone(),
                embeddings=embeddings[start:end].clone(),
            )
        )
    return gathered


def _exchange_sparse_grads_all_to_all(
    payload: SparseGradPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[SparseGradPayload]:
    exchange_backend, exchange_group, device = _resolve_all_to_all_backend(backend)
    send_counts, recv_counts, grouped_indices = _build_exchange_plan(
        destination_ranks=payload.destination_ranks,
        world_size=world_size,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    packed_int = torch.stack(
        [
            payload.destination_ranks.to(torch.int64),
            payload.source_ranks.to(torch.int64),
            payload.row_positions.to(torch.int64),
            payload.fused_ids.to(torch.int64),
        ],
        dim=1,
    )
    received_int = _exchange_int_matrix_by_plan(
        packed_int,
        send_counts=send_counts,
        recv_counts=recv_counts,
        grouped_indices=grouped_indices,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    grads = _exchange_matrix_by_plan(
        payload.grads,
        send_counts=send_counts,
        recv_counts=recv_counts,
        grouped_indices=grouped_indices,
        backend=exchange_backend,
        group=exchange_group,
        device=device,
    )
    gathered: list[SparseGradPayload] = []
    for source_rank, (start, end) in enumerate(_slice_offsets(recv_counts)):
        gathered.append(
            SparseGradPayload(
                rank=source_rank,
                destination_ranks=received_int[start:end, 0].clone(),
                source_ranks=received_int[start:end, 1].clone(),
                row_positions=received_int[start:end, 2].clone(),
                fused_ids=received_int[start:end, 3].clone(),
                grads=grads[start:end].clone(),
            )
        )
    return gathered


def _gather_payloads(
    payload: PayloadT,
    *,
    world_size: int,
    backend: Any,
) -> list[PayloadT]:
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    if world_size == 1:
        return [payload.clone()]
    if backend is None:
        raise ValueError("backend is required when world_size > 1")
    if hasattr(backend, "all_to_all_single"):
        if isinstance(payload, LookupIdsPayload):
            return _exchange_lookup_ids_all_to_all(payload, world_size=world_size, backend=backend)
        if isinstance(payload, LookupEmbeddingResponsePayload):
            return _exchange_lookup_embedding_responses_all_to_all(
                payload,
                world_size=world_size,
                backend=backend,
            )
        if isinstance(payload, SparseGradPayload):
            return _exchange_sparse_grads_all_to_all(payload, world_size=world_size, backend=backend)
        raise TypeError(f"unsupported payload type: {type(payload).__name__}")
    if hasattr(backend, "all_gather"):
        if isinstance(payload, LookupIdsPayload):
            gathered = _gather_lookup_ids_tensor(payload, world_size=world_size, backend=backend)
        elif isinstance(payload, LookupEmbeddingResponsePayload):
            gathered = _gather_lookup_embedding_responses_tensor(
                payload,
                world_size=world_size,
                backend=backend,
            )
        elif isinstance(payload, SparseGradPayload):
            gathered = _gather_sparse_grads_tensor(payload, world_size=world_size, backend=backend)
        else:
            raise TypeError(f"unsupported payload type: {type(payload).__name__}")
        for idx, gathered_payload in enumerate(gathered):
            if int(gathered_payload.rank) != idx:
                raise RuntimeError(
                    "gathered payload rank metadata mismatch: "
                    f"slot {idx} contains payload.rank={int(gathered_payload.rank)}"
                )
        return gathered
    gather_buffer: list[PayloadT | None] = [None] * world_size
    backend.all_gather_object(gather_buffer, payload.clone())
    gathered: list[PayloadT] = []
    for idx, gathered_payload in enumerate(gather_buffer):
        if gathered_payload is None:
            raise RuntimeError(f"missing gathered payload at rank slot {idx}")
        if not isinstance(gathered_payload, type(payload)):
            raise TypeError(
                "gathered payload type mismatch: "
                f"expected {type(payload).__name__}, got {type(gathered_payload).__name__}"
            )
        if int(gathered_payload.rank) != idx:
            raise RuntimeError(
                "gathered payload rank metadata mismatch: "
                f"slot {idx} contains payload.rank={int(gathered_payload.rank)}"
            )
        gathered.append(gathered_payload)
    return gathered


def exchange_lookup_ids(
    payload: LookupIdsPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupIdsPayload]:
    return _gather_payloads(payload, world_size=world_size, backend=backend)


def exchange_lookup_embedding_responses(
    payload: LookupEmbeddingResponsePayload,
    *,
    world_size: int,
    backend: Any,
) -> list[LookupEmbeddingResponsePayload]:
    return _gather_payloads(payload, world_size=world_size, backend=backend)


def exchange_sparse_grads(
    payload: SparseGradPayload,
    *,
    world_size: int,
    backend: Any,
) -> list[SparseGradPayload]:
    return _gather_payloads(payload, world_size=world_size, backend=backend)


def reassemble_lookup_embedding_responses(
    responses: list[LookupEmbeddingResponsePayload],
    *,
    requestor_rank: int,
    total_rows: int,
) -> torch.Tensor:
    if not isinstance(total_rows, int) or isinstance(total_rows, bool) or total_rows < 0:
        raise ValueError(f"total_rows must be a non-negative integer, got {total_rows!r}")
    filtered = [
        payload
        for payload in responses
        if torch.any(payload.requestor_ranks == int(requestor_rank))
    ]
    if total_rows == 0:
        if not filtered:
            return torch.empty((0, 0), dtype=torch.float32)
        embedding_dim = int(filtered[0].embeddings.shape[1])
        return torch.empty(
            (0, embedding_dim),
            dtype=filtered[0].embeddings.dtype,
            device=filtered[0].embeddings.device,
        )
    if not filtered:
        raise ValueError(f"no lookup embedding responses found for requestor_rank={requestor_rank}")

    embedding_dim = int(filtered[0].embeddings.shape[1])
    rebuilt = torch.empty(
        (total_rows, embedding_dim),
        dtype=filtered[0].embeddings.dtype,
        device=filtered[0].embeddings.device,
    )
    seen_positions = torch.zeros((total_rows,), dtype=torch.bool, device=rebuilt.device)

    for payload in filtered:
        if int(payload.embeddings.shape[1]) != embedding_dim:
            raise ValueError("embedding response payloads must share the same embedding dimension")
        if (
            payload.embeddings.device != rebuilt.device
            or payload.requestor_ranks.device != rebuilt.device
            or payload.row_positions.device != rebuilt.device
        ):
            payload = LookupEmbeddingResponsePayload(
                rank=payload.rank,
                requestor_ranks=payload.requestor_ranks.to(rebuilt.device),
                row_positions=payload.row_positions.to(rebuilt.device),
                embeddings=payload.embeddings.to(rebuilt.device),
            )
        if payload.embeddings.dtype != rebuilt.dtype:
            raise ValueError("embedding response payloads must share the same dtype")
        requestor_mask = payload.requestor_ranks == int(requestor_rank)
        selected_positions = payload.row_positions[requestor_mask].to(dtype=torch.int64)
        selected_embeddings = payload.embeddings[requestor_mask]
        if selected_positions.numel() == 0:
            continue
        out_of_bounds = torch.logical_or(selected_positions < 0, selected_positions >= total_rows)
        if bool(out_of_bounds.any().item()):
            bad_position = int(selected_positions[out_of_bounds][0].item())
            raise ValueError(
                f"row position {bad_position} is out of bounds for total_rows={total_rows}"
            )
        sorted_positions, _ = torch.sort(selected_positions)
        duplicate_mask = sorted_positions[1:] == sorted_positions[:-1]
        if bool(duplicate_mask.any().item()):
            duplicate = int(sorted_positions[1:][duplicate_mask][0].item())
            raise ValueError(f"duplicate row position detected during reassembly: {duplicate}")
        previously_seen = seen_positions.index_select(0, selected_positions)
        if bool(previously_seen.any().item()):
            duplicate = int(selected_positions[previously_seen][0].item())
            raise ValueError(f"duplicate row position detected during reassembly: {duplicate}")
        rebuilt.index_copy_(0, selected_positions, selected_embeddings)
        seen_positions.index_fill_(0, selected_positions, True)

    if not bool(seen_positions.all().item()):
        missing_positions = (
            torch.nonzero(~seen_positions, as_tuple=False).view(-1).cpu().tolist()
        )
        raise ValueError(
            "lookup embedding responses did not cover every requested row position; "
            f"missing positions: {missing_positions}"
        )
    return rebuilt
