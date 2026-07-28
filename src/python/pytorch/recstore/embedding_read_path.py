"""Embedding lookup read strategies for training loops.

``read_mode`` axes (orthogonal to fusion layout on the embedding module):

- ``direct``: synchronous pull inside forward (accuracy baseline).
- ``prefetch``: async get with optional lookahead window ``prefetch_depth``.
  Does **not** wait for in-flight sparse updates; may observe stale values.
- ``bagpipe``: async get that stalls conflicting reads until updates land
  (same accuracy as ``direct``). Not wired yet.

Fusion on/off only affects which module APIs are used to encode ids; it must
not rewrite ``read_mode`` semantics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PrefetchSlot:
    handle: int
    num_ids: int
    issue_ts: float
    fused_ids_cpu: Any
    fused_inverse: Any
    full_batch: bool


class LookaheadPrefetcher:
    """Owns cross-step fused prefetch scheduling for ``read_mode=prefetch``."""

    def __init__(
        self,
        embedding_module: Any,
        depth: int,
        *,
        embedding_dim: int,
        value_bytes: int = 4,
    ) -> None:
        self._embedding_module = embedding_module
        self._depth = max(0, int(depth))
        self._embedding_dim = max(0, int(embedding_dim))
        self._value_bytes = max(1, int(value_bytes))
        self._pending: deque[PrefetchSlot] = deque()
        self._ready: deque[PrefetchSlot] = deque()

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def live_ids(self) -> int:
        return sum(slot.num_ids for slot in self._pending) + sum(
            slot.num_ids for slot in self._ready
        )

    @property
    def live_bytes(self) -> int:
        return int(self.live_ids) * int(self._embedding_dim) * int(self._value_bytes)

    def enqueue(self, sparse_features: Any) -> None:
        if self._depth <= 0:
            return
        result = self._embedding_module.issue_fused_prefetch(
            sparse_features,
            record_handle=False,
        )
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result
        self._pending.append(
            PrefetchSlot(
                handle=int(handle),
                num_ids=int(num_ids),
                issue_ts=float(issue_ts),
                fused_ids_cpu=fused_ids_cpu,
                fused_inverse=fused_inverse,
                full_batch=True,
            )
        )

    def enqueue_fused_ids(self, fused_ids: Any) -> None:
        if self._depth <= 0:
            return
        issue = getattr(self._embedding_module, "issue_fused_id_prefetch", None)
        if not callable(issue):
            raise RuntimeError(
                "fused-id prefetch requires issue_fused_id_prefetch()."
            )
        result = issue(fused_ids, record_handle=False)
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result
        self._pending.append(
            PrefetchSlot(
                handle=int(handle),
                num_ids=int(num_ids),
                issue_ts=float(issue_ts),
                fused_ids_cpu=fused_ids_cpu,
                fused_inverse=fused_inverse,
                full_batch=False,
            )
        )

    def advance(self) -> bool:
        if self._depth <= 0 or len(self._pending) <= self._depth:
            return False
        self._ready.append(self._pending.popleft())
        return True

    def advance_all(self) -> int:
        moved = 0
        while self._pending:
            self._ready.append(self._pending.popleft())
            moved += 1
        return moved

    def attach_next(self, *, invalid_fused_ids: Any = None) -> bool:
        if self._depth <= 0 or not self._ready:
            return False
        slot = self._ready.popleft()
        self._embedding_module.set_fused_prefetch_handle(
            slot.handle,
            num_ids=slot.num_ids,
            issue_ts=slot.issue_ts,
            fused_ids_cpu=slot.fused_ids_cpu,
            fused_inverse=slot.fused_inverse,
            invalid_fused_ids_cpu=invalid_fused_ids,
            full_batch=slot.full_batch,
        )
        return True

    def discard_next_ready(self) -> bool:
        if self._depth <= 0 or not self._ready:
            return False
        self._ready.popleft()
        return True


def prepare_fused_ids_from_sparse_batch(
    sparse_batch: Any,
    feature_offsets: Any,
) -> tuple[Any, Any, int]:
    """CPU unique fused ids from a dense sparse batch (fusion-EBC optimization)."""
    import torch

    if sparse_batch.ndim != 2 or sparse_batch.shape[1] != feature_offsets.numel():
        raise ValueError("sparse batch shape does not match feature offsets")
    fused_ids = (
        sparse_batch.to(dtype=torch.int64, device="cpu") + feature_offsets
    ).T.reshape(-1)
    unique_ids, inverse = torch.unique(fused_ids, return_inverse=True)
    return unique_ids, inverse, int(fused_ids.numel())


class EmbeddingReadPath(Protocol):
    @property
    def depth(self) -> int: ...

    @property
    def desired_buffer_size(self) -> int:
        """How many prepared batches the runner should keep in its buffer.

        The runner pre-prepares batches (dataloader + prefetch issue) up to
        this many items ahead so the read path's internal pipeline stays full.
        """

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        """Prepare-phase hook. Return a ticket consumed by ``before_lookup``."""

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        """Issue/attach async handle before ``embedding_module(...)``."""

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        """Post-update hook (window advance / future bagpipe wait)."""

    def advance_all(self) -> int:
        """Drain pending lookahead into ready (end-of-run / drain)."""


class DirectReadPath:
    """Synchronous pull inside forward; no async issue."""

    @property
    def depth(self) -> int:
        return 0

    @property
    def desired_buffer_size(self) -> int:
        return 0  # synchronous: no prefetch pipeline to fill

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        del step, sparse_features, sparse_batch, row
        return None

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        del step, sparse_features, ticket, row

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        del step, sparse_features, sparse_optimizer, row

    def advance_all(self) -> int:
        return 0


class PrefetchReadPath:
    """Async embedding read with optional lookahead window.

    Overlaps gets with later work and does **not** block on in-flight sparse
    updates, so values may be stale relative to ``direct`` / ``bagpipe``.
    """

    def __init__(
        self,
        embedding_module: Any,
        *,
        prefetch_depth: int,
        embedding_dim: int,
        feature_offsets: Any | None = None,
    ) -> None:
        if not bool(getattr(embedding_module, "_enable_fusion", False)):
            raise RuntimeError(
                "read_mode=prefetch currently requires a fusion-enabled "
                "embedding module (non-fused async APIs are not wired yet)"
            )
        if not callable(getattr(embedding_module, "issue_fused_prefetch", None)):
            raise RuntimeError(
                "read_mode=prefetch requires embedding_module.issue_fused_prefetch"
            )
        self._module = embedding_module
        self._feature_offsets = feature_offsets
        self._lookahead = LookaheadPrefetcher(
            embedding_module,
            int(prefetch_depth),
            embedding_dim=int(embedding_dim),
        )

    @property
    def depth(self) -> int:
        return self._lookahead.depth

    @property
    def desired_buffer_size(self) -> int:
        # LookaheadPrefetcher has two internal queues (pending + ready),
        # each holding up to ``depth`` items.  The runner must prepare
        # 2*depth batches to fill both.
        return self._lookahead.depth * 2

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        del step
        if self._lookahead.depth > 0:
            self._lookahead.enqueue(sparse_features)
            while self._lookahead.advance():
                pass
            return None

        # Same-step async get: optionally prebuild unique fused ids on CPU.
        if (
            sparse_batch is not None
            and self._feature_offsets is not None
            and callable(getattr(self._module, "issue_prepared_fused_prefetch", None))
        ):
            fused_id_start = time.perf_counter()
            ticket = prepare_fused_ids_from_sparse_batch(
                sparse_batch, self._feature_offsets
            )
            row["lookup_ids_build_ms"] = (time.perf_counter() - fused_id_start) * 1e3
            return ticket
        return "issue_on_lookup"

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        del step, row
        if self._lookahead.depth > 0:
            if not self._lookahead.attach_next():
                self._module.issue_fused_prefetch(sparse_features)
            return
        if ticket is not None and ticket != "issue_on_lookup":
            self._module.issue_prepared_fused_prefetch(*ticket)
            return
        self._module.issue_fused_prefetch(sparse_features)

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        del step, sparse_features, sparse_optimizer, row
        # Intentionally no stale repair — that is bagpipe's job.

    def advance_all(self) -> int:
        return self._lookahead.advance_all()


class BagPipeReadPath:
    """Placeholder for update-aware async reads (not wired yet)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "read_mode=bagpipe is not wired in recstore yet; use direct or prefetch"
        )


def build_embedding_read_path(
    read_mode: str,
    *,
    embedding_module: Any,
    prefetch_depth: int = 0,
    embedding_dim: int = 0,
    feature_offsets: Any | None = None,
) -> EmbeddingReadPath:
    mode = str(read_mode).strip().lower()
    if mode == "direct":
        return DirectReadPath()
    if mode == "prefetch":
        return PrefetchReadPath(
            embedding_module,
            prefetch_depth=prefetch_depth,
            embedding_dim=embedding_dim,
            feature_offsets=feature_offsets,
        )
    if mode == "bagpipe":
        return BagPipeReadPath()
    raise RuntimeError(
        f"unsupported read_mode={read_mode!r}; expected direct|prefetch|bagpipe"
    )
