from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefetchSlot:
    handle: int
    num_ids: int
    issue_ts: float
    fused_ids_cpu: Any
    fused_inverse: Any
    full_batch: bool


class LookaheadPrefetcher:
    """Owns lookahead prefetch scheduling."""

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
                "BagPipe fused-id prefetch requires issue_fused_id_prefetch()."
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
