"""Data structures and work-handle objects for BagPipe cache controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class PrefetchSlot:
    """A batched prefetch in flight."""
    handle: int
    ids_cpu: torch.Tensor       # unique fused IDs prefetched (sorted, CPU)
    ttl: int                    # expiry batch; every ID in this slot shares it
    issue_ts: float
    num_ids: int


class _CompositeWork:
    """Work handle for sparse all_gather + aggregate across GPUs.

    The per-rank payload is sentinel-padded (rows beyond a rank's true count
    carry id ``-1``), so ``wait()`` recovers each rank's contribution with a
    GPU-side boolean mask — no per-rank ``.item()`` count sync.  ``wait()``
    runs at the next prefill barrier where synchronization is expected; the
    only remaining host sync of the sparse path is the sizing ``.item()`` in
    :meth:`BagPipeCommMixin._all_gather_sparse_async`.

    ``padded`` keeps the launch-time padded input buffers referenced until
    ``wait()`` returns: the all_gather reads them on the side stream, and a
    host-side free before that stream passes the collective would let the
    caching allocator hand the block to a concurrent main-stream allocation.
    """

    _SENTINEL = -1

    def __init__(self, work_ids, work_grads, ids_list, grads_list,
                 world_size, dim, device, stats, padded=()):
        self._work_ids = work_ids
        self._work_grads = work_grads
        self._ids_list = ids_list
        self._grads_list = grads_list
        self._world_size = world_size
        self._dim = dim
        self._device = device
        self._stats = stats
        self._padded = tuple(padded)
        self._result = None

    def wait(self):
        self._work_ids.wait()
        self._work_grads.wait()
        all_ids = []
        all_grads = []
        for r in range(self._world_size):
            ids_r = self._ids_list[r]
            valid = ids_r != self._SENTINEL
            all_ids.append(ids_r[valid])
            all_grads.append(self._grads_list[r][valid])
        cat_ids = torch.cat(all_ids)
        cat_grads = torch.cat(all_grads)
        if cat_ids.numel() == 0:
            self._result = (
                torch.tensor([], dtype=torch.int64, device=self._device),
                torch.zeros(0, self._dim, device=self._device),
            )
            return
        unique_ids, inverse = torch.unique(cat_ids, return_inverse=True)
        aggregated = torch.zeros(
            (unique_ids.numel(), self._dim), dtype=cat_grads.dtype, device=cat_grads.device)
        aggregated.index_add_(0, inverse, cat_grads)
        self._result = (unique_ids, aggregated)

    @property
    def result(self):
        return self._result


class _DenseWork:
    """Work handle for dense all_reduce that extracts reduced grads for our IDs."""

    def __init__(self, work, dense_grads, valid_indices, ids, dim, device):
        self._work = work
        self._dense_grads = dense_grads
        self._valid_indices = valid_indices
        self._ids = ids
        self._dim = dim
        self._device = device
        self._result = None

    def wait(self):
        self._work.wait()
        reduced = self._dense_grads.index_select(0, self._valid_indices)
        self._result = (self._ids.to(self._device), reduced)

    @property
    def result(self):
        return self._result
