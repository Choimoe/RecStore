"""OptimizationConfig — nested dataclass for optimization strategy settings.

This is embedded as ``RunConfig.optimization`` so that all optimization-related
fields live in one place instead of scattering across the top-level config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class OptimizationConfig:
    """Optimization strategy configuration — nested under ``RunConfig``.

    Fields:
        plugin: Strategy name — ``"none"``, ``"bagpipe"``, ``"lookahead"``,
            or any future registered plugin (e.g. ``"fusedrec"``).
        lookahead: Prefetch depth (shared by bagpipe / lookahead).
        cleanup_proportion: BagPipe-specific — fraction of lookahead batches
            at which to evict and write back.
        cache_capacity: GPU cache capacity (number of embedding rows).
        embedding_dim: Embedding dimension.
        prescan: BagPipe-specific — run the oracle prescan over the dataset
            before training so the shared-ID set is complete (issues #278).
            Without it the fallback set only covers the first steps and every
            ID outside it must take the cross-rank sparse aggregation path.
        cuda_graphs: Capture supported DLRM dense forward/backward with fixed
            FP32 gradient storage; communication and SGD remain outside capture.
        sparse_collective: Planned BagPipe gradient transport: ``all_reduce``
            (default), or the experimental padded ``all_gather`` alternative.
        plugin_config: Extensible per-plugin config for future strategies.
    """

    plugin: str = "none"
    lookahead: int = 0
    cleanup_proportion: float = 0.25
    cache_capacity: int = 0
    embedding_dim: int = 128
    prescan: bool = False
    cuda_graphs: bool = False
    sparse_collective: str = "all_reduce"
    plugin_config: Dict[str, Any] = field(default_factory=dict)
