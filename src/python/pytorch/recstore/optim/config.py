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
        cleanup_proportion: Legacy cache cleanup setting; the bounded PS
            runtime evicts clean rows on admission.
        cache_capacity: GPU cache capacity in sets of 64 embedding rows.
        embedding_dim: Embedding dimension.
        prescan: Precompute CPU access plans; otherwise plan online.
        aggregation: ``ps`` aggregates on PS; ``collective`` aggregates on
            trainers before a unique per-row PS submission.
        plugin_config: Extensible per-plugin config for future strategies.
    """

    plugin: str = "none"
    lookahead: int = 0
    cleanup_proportion: float = 0.25
    cache_capacity: int = 0
    embedding_dim: int = 128
    prescan: bool = False
    aggregation: str = "ps"
    plugin_config: Dict[str, Any] = field(default_factory=dict)
