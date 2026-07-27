"""RankMixer model plugin for the shared rs_demo training harness.

Implements the model-plugin seam (see ``rs_demo/runtime/hybrid_dlrm.py``):
``build_dense_module`` / ``build_criterion`` / ``compute_loss`` / ``task_names``,
plus the CLI arguments it owns.  ``torch`` and the RankMixer model are imported
lazily inside the methods so ``import RankMixer.plugin`` stays cheap for argument
registration at CLI-parse time.
"""
from __future__ import annotations


def _model():
    """Import the RankMixer model module under either layout (model_zoo on-path
    for production, repo-root for tests)."""
    try:
        from RankMixer import model
    except ImportError:
        from model_zoo.RankMixer import model
    return model


def _model_arg(cfg, key, default):
    return getattr(cfg, "model_args", {}).get(key, default)


def _segment_dims(cfg, num_sparse_features, embedding_dim):
    raw = _model_arg(cfg, "rankmixer_segment_dims", "")
    if not raw:
        return _model().default_segment_dims(num_sparse_features, embedding_dim, num_segments=5)
    return [int(p) for p in str(raw).split(",") if p.strip()]


def _labels_to_multitask(labels, task_names):
    """Deterministic per-task labels from the single binary label.

    Both embedding backends receive identical labels, so forward/backward stay
    numerically comparable.  Binary tasks reuse the 0/1 label; regression (mse)
    tasks get a deterministic float target derived from it.
    """
    task_loss_cfg = _model().RankMixerLoss.TASK_LOSS_CFG
    base = labels.view(-1).float()
    task_labels = {}
    for i, task in enumerate(task_names):
        loss_type, _ = task_loss_cfg.get(task, ("logloss", 1.0))
        if loss_type == "mse":
            task_labels[task] = (base * 0.5 + 0.1 * (i % 5)).clamp(0.0, 1.0)
        else:
            task_labels[task] = base
    return task_labels


class RankMixerPlugin:
    #: CLI dests routed into ``cfg.model_args`` by ``rs_demo.config.parse_config``.
    ARG_DESTS = (
        "rankmixer_tokens_split_dim",
        "rankmixer_blocks",
        "rankmixer_gate_num",
        "rankmixer_masked_dim",
        "rankmixer_segment_dims",
    )

    def add_arguments(self, parser) -> None:
        group = parser.add_argument_group("rankmixer model")
        group.add_argument(
            "--rankmixer-tokens-split-dim", type=int, default=2400,
            help="RankMixer LT projection output dim (token dim). Production: 2400.",
        )
        group.add_argument(
            "--rankmixer-blocks", type=int, default=2,
            help="Number of TokenMixer+PFFN blocks. Production: 2.",
        )
        group.add_argument(
            "--rankmixer-gate-num", type=int, default=6,
            help="PLE expert (gate) count = 1 base + task groups. Production: 6.",
        )
        group.add_argument(
            "--rankmixer-masked-dim", type=int, default=56,
            help="Mask feature dim for PLE/MMoE gate. Production: 4*(6+8)=56.",
        )
        group.add_argument(
            "--rankmixer-segment-dims", type=str, default="",
            help="Comma-separated per-segment deep-input dims. Empty = auto-partition.",
        )

    def build_dense_module(self, cfg, *, num_sparse_features, embedding_dim, device):
        module = _model().build_rankmixer_arch(
            embedding_dim=embedding_dim,
            num_sparse_features=num_sparse_features,
            segment_dims=_segment_dims(cfg, num_sparse_features, embedding_dim),
            tokens_split_dim=_model_arg(cfg, "rankmixer_tokens_split_dim", 2400),
            rankmixer_blocks=_model_arg(cfg, "rankmixer_blocks", 2),
            gate_num=_model_arg(cfg, "rankmixer_gate_num", 6),
            masked_dim=_model_arg(cfg, "rankmixer_masked_dim", 56),
            device=device,
        )
        module.model_type = "rankmixer"
        return module

    def build_criterion(self, cfg, dense_module):
        return _model().RankMixerLoss(self.task_names(dense_module))

    def compute_loss(self, dense_module, criterion, dense_features, embedded_sparse, labels):
        logits = dense_module(embedded_sparse)
        task_labels = _labels_to_multitask(labels, list(logits.keys()))
        return criterion(logits, task_labels), logits

    def task_names(self, dense_module) -> list[str]:
        ple = getattr(dense_module, "ple", None)
        if ple is None:
            return []
        return [t for g in ple.ple_groups.values() for t in g.task_group]


PLUGIN = RankMixerPlugin()
