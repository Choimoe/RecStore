import math

import torch


def select_topk_rows(
    ids: torch.Tensor, grads: torch.Tensor, keep_ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep ceil(N * keep_ratio) rows by squared L2 energy.

    Callers aggregate duplicate IDs before selection. Selected rows retain
    their dtype and device and are ordered by descending energy; ties have
    no guaranteed order. A ratio of 1 returns the original tensors unchanged.
    """
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    if grads.ndim != 2:
        raise ValueError("grads must be a 2D tensor")
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1]")
    if ids.size(0) != grads.size(0):
        raise ValueError("IDs and gradients row count mismatch")
    if keep_ratio == 1.0 or grads.size(0) == 0:
        return ids, grads

    # Promote before squaring: FP16 can overflow even for finite gradients.
    energy_grads = grads.float() if grads.dtype in (torch.float16, torch.bfloat16) else grads
    energies = torch.sum(energy_grads.square(), dim=1)
    k = math.ceil(grads.size(0) * keep_ratio)
    topk_indices = torch.topk(energies, k=k).indices
    return ids[topk_indices], grads[topk_indices]
