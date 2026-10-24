"""Contiguous FP32 dense gradients for planned sparse communication."""

import torch
import torch.distributed as dist


class PlannedDenseGradientBuffer:
    """Use DDP.no_sync() around both forward and backward with this buffer."""

    def __init__(self, module):
        self.params = [param for param in module.parameters() if param.requires_grad]
        if not self.params or any(
            p.dtype != torch.float32 or not p.is_contiguous()
            or p.device != self.params[0].device for p in self.params
        ):
            raise ValueError("Expected nonempty contiguous FP32 dense parameters on one device")
        self.flat = torch.empty(
            sum(p.numel() for p in self.params),
            device=self.params[0].device, dtype=torch.float32,
        )
        self.views = [chunk.view_as(param) for param, chunk in zip(
            self.params, self.flat.split([p.numel() for p in self.params])
        )]
        self.seen = set()
        self.hooks = [param.register_hook(self._mark(index)) for index, param in enumerate(self.params)]

    def _mark(self, index):
        def mark(grad):
            if grad.is_sparse:
                raise RuntimeError("Sparse dense-module gradients are unsupported")
            self.seen.add(index)
        return mark

    @torch.no_grad()
    def begin_step(self):
        self.flat.zero_()
        self.seen.clear()
        for param, view in zip(self.params, self.views):
            param.grad = view

    @torch.no_grad()
    def reduce(self):
        if len(self.seen) != len(self.params):
            raise RuntimeError("Every trainable dense parameter must participate in this backward")
        if any(p.grad is not view for p, view in zip(self.params, self.views)):
            raise RuntimeError("Backward replaced a preallocated dense gradient view")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return
        # The caller submits sparse communication during backward, then this
        # dense reduction on the same group. DDP must not submit other buckets.
        # Match DDP's division before SUM, avoiding unnecessary SUM overflow.
        self.flat.div_(world_size)
        work = dist.all_reduce(self.flat, async_op=True)
        work.wait()

    def close(self):
        for hook in self.hooks:
            hook.remove()
