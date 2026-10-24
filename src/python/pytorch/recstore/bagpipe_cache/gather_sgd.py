"""Optional exact FP32 gather aggregation and SGD for two resident replicas."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=(4, 5, 6))
    def _gather_sgd(values, gathered, active, sources, union_rows, gather_width, alpha,
                    dimension: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        rows, columns = offsets // dimension, offsets % dimension
        valid = rows < union_rows
        resident = tl.load(active + rows, valid, other=0)
        first = tl.load(sources + rows, valid, other=-1)
        second = tl.load(sources + union_rows + rows, valid, other=-1)
        g0 = tl.load(gathered + first * dimension + columns, valid & (first >= 0), other=0)
        g1 = tl.load(gathered + (gather_width + second) * dimension + columns,
                     valid & (second >= 0), other=0)
        old = tl.load(values + resident * dimension + columns, valid, other=0)
        # Fixed rank order and separate multiply/add match the eager index_add_ path.
        updated = old + (g0 + g1) * alpha
        tl.store(values + resident * dimension + columns, updated, valid)


def gather_sgd_(values, gathered, active_rows, sources, gather_width, alpha):
    """Return False when unsupported, otherwise enqueue one in-place SGD kernel."""
    if triton is None or values.device.type != "cuda":
        return False
    union_rows, dimension = active_rows.numel(), values.shape[1]
    if (
        values.dtype != torch.float32 or gathered.dtype != torch.float32
        or active_rows.dtype != torch.int64 or sources.dtype != torch.int64
        or sources.shape != (2, union_rows) or gathered.shape != (2 * gather_width, dimension)
        or any(t.device != values.device or not t.is_contiguous()
               for t in (values, gathered, active_rows, sources))
    ):
        raise ValueError("Two-rank gather SGD requires contiguous FP32 rows and int64 maps on one device")
    if union_rows:
        _gather_sgd[(triton.cdiv(union_rows * dimension, 256),)](
            values, gathered, active_rows, sources, union_rows, gather_width, alpha,
            dimension, 256, num_warps=4, enable_fp_fusion=False,
        )
    return True
