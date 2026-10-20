
import pytest
import torch

from recstore.gradient_compression import select_topk_rows

import recstore.gradient_compression as compression


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_topk_low_precision_energy_uses_float32(dtype):
    ids = torch.tensor([10, 20, 30])
    grads = torch.tensor([[300.0], [600.0], [1.0]], dtype=dtype)

    selected_ids, selected_grads = select_topk_rows(ids, grads, keep_ratio=0.25)

    torch.testing.assert_close(selected_ids, torch.tensor([20]))
    torch.testing.assert_close(selected_grads, grads[1:2])
    assert selected_grads.dtype == dtype


def test_topk_preserves_float64_energy_precision():
    ids = torch.tensor([10, 20])
    grads = torch.tensor([[1.0], [1.0 + 1e-10]], dtype=torch.float64)

    selected_ids, selected_grads = select_topk_rows(ids, grads, keep_ratio=0.5)

    torch.testing.assert_close(selected_ids, torch.tensor([20]))
    torch.testing.assert_close(selected_grads, grads[1:])


def test_topk_selects_highest_energy_rows():
    """Select the K rows with the highest gradient energy."""
    ids = torch.tensor([10, 20, 30, 40])
    grads = torch.tensor([
        [3.0, 4.0],
        [0.1, 0.0],
        [0.0, 2.0],
        [0.3, 0.4],
    ])

    selected_ids, selected_grads = select_topk_rows(
        ids, grads, keep_ratio=0.5
    )

    torch.testing.assert_close(
        selected_ids,
        torch.tensor([10, 30])
    )
    torch.testing.assert_close(
        selected_grads,
        torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    )


def test_topk_keep_ratio_one():
    """A ratio of one preserves the original tensors and order."""
    ids = torch.tensor([10, 20, 30])
    grads = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    selected_ids, selected_grads = select_topk_rows(
        ids, grads, keep_ratio=1.0
    )

    torch.testing.assert_close(selected_ids, ids)
    torch.testing.assert_close(selected_grads, grads)


def test_topk_empty_input():
    """Empty gradients retain their shape."""
    ids = torch.empty((0,), dtype=torch.int64)
    grads = torch.empty((0, 128), dtype=torch.float32)

    selected_ids, selected_grads = select_topk_rows(
        ids, grads, keep_ratio=0.25
    )

    assert selected_ids.shape == (0,)
    assert selected_grads.shape == (0, 128)


def test_topk_rejects_mismatched_rows():
    """Reject mismatched ID and gradient row counts."""
    ids = torch.tensor([10, 20, 30])
    grads = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    # Shape validation also applies when compression is disabled.
    with pytest.raises(ValueError):
        select_topk_rows(ids, grads, keep_ratio=1.0)


def test_topk_rejects_invalid_ratio():
    """Reject invalid keep ratios."""
    ids = torch.tensor([10, 20])
    grads = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    for ratio in [0.0, -0.1, 1.5, float("nan")]:
        with pytest.raises(ValueError):
            select_topk_rows(ids, grads, keep_ratio=ratio)


def test_topk_rejects_invalid_dimensions():
    """Validate input tensor dimensions."""
    ids = torch.tensor([10, 20])
    grads = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    # IDs must be one-dimensional.
    with pytest.raises(ValueError):
        select_topk_rows(
            ids.unsqueeze(1), grads, keep_ratio=0.5
        )

    # Gradients must be two-dimensional.
    with pytest.raises(ValueError):
        select_topk_rows(
            ids, grads.flatten(), keep_ratio=0.5
        )

def test_warmup_topk_cpu(monkeypatch):
    """CPU warmup uses the expected tensor shapes and dtypes."""
    calls = []
    original_select = compression.select_topk_rows

    def record_select(ids, grads, keep_ratio):
        calls.append((
            tuple(ids.shape),
            ids.dtype,
            tuple(grads.shape),
            grads.dtype,
            ids.device.type,
            keep_ratio,
        ))
        return original_select(ids, grads, keep_ratio)

    monkeypatch.setattr(
        compression, "select_topk_rows", record_select
    )

    def forbid_cuda_sync(*args, **kwargs):
        pytest.fail("CPU warmup must not synchronize CUDA")

    monkeypatch.setattr(
        torch.cuda, "synchronize", forbid_cuda_sync
    )

    result = compression.warmup_topk(
        device=torch.device("cpu"),
        embedding_dim=4,
        num_rows=8,
        keep_ratio=0.25,
    )

    assert result is None
    assert calls == [
        ((8,), torch.int64, (8, 4), torch.float32, "cpu", 0.25)
    ]


def test_warmup_topk_disabled(monkeypatch):
    """Disabled warmup neither allocates tensors nor invokes Top-K."""

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled warmup must not execute Top-K")

    monkeypatch.setattr(
        compression, "select_topk_rows", forbidden
    )
    monkeypatch.setattr(torch, "arange", forbidden)

    result = compression.warmup_topk(
        device="cpu",
        embedding_dim=4,
        num_rows=8,
        keep_ratio=1.0,
    )

    assert result is None


@pytest.mark.parametrize(
    "embedding_dim,num_rows,keep_ratio",
    [
        (0, 8, 0.25),
        (-1, 8, 0.25),
        (4, 0, 0.25),
        (4, -1, 0.25),
        (4, 2.5, 0.25),
        (True, 8, 0.25),
        (4, 8, 0.0),
        (4, 8, -0.1),
        (4, 8, 1.5),
        (4, 8, float("nan")),
        (0, 8, 1.0),
    ],
)
def test_warmup_topk_invalid_parameters(
    embedding_dim, num_rows, keep_ratio
):
    """Reject invalid parameters even when compression is disabled."""

    with pytest.raises(ValueError):
        compression.warmup_topk(
            device="cpu",
            embedding_dim=embedding_dim,
            num_rows=num_rows,
            keep_ratio=keep_ratio,
        )
