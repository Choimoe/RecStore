
import pytest
import torch

from recstore.gradient_compression import select_topk_rows


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
