# ruff: noqa: D, PLR2004, S101
"""Unit tests for masked_reconstruction_loss_sum."""

import torch

from chronocratic.models.utils import masked_reconstruction_loss_sum


def test_masked_sum_valid_elements() -> None:
    """Ones with half masked → sum equals count of valid elements."""
    per_element = torch.ones(2, 10, 3)
    keep_mask = torch.ones(2, 10, dtype=torch.bool)
    keep_mask[:, 5:] = False  # exclude last 5 timesteps
    loss = masked_reconstruction_loss_sum(per_element, keep_mask)
    # 2 batches × 5 valid × 3 channels = 30
    assert loss.item() == 30.0


def test_masked_sum_all_valid() -> None:
    """All True mask → sum equals full tensor sum."""
    per_element = torch.arange(60, dtype=torch.float32).reshape(2, 10, 3)
    keep_mask = torch.ones(2, 10, dtype=torch.bool)
    loss = masked_reconstruction_loss_sum(per_element, keep_mask)
    torch.testing.assert_close(loss, per_element.sum())


def test_masked_sum_all_nan() -> None:
    """All False mask → returns 0.0."""
    per_element = torch.randn(2, 10, 3)
    keep_mask = torch.zeros(2, 10, dtype=torch.bool)
    loss = masked_reconstruction_loss_sum(per_element, keep_mask)
    assert loss.item() == 0.0


def test_masked_sum_partial_values() -> None:
    """Non-uniform values → only masked-in elements contribute."""
    per_element = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    # [0,1,2,3] [4,5,6,7] [8,9,10,11]
    keep_mask = torch.tensor([[True, False, True, False]])
    loss = masked_reconstruction_loss_sum(per_element, keep_mask)
    # Column 0: 0+1+2 = 3, Column 2: 8+9+10 = 27 → 30
    expected = per_element[:, 0, :].sum() + per_element[:, 2, :].sum()
    torch.testing.assert_close(loss, expected)
