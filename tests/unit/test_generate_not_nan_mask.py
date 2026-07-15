# ruff: noqa: D, PLR2004, S101
"""Unit tests for generate_not_nan_mask."""

import torch

from chronocratic.models.utils import generate_not_nan_mask


def test_generate_not_nan_mask_basic() -> None:
    """Trailing NaN timesteps -> mask True for real, False for NaN."""
    x = torch.randn(2, 10, 3)  # (B=2, T=10, C=3)
    x[:, 8:, :] = float("nan")  # last 2 timesteps NaN
    mask = generate_not_nan_mask(x)
    assert mask.shape == (2, 10)
    assert mask.dtype == torch.bool
    assert mask[:, :8].all()
    assert not mask[:, 8:].any()


def test_generate_not_nan_mask_all_nan() -> None:
    """Fully-NaN batch -> mask all False."""
    x = torch.full((2, 5, 3), float("nan"))
    mask = generate_not_nan_mask(x)
    assert mask.shape == (2, 5)
    assert not mask.any()


def test_generate_not_nan_mask_no_nan() -> None:
    """Clean input -> mask all True."""
    x = torch.randn(2, 10, 3)
    mask = generate_not_nan_mask(x)
    assert mask.shape == (2, 10)
    assert mask.all()


def test_generate_not_nan_mask_partial_channel() -> None:
    """Only some channels NaN -> timestep marked False."""
    x = torch.randn(2, 10, 3)
    x[0, 5, 0] = float("nan")  # only channel 0 of timestep 5 is NaN
    mask = generate_not_nan_mask(x)
    assert mask[0, 5].item() is False
    assert mask[1, 5].item() is True  # other sample unaffected
