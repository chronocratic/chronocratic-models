# ruff: noqa: D, PLR2004, S101
"""Unit tests for zero_fill_padding."""

import torch

from chronocratic.models.utils import zero_fill_padding


def test_zero_fill_returns_finite() -> None:
    """Input with trailing NaN timesteps -> x_filled has no NaN."""
    x = torch.randn(2, 10, 3)  # (B=2, T=10, C=3)
    x[:, 8:, :] = float("nan")  # last 2 timesteps NaN
    x_filled, keep_mask = zero_fill_padding(x)
    assert x_filled.shape == x.shape
    assert torch.isfinite(x_filled).all()


def test_zero_fill_correct_keep_mask() -> None:
    """Same input -> keep_mask True for real timesteps, False for NaN."""
    x = torch.randn(2, 10, 3)
    x[:, 8:, :] = float("nan")
    _, keep_mask = zero_fill_padding(x)
    assert keep_mask.shape == (2, 10)
    assert keep_mask[:, :8].all()
    assert not keep_mask[:, 8:].any()


def test_zero_fill_all_nan() -> None:
    """Fully-NaN batch -> keep_mask all False, x_filled finite."""
    x = torch.full((2, 5, 3), float("nan"))
    x_filled, keep_mask = zero_fill_padding(x)
    assert keep_mask.shape == (2, 5)
    assert not keep_mask.any()
    assert torch.isfinite(x_filled).all()


def test_zero_fill_no_nan() -> None:
    """Clean input -> keep_mask all True, x_filled equals original."""
    x = torch.randn(2, 10, 3)
    x_filled, keep_mask = zero_fill_padding(x)
    assert keep_mask.all()
    torch.testing.assert_close(x_filled, x)


def test_zero_fill_partial_trailing() -> None:
    """Last 3 timesteps NaN -> keep_mask True for first T-3, False for last 3."""
    x = torch.randn(1, 12, 4)
    x[:, 9:, :] = float("nan")  # last 3 timesteps NaN
    _, keep_mask = zero_fill_padding(x)
    assert keep_mask.shape == (1, 12)
    assert keep_mask[:, :9].all()
    assert not keep_mask[:, 9:].any()
