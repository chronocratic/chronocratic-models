# ruff: noqa: D, PLR2004, S101
"""Unit tests for masked_reconstruction_loss_mean."""

import torch

from chronocratic.models.utils import masked_reconstruction_loss_mean


def test_masked_loss_ignores_padding() -> None:
    """Per-element ones, keep_mask excludes half -> mean equals 1.0."""
    per_element = torch.ones(2, 10, 3)
    keep_mask = torch.ones(2, 10, dtype=torch.bool)
    keep_mask[:, 5:] = False  # exclude last 5 timesteps
    loss = masked_reconstruction_loss_mean(per_element, keep_mask)
    assert loss.ndim == 0  # scalar
    assert loss.item() == 1.0


def test_masked_loss_all_nan() -> None:
    """Per-element arbitrary, keep_mask all False -> finite result (denom clamped)."""
    per_element = torch.randn(2, 10, 3)
    keep_mask = torch.zeros(2, 10, dtype=torch.bool)
    loss = masked_reconstruction_loss_mean(per_element, keep_mask)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss).item()


def test_masked_loss_no_padding() -> None:
    """Keep_mask all True -> equals standard mean."""
    per_element = torch.randn(2, 10, 3)
    keep_mask = torch.ones(2, 10, dtype=torch.bool)
    loss = masked_reconstruction_loss_mean(per_element, keep_mask)
    expected = per_element.mean()
    torch.testing.assert_close(loss, expected)
