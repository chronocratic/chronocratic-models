# ruff: noqa: D, PLR2004, S101
"""Unit tests for normalize_per_series."""

import torch

from chronocratic.models.utils import normalize_per_series


def test_zero_mean_unit_std_per_series_and_channel() -> None:
    torch.manual_seed(0)
    x = torch.rand(4, 100, 3) * 5.0 + 2.0
    normalized = normalize_per_series(x)
    assert torch.allclose(normalized.mean(dim=1), torch.zeros(4, 3), atol=1e-5)
    assert torch.allclose(normalized.std(dim=1, unbiased=False), torch.ones(4, 3), atol=1e-4)


def test_constant_series_gives_zeros() -> None:
    normalized = normalize_per_series(torch.full((2, 50, 1), 0.7))
    assert not torch.isnan(normalized).any()
    assert torch.equal(normalized, torch.zeros_like(normalized))
