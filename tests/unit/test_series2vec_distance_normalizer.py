"""Tests for Series2Vec _distance_normalizer gradient safety at batch_size=1.

Verifies that _distance_normalizer returns a detached tensor (preserving original
values) in edge-case branches rather than torch.zeros_like(), which breaks the
gradient path at batch_size=1. The normalizer produces loss weights, not
representations — gradient through the normalizer itself is not needed. The
critical gradient path is through encode() → representations.
"""

import pytest
import torch

from chronocratic.models.convolutional.standard.series2vec.losses import (
    _distance_normalizer,
    pretraining_loss,
)
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


class TestDistanceNormalizerGradientFlow:
    """Test _distance_normalizer edge-case branches and gradient safety."""

    def test_numel_le_1_returns_detached_tensor(self) -> None:
        """1-element input with requires_grad=True should return detached output
        preserving original values, not zeros."""
        distance = torch.tensor([5.0], requires_grad=True)
        result = _distance_normalizer(distance)

        # Should not crash
        assert result is not None
        # Output must be detached (no gradient through normalizer needed)
        assert not result.requires_grad
        # Output should preserve original values (detach, not zeros)
        assert torch.allclose(result, torch.tensor([5.0]))

    def test_empty_tensor_returns_detached_tensor(self) -> None:
        """Empty input should return detached empty output, not crash."""
        distance = torch.tensor([], requires_grad=True)
        result = _distance_normalizer(distance)

        assert result.numel() == 0
        assert not result.requires_grad

    def test_zero_variance_returns_detached_tensor(self) -> None:
        """Equal-valued input with requires_grad=True should return detached output
        preserving original values, not zeros."""
        distance = torch.tensor([3.0, 3.0, 3.0], requires_grad=True)
        result = _distance_normalizer(distance)

        # Should not crash
        assert result is not None
        # Output must be detached
        assert not result.requires_grad
        # Output should preserve original values (detach, not zeros)
        assert torch.allclose(result, distance.detach())

    def test_normal_min_max_normalization(self) -> None:
        """Distinct values should produce standard min-max normalization."""
        distance = torch.tensor([1.0, 3.0, 5.0])
        result = _distance_normalizer(distance)

        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

    def test_non_gradient_input(self) -> None:
        """Input without requires_grad should not crash."""
        distance = torch.tensor([4.0])  # requires_grad=False by default
        result = _distance_normalizer(distance)

        assert result is not None
        assert not result.requires_grad
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_single_zero_value(self) -> None:
        """Single zero value should return detached zero, not crash."""
        distance = torch.tensor([0.0], requires_grad=True)
        result = _distance_normalizer(distance)

        assert not result.requires_grad
        assert torch.allclose(result, torch.tensor([0.0]))


class TestEncodeGradientAtBatchSize1:
    """Verify that model.encode() produces gradients through representations
    at batch_size=1 (the critical path, not the normalizer)."""

    def test_encode_gradient_flows_at_batch_size_1(self) -> None:
        """model.encode() at batch_size=1 should produce non-zero finite gradients."""
        model = Series2Vec(input_dims=1, embedding_dims=8, representation_dims=16)
        model.train()

        x = torch.randn(1, 32, 1, requires_grad=True)
        out = model.encode_batch(x)

        # Verify output shape
        assert out.shape[0] == 1  # batch_size=1 preserved

        # Verify gradients flow through encode()
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient — encode() path is broken at batch_size=1"
        assert x.grad.numel() == x.numel()
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN/Inf"

    def test_encode_gradient_flows_at_batch_size_2(self) -> None:
        """Baseline: model.encode() at batch_size=2 should produce gradients."""
        model = Series2Vec(input_dims=1, embedding_dims=8, representation_dims=16)
        model.train()

        x = torch.randn(2, 32, 1, requires_grad=True)
        out = model.encode_batch(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.numel() == x.numel()
        assert torch.isfinite(x.grad).all()
