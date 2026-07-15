"""Tests for encoder short-sequence handling.

Verifies auto-clamp behavior for Series2Vec DisjoinEncoder when the
input sequence is shorter than the convolutional kernel chain requires.

Requirements: D-15 (Series2Vec short-sequence auto-clamp + forward)
"""

import warnings

import pytest
import torch

from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


# --------------------------------------------------------------------------- #
# Series2Vec short-sequence tests
# --------------------------------------------------------------------------- #


class TestSeries2VecShortSequenceAutoClamp:
    """DisjoinEncoder auto-clamps temporal_kernel_size for short sequences (D-15)."""

    def test_series2vec_short_sequence_auto_clamp(self) -> None:
        """seq_len=8, input_dim=2 -> encoder clamps temporal_kernel_size.

        With temporal_kernel_size=8 and representation_kernel_size=3,
        MIN_SAFE_T = 8 + 3 - 1 = 10.  seq_len=8 < 10, so clamp to
        max(1, 8 - 3 + 1) = 6.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = Series2Vec(
                input_dim=2,
                embedding_dim=8,
                representation_dim=16,
                temporal_kernel_size=8,
                sequence_length=8,
                num_heads=2,
                feedforward_dim=32,
            )

        # Should have emitted a UserWarning about clamping
        clamp_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(clamp_warnings) >= 1, (
            f"Expected UserWarning about kernel clamping, got {len(clamp_warnings)} warnings"
        )

        # Verify the clamped value
        assert model.network.embed_layer.temporal_kernel_size == 6, (
            f"Expected clamped temporal_kernel_size=6, got "
            f"{model.network.embed_layer.temporal_kernel_size}"
        )

    def test_series2vec_long_sequence_no_clamp(self) -> None:
        """seq_len=32 should not trigger clamping."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = Series2Vec(
                input_dim=2,
                embedding_dim=8,
                representation_dim=16,
                temporal_kernel_size=8,
                sequence_length=32,
                num_heads=2,
                feedforward_dim=32,
            )

        clamp_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(clamp_warnings) == 0, (
            f"Expected no warnings for long sequence, got {len(clamp_warnings)}"
        )
        assert model.network.embed_layer.temporal_kernel_size == 8


class TestSeries2VecForwardOnPenDigits:
    """Forward pass on PenDigits-shaped input (D-15)."""

    def test_series2vec_forward_on_pen_digits(self) -> None:
        """forward((batch, 8, 2)) succeeds with correct output shape."""
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            representation_dim=16,
            temporal_kernel_size=8,
            sequence_length=8,
            num_heads=2,
            feedforward_dim=32,
        )
        model.eval()
        x = torch.randn(2, 8, 2)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 16), f"Expected (2, 16), got {output.shape}"

    def test_series2vec_gradients_on_short_seq(self) -> None:
        """backward produces non-zero finite gradients on short sequences (D-15)."""
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            representation_dim=16,
            temporal_kernel_size=8,
            sequence_length=8,
            num_heads=2,
            feedforward_dim=32,
        )
        model.train()
        x = torch.randn(2, 8, 2, requires_grad=True)
        output = model(x)
        output.sum().backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"
