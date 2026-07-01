"""Tests for TST output-aware _encode_batch with VECTOR/SEQUENCE.

Verifies TST (Time Series Transformer) correctly handles the EncodingOutputShape
parameter: VECTOR returns (B, hidden_dims) via mean-pool, SEQUENCE returns
(B, seq_len, hidden_dims) natively. Gradient flow and supported_outputs are
also tested.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.transformer.tst.model import TST


@pytest.fixture
def tst_model() -> TST:
    """Create a small TST model for testing."""
    return TST(
        input_dims=3, sequence_length=16, hidden_dims=8, num_heads=2, depth=1, feedforward_dims=32
    )


class TestTSTEncodeBatchShapes:
    """TST._encode_batch returns correct shapes for VECTOR and SEQUENCE."""

    def test_encode_batch_vector_returns_2d(self, tst_model: TST) -> None:
        """VECTOR produces (B, hidden_dims) via mean(dim=1)."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(4, 16, 3)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.VECTOR)
        assert result.ndim == 2
        assert result.shape == (4, 8)  # (B, hidden_dims)

    def test_encode_batch_sequence_returns_3d(self, tst_model: TST) -> None:
        """SEQUENCE produces (B, seq_len, hidden_dims) natively."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(4, 16, 3)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3
        assert result.shape == (4, 16, 8)  # (B, seq_len, hidden_dims)

    def test_encode_batch_default_is_vector(self, tst_model: TST) -> None:
        """Default output produces VECTOR shape."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(4, 16, 3)
        result_default = tst_model._encode_batch(encoder, batch_x)
        result_vector = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.VECTOR)
        assert result_default.shape == result_vector.shape


class TestTSTMeanPoolReduction:
    """VECTOR reduction uses mean-pool (not last-step or max-pool)."""

    def test_vector_is_mean_of_sequence(self, tst_model: TST) -> None:
        """VECTOR output equals mean(sequence, dim=1)."""
        encoder = tst_model._get_encoder()
        # Eval mode ensures deterministic output (no dropout) for comparison
        encoder.eval()
        batch_x = torch.randn(2, 16, 3)
        sequence = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.SEQUENCE)
        vector = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.VECTOR)
        expected = sequence.mean(dim=1)
        assert torch.allclose(vector, expected)


class TestTSTGradientFlow:
    """Gradients flow through both VECTOR and SEQUENCE paths."""

    def test_gradient_flows_through_vector(self, tst_model: TST) -> None:
        """Backprop works through VECTOR path."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(2, 16, 3, requires_grad=True)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.VECTOR)
        loss = result.pow(2).sum()
        loss.backward()
        assert batch_x.grad is not None
        assert torch.isfinite(batch_x.grad).all()

    def test_gradient_flows_through_sequence(self, tst_model: TST) -> None:
        """Backprop works through SEQUENCE path."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(2, 16, 3, requires_grad=True)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.SEQUENCE)
        loss = result.pow(2).sum()
        loss.backward()
        assert batch_x.grad is not None
        assert torch.isfinite(batch_x.grad).all()

    def test_no_detach_in_vector_path(self, tst_model: TST) -> None:
        """VECTOR result requires grad when input requires grad."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(2, 16, 3, requires_grad=True)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.VECTOR)
        assert result.requires_grad

    def test_no_detach_in_sequence_path(self, tst_model: TST) -> None:
        """SEQUENCE result requires grad when input requires grad."""
        encoder = tst_model._get_encoder()
        batch_x = torch.randn(2, 16, 3, requires_grad=True)
        result = tst_model._encode_batch(encoder, batch_x, output=EncodingOutputShape.SEQUENCE)
        assert result.requires_grad


class TestTSTSupportedOutputs:
    """TST.supported_outputs declares VECTOR and SEQUENCE capability."""

    def test_supported_outputs_exists(self, tst_model: TST) -> None:
        """supported_outputs is a class attribute."""
        assert hasattr(TST, "supported_outputs")

    def test_supported_outputs_is_frozenset(self, tst_model: TST) -> None:
        """supported_outputs is a frozenset."""
        assert isinstance(TST.supported_outputs, frozenset)

    def test_supported_outputs_includes_vector(self, tst_model: TST) -> None:
        """VECTOR is in supported_outputs."""
        assert EncodingOutputShape.VECTOR in TST.supported_outputs

    def test_supported_outputs_includes_sequence(self, tst_model: TST) -> None:
        """SEQUENCE is in supported_outputs."""
        assert EncodingOutputShape.SEQUENCE in TST.supported_outputs


class TestTSTEncodeBatchIntegration:
    """Integration tests via encode_batch() and encode()."""

    def test_encode_batch_vector(self, tst_model: TST) -> None:
        """encode_batch() with VECTOR returns (B, hidden_dims)."""
        batch_x = torch.randn(3, 16, 3)
        result = tst_model.encode_batch(batch_x, output=EncodingOutputShape.VECTOR)
        assert result.shape == (3, 8)

    def test_encode_batch_sequence(self, tst_model: TST) -> None:
        """encode_batch() with SEQUENCE returns (B, seq_len, hidden_dims)."""
        batch_x = torch.randn(3, 16, 3)
        result = tst_model.encode_batch(batch_x, output=EncodingOutputShape.SEQUENCE)
        assert result.shape == (3, 16, 8)

    def test_encode_vector(self, tst_model: TST) -> None:
        """encode() with VECTOR returns (N, hidden_dims)."""
        data = torch.randn(5, 16, 3)
        result = tst_model.encode(data, batch_size=2, output=EncodingOutputShape.VECTOR)
        assert result.shape == (5, 8)

    def test_encode_sequence(self, tst_model: TST) -> None:
        """encode() with SEQUENCE returns (N, seq_len, hidden_dims)."""
        data = torch.randn(5, 16, 3)
        result = tst_model.encode(data, batch_size=2, output=EncodingOutputShape.SEQUENCE)
        assert result.shape == (5, 16, 8)
