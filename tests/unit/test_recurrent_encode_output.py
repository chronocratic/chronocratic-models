"""Tests for TimeNet and RecurrentAE output-aware _encode_batch.

Verifies TimeNet and RecurrentAE honor the `output` parameter in `_encode_batch`
to return both VECTOR (2-D) and SEQUENCE (3-D) representations.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.model import TimeNet


# ---------------------------------------------------------------------------
# TimeNet
# ---------------------------------------------------------------------------


class TestTimeNetSupportedOutputs:
    """TimeNet.supported_outputs includes VECTOR and SEQUENCE."""

    def test_has_vector(self) -> None:
        """VECTOR is in supported_outputs."""
        assert EncodingOutputShape.VECTOR in TimeNet.supported_outputs

    def test_has_sequence(self) -> None:
        """SEQUENCE is in supported_outputs."""
        assert EncodingOutputShape.SEQUENCE in TimeNet.supported_outputs

    def test_is_frozenset(self) -> None:
        """supported_outputs is a frozenset."""
        assert isinstance(TimeNet.supported_outputs, frozenset)


class TestTimeNetVectorOutput:
    """TimeNet._encode_batch with VECTOR returns (B, H)."""

    def test_vector_shape(self) -> None:
        """VECTOR produces 2-D tensor via last-step slice."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(4, 20, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        assert result.shape == (4, 16)

    def test_vector_ndim(self) -> None:
        """VECTOR output is exactly 2-D."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        assert result.ndim == 2


class TestTimeNetSequenceOutput:
    """TimeNet._encode_batch with SEQUENCE returns (B, T, H)."""

    def test_sequence_shape(self) -> None:
        """SEQUENCE produces 3-D tensor from full encoder output."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(4, 20, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.shape == (4, 20, 16)

    def test_sequence_ndim(self) -> None:
        """SEQUENCE output is exactly 3-D."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3

    def test_sequence_length_matches_input(self) -> None:
        """SEQUENCE temporal dimension matches input temporal dimension."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(3, 25, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.shape[1] == 25


class TestTimeNetGradientFlow:
    """Gradients flow through both VECTOR and SEQUENCE paths."""

    def test_vector_gradient_flows(self) -> None:
        """Gradients flow back through VECTOR encoding path."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3, requires_grad=True)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()

    def test_sequence_gradient_flows(self) -> None:
        """Gradients flow back through SEQUENCE encoding path."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3, requires_grad=True)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestTimeNetEncodeIntegration:
    """encode() with output=SEQUENCE works end-to-end."""

    def test_encode_sequence_via_mixin(self) -> None:
        """encode() mixin passes SEQUENCE through to _encode_batch."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        data = torch.randn(4, 20, 3)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.shape == (4, 20, 16)

    def test_encode_vector_via_mixin(self) -> None:
        """encode() mixin passes VECTOR through to _encode_batch."""
        model = TimeNet(hidden_dims=16, depth=1, input_dims=3)
        data = torch.randn(4, 20, 3)
        result = model.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.VECTOR)
        assert result.shape == (4, 16)


# ---------------------------------------------------------------------------
# RecurrentAutoEncoder
# ---------------------------------------------------------------------------


class TestRecurrentAESupportedOutputs:
    """RecurrentAutoEncoder.supported_outputs includes VECTOR and SEQUENCE."""

    def test_has_vector(self) -> None:
        """VECTOR is in supported_outputs."""
        assert EncodingOutputShape.VECTOR in RecurrentAutoEncoder.supported_outputs

    def test_has_sequence(self) -> None:
        """SEQUENCE is in supported_outputs."""
        assert EncodingOutputShape.SEQUENCE in RecurrentAutoEncoder.supported_outputs

    def test_is_frozenset(self) -> None:
        """supported_outputs is a frozenset."""
        assert isinstance(RecurrentAutoEncoder.supported_outputs, frozenset)


class TestRecurrentAEVectorOutput:
    """RecurrentAE._encode_batch with VECTOR returns (B, H)."""

    def test_vector_shape(self) -> None:
        """VECTOR produces 2-D tensor via last-step slice."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(4, 20, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        assert result.shape == (4, 16)

    def test_vector_ndim(self) -> None:
        """VECTOR output is exactly 2-D."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        assert result.ndim == 2


class TestRecurrentAESequenceOutput:
    """RecurrentAE._encode_batch with SEQUENCE returns (B, T, H)."""

    def test_sequence_shape(self) -> None:
        """SEQUENCE produces 3-D tensor from full encoder output."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(4, 20, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.shape == (4, 20, 16)

    def test_sequence_ndim(self) -> None:
        """SEQUENCE output is exactly 3-D."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.ndim == 3

    def test_sequence_length_matches_input(self) -> None:
        """SEQUENCE temporal dimension matches input temporal dimension."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(3, 25, 3)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        assert result.shape[1] == 25


class TestRecurrentAEGradientFlow:
    """Gradients flow through both VECTOR and SEQUENCE paths."""

    def test_vector_gradient_flows(self) -> None:
        """Gradients flow back through VECTOR encoding path."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3, requires_grad=True)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.VECTOR)
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()

    def test_sequence_gradient_flows(self) -> None:
        """Gradients flow back through SEQUENCE encoding path."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        encoder = model._get_encoder()
        data = torch.randn(2, 10, 3, requires_grad=True)
        result = model._encode_batch(encoder, data, output=EncodingOutputShape.SEQUENCE)
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestRecurrentAEEncodeIntegration:
    """encode() with output=SEQUENCE works end-to-end."""

    def test_encode_sequence_via_mixin(self) -> None:
        """encode() mixin passes SEQUENCE through to _encode_batch."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        data = torch.randn(4, 20, 3)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.shape == (4, 20, 16)

    def test_encode_vector_via_mixin(self) -> None:
        """encode() mixin passes VECTOR through to _encode_batch."""
        model = RecurrentAutoEncoder(input_dims=3, layers=(16,))
        data = torch.randn(4, 20, 3)
        result = model.encode(data, batch_size=2, num_workers=0, output=EncodingOutputShape.VECTOR)
        assert result.shape == (4, 16)
