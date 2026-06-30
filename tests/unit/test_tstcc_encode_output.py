"""Tests for TSTCC output-aware _encode_batch.

Verifies TSTCC honors the `output` parameter in `_encode_batch` to return
both VECTOR (2-D) and SEQUENCE (3-D) representations.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.enums.encoding import EncodingOutputShape


class TestTSTCCSupportedOutputs:
    """TSTCC.supported_outputs includes VECTOR and SEQUENCE."""

    def test_has_vector(self) -> None:
        """VECTOR is in supported_outputs."""
        assert EncodingOutputShape.VECTOR in TSTCC.supported_outputs

    def test_has_sequence(self) -> None:
        """SEQUENCE is in supported_outputs."""
        assert EncodingOutputShape.SEQUENCE in TSTCC.supported_outputs

    def test_is_frozenset(self) -> None:
        """supported_outputs is a frozenset."""
        assert isinstance(TSTCC.supported_outputs, frozenset)


class TestTSTCCVectorOutput:
    """TSTCC._encode_batch with VECTOR returns (B, output_dims)."""

    def test_vector_shape(self) -> None:
        """VECTOR produces 2-D tensor with pooled feature shape."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(4, 256, 3)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.VECTOR
        )
        assert result.shape == (4, 16)

    def test_vector_ndim(self) -> None:
        """VECTOR output is exactly 2-D."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(2, 128, 3)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.VECTOR
        )
        assert result.ndim == 2


class TestTSTCCSequenceOutput:
    """TSTCC._encode_batch with SEQUENCE returns (B, L_prime, output_dims)."""

    def test_sequence_shape(self) -> None:
        """SEQUENCE produces 3-D tensor from conv feature map transpose."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(4, 256, 3)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.SEQUENCE
        )
        # conv with kernel=8, stride=4: L' = (256 - 8) // 4 + 1 = 63
        # then two inner blocks with kernel=8, stride=1: L'' = 63 - 8 + 1 = 56
        # (exact length depends on encoder internals; assert ndim and last dim)
        assert result.ndim == 3
        assert result.shape[0] == 4  # batch
        assert result.shape[2] == 16  # output_dims

    def test_sequence_ndim(self) -> None:
        """SEQUENCE output is exactly 3-D."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(2, 128, 3)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3

    def test_sequence_length_matches_conv_output(self) -> None:
        """SEQUENCE length dimension matches encoder output spatial dimension."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(2, 256, 3)
        # encoder outputs (B, C, L')
        raw = encoder(data.float())
        seq = model._encode_batch(
            encoder, data, output=EncodingOutputShape.SEQUENCE
        )
        # SEQUENCE should be (B, L', C) so seq.shape[1] == raw.shape[2]
        assert seq.shape[1] == raw.shape[2]


class TestTSTCCGradientFlow:
    """Gradients flow through both VECTOR and SEQUENCE paths."""

    def test_vector_gradient_flows(self) -> None:
        """Gradients flow back through VECTOR encoding path."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(2, 128, 3, requires_grad=True)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.VECTOR
        )
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()

    def test_sequence_gradient_flows(self) -> None:
        """Gradients flow back through SEQUENCE encoding path."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        encoder = model._get_encoder()
        data = torch.randn(2, 128, 3, requires_grad=True)
        result = model._encode_batch(
            encoder, data, output=EncodingOutputShape.SEQUENCE
        )
        result.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestTSTCCEncodeIntegration:
    """encode() with output=SEQUENCE works end-to-end."""

    def test_encode_sequence_via_mixin(self) -> None:
        """encode() mixin passes SEQUENCE through to _encode_batch."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        data = torch.randn(4, 256, 3)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.SEQUENCE
        )
        assert result.ndim == 3
        assert result.shape[0] == 4
        assert result.shape[2] == 16

    def test_encode_vector_via_mixin(self) -> None:
        """encode() mixin passes VECTOR through to _encode_batch."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, output_dims=16)
        data = torch.randn(4, 256, 3)
        result = model.encode(
            data, batch_size=2, num_workers=0, output=EncodingOutputShape.VECTOR
        )
        assert result.ndim == 2
        assert result.shape == (4, 16)
