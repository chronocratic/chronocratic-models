"""Tests for the norm parameter across Series2Vec's DisjoinEncoder and model.

Verifies DisjoinEncoder and Series2Vec accept a `norm` parameter defaulting to
"layer" (GroupNorm, batch-size-1 safe), with "batch" reproducing the upstream
BatchNorm2d/BatchNorm1d architecture exactly.
"""

import pytest
import torch
from torch import nn

from chronocratic.models.convolutional.standard.series2vec.encoder import DisjoinEncoder
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


def _make_encoder(norm: str = "layer") -> DisjoinEncoder:
    return DisjoinEncoder(
        input_dims=3, embedding_dims=8, representation_dims=16, kernel_size=4, norm=norm
    )


class TestDisjoinEncoderNormDefault:
    """DisjoinEncoder defaults to norm='layer' (GroupNorm)."""

    def test_default_norm_is_group_norm(self) -> None:
        """No norm arg uses GroupNorm in all 3 conv blocks."""
        encoder = _make_encoder()
        # temporal_CNN[1], spatial_CNN[1], rep_CNN[1] are the norm layers.
        assert isinstance(encoder.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.spatial_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.rep_CNN[1], nn.GroupNorm)

    def test_default_group_norm_single_group(self) -> None:
        """GroupNorm uses num_groups=1 in all conv blocks."""
        encoder = _make_encoder()
        assert encoder.temporal_CNN[1].num_groups == 1
        assert encoder.spatial_CNN[1].num_groups == 1
        assert encoder.rep_CNN[1].num_groups == 1


class TestDisjoinEncoderNormExplicit:
    """DisjoinEncoder respects explicit norm='layer' and norm='batch'."""

    def test_explicit_layer_norm(self) -> None:
        """DisjoinEncoder(norm='layer') uses GroupNorm."""
        encoder = _make_encoder(norm="layer")
        assert isinstance(encoder.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.spatial_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.rep_CNN[1], nn.GroupNorm)

    def test_explicit_batch_norm(self) -> None:
        """DisjoinEncoder(norm='batch') uses BatchNorm2d x2 + BatchNorm1d (upstream)."""
        encoder = _make_encoder(norm="batch")
        assert isinstance(encoder.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(encoder.spatial_CNN[1], nn.BatchNorm2d)
        assert isinstance(encoder.rep_CNN[1], nn.BatchNorm1d)


class TestDisjoinEncoderNormValidation:
    """DisjoinEncoder validates the norm parameter."""

    def test_invalid_norm_raises(self) -> None:
        """DisjoinEncoder(norm='invalid') raises ValueError."""
        with pytest.raises(ValueError, match="norm must be"):
            _make_encoder(norm="invalid")


class TestDisjoinEncoderGradientFlow:
    """Gradients flow through the GroupNorm-normalized encoder at batch_size=1."""

    def test_gradient_flows_with_group_norm(self) -> None:
        """Gradient flows through DisjoinEncoder(norm='layer') at batch_size=1."""
        encoder = _make_encoder(norm="layer")
        # DisjoinEncoder expects (batch, channels, time).
        data = torch.randn(1, 3, 32, requires_grad=True)
        output = encoder(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


def _make_model(norm: str = "layer") -> Series2Vec:
    return Series2Vec(
        input_dims=3,
        embedding_dims=8,
        representation_dims=16,
        encoder_kernel_size=4,
        num_heads=2,
        feedforward_dims=32,
        norm=norm,
    )


class TestSeries2VecNormThreading:
    """Series2Vec threads norm into both temporal and frequency encoders."""

    def test_default_uses_group_norm_in_both_branches(self) -> None:
        """Default constructor uses GroupNorm in embed_layer and embed_layer_f."""
        model = _make_model()
        assert isinstance(model.network.embed_layer.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(model.network.embed_layer_f.temporal_CNN[1], nn.GroupNorm)

    def test_explicit_batch_norm_in_both_branches(self) -> None:
        """norm='batch' passes BatchNorm to embed_layer and embed_layer_f."""
        model = _make_model(norm="batch")
        assert isinstance(model.network.embed_layer.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(model.network.embed_layer.rep_CNN[1], nn.BatchNorm1d)
        assert isinstance(model.network.embed_layer_f.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(model.network.embed_layer_f.rep_CNN[1], nn.BatchNorm1d)

    def test_invalid_norm_raises(self) -> None:
        """Series2Vec(norm='invalid') raises ValueError from the encoder."""
        with pytest.raises(ValueError, match="norm must be"):
            _make_model(norm="invalid")
