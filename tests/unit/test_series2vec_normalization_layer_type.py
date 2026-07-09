"""Tests for normalization_layer_type across Series2Vec's DisjoinEncoder and model.

Verifies DisjoinEncoder and Series2Vec accept normalization_layer_type parameter
defaulting to CHANNEL (GroupNorm, batch-size-1 safe), with BATCH reproducing
the upstream BatchNorm2d/BatchNorm1d architecture exactly.
"""

import torch
from torch import nn

from chronocratic.models.convolutional.standard.series2vec.encoder import DisjoinEncoder
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.enums.layers import NormalizationLayerType


def _make_encoder(
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
) -> DisjoinEncoder:
    return DisjoinEncoder(
        input_dim=3,
        embedding_dim=8,
        representation_dim=16,
        kernel_size=4,
        normalization_layer_type=normalization_layer_type,
    )


class TestDisjoinEncoderNormDefault:
    """DisjoinEncoder defaults to CHANNEL (GroupNorm)."""

    def test_default_norm_is_group_norm(self) -> None:
        encoder = _make_encoder()
        assert isinstance(encoder.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.spatial_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.rep_CNN[1], nn.GroupNorm)

    def test_default_group_norm_single_group(self) -> None:
        encoder = _make_encoder()
        assert encoder.temporal_CNN[1].num_groups == 1
        assert encoder.spatial_CNN[1].num_groups == 1
        assert encoder.rep_CNN[1].num_groups == 1


class TestDisjoinEncoderNormExplicit:
    """DisjoinEncoder respects explicit CHANNEL and BATCH."""

    def test_explicit_channel(self) -> None:
        encoder = _make_encoder(NormalizationLayerType.CHANNEL)
        assert isinstance(encoder.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.spatial_CNN[1], nn.GroupNorm)
        assert isinstance(encoder.rep_CNN[1], nn.GroupNorm)

    def test_explicit_batch(self) -> None:
        encoder = _make_encoder(NormalizationLayerType.BATCH)
        assert isinstance(encoder.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(encoder.spatial_CNN[1], nn.BatchNorm2d)
        assert isinstance(encoder.rep_CNN[1], nn.BatchNorm1d)


class TestDisjoinEncoderGradientFlow:
    """Gradients flow through GroupNorm encoder at batch_size=1."""

    def test_gradient_flows_with_group_norm(self) -> None:
        encoder = _make_encoder(NormalizationLayerType.CHANNEL)
        data = torch.randn(1, 3, 32, requires_grad=True)
        output = encoder(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


def _make_model(
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
) -> Series2Vec:
    return Series2Vec(
        input_dim=3,
        embedding_dim=8,
        representation_dim=16,
        encoder_kernel_size=4,
        num_heads=2,
        feedforward_dim=32,
        normalization_layer_type=normalization_layer_type,
    )


class TestSeries2VecNormThreading:
    """Series2Vec threads normalization_layer_type into both branches."""

    def test_default_uses_group_norm_in_both_branches(self) -> None:
        model = _make_model()
        assert isinstance(model.network.embed_layer.temporal_CNN[1], nn.GroupNorm)
        assert isinstance(model.network.embed_layer_f.temporal_CNN[1], nn.GroupNorm)

    def test_explicit_batch_in_both_branches(self) -> None:
        model = _make_model(NormalizationLayerType.BATCH)
        assert isinstance(model.network.embed_layer.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(model.network.embed_layer.rep_CNN[1], nn.BatchNorm1d)
        assert isinstance(model.network.embed_layer_f.temporal_CNN[1], nn.BatchNorm2d)
        assert isinstance(model.network.embed_layer_f.rep_CNN[1], nn.BatchNorm1d)
