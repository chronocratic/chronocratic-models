"""Tests for normalization_layer_type parameter in FCNEncoder and MCL.

Verifies that FCNEncoder accepts a normalization_layer_type parameter defaulting
to CHANNEL, uses GroupNorm(1, ch) for CHANNEL, and preserves BatchNorm1d(ch)
for BATCH. MCL passes normalization_layer_type through to both FCNEncoder and
proj_head.
"""

from __future__ import annotations

import torch
from torch import nn

from chronocratic.models import MCL
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder
from chronocratic.models.enums.layers import NormalizationLayerType


class TestFCNEncoderNormDefault:
    """FCNEncoder defaults to CHANNEL (GroupNorm)."""

    def test_default_uses_group_norm(self) -> None:
        encoder = FCNEncoder(input_dims=3)
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(norm_layers) == 3  # one per conv block
        assert len(bn_layers) == 0

    def test_default_group_norm_config(self) -> None:
        encoder = FCNEncoder(input_dims=3)
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        expected_channels = [128, 256, 128]
        for nl, ch in zip(norm_layers, expected_channels, strict=True):
            assert nl.num_groups == 1
            assert nl.num_channels == ch


class TestFCNEncoderNormChannel:
    """FCNEncoder(normalization_layer_type=CHANNEL) uses GroupNorm."""

    def test_explicit_channel_uses_group_norm(self) -> None:
        encoder = FCNEncoder(input_dims=3, normalization_layer_type=NormalizationLayerType.CHANNEL)
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(norm_layers) == 3
        assert len(bn_layers) == 0

    def test_channel_forward_works_batch_size_1(self) -> None:
        """GroupNorm works at batch_size=1 without degeneracy."""
        encoder = FCNEncoder(input_dims=3, normalization_layer_type=NormalizationLayerType.CHANNEL)
        x = torch.randn(1, 100, 3)
        out = encoder(x)
        assert out.shape == (1, 128)
        assert not torch.isnan(out).any()

    def test_channel_forward_gradient_batch_size_1(self) -> None:
        """Gradients flow correctly with batch_size=1 and GroupNorm."""
        encoder = FCNEncoder(input_dims=3, normalization_layer_type=NormalizationLayerType.CHANNEL)
        x = torch.randn(1, 100, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestFCNEncoderNormBatch:
    """FCNEncoder(normalization_layer_type=BATCH) preserves BatchNorm1d."""

    def test_batch_uses_batch_norm_1d(self) -> None:
        encoder = FCNEncoder(input_dims=3, normalization_layer_type=NormalizationLayerType.BATCH)
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        gn_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(bn_layers) == 3
        assert len(gn_layers) == 0

    def test_batch_preserves_layer_count(self) -> None:
        """BATCH preserves original layer count."""
        encoder = FCNEncoder(input_dims=3, normalization_layer_type=NormalizationLayerType.BATCH)
        assert len(encoder.layers) == 12


class TestFCNEncoderCustomChannelsNorm:
    """Normalization works with custom encoder channels."""

    def test_two_blocks_channel(self) -> None:
        encoder = FCNEncoder(
            input_dims=3,
            output_dims=64,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(norm_layers) == 2
        assert norm_layers[0].num_channels == 64
        assert norm_layers[1].num_channels == 128

    def test_two_blocks_batch(self) -> None:
        encoder = FCNEncoder(
            input_dims=3,
            output_dims=64,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
            normalization_layer_type=NormalizationLayerType.BATCH,
        )
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 2
        assert bn_layers[0].num_features == 64
        assert bn_layers[1].num_features == 128


class TestMCLNormDefault:
    """MCL defaults to CHANNEL and passes it through."""

    def test_default_uses_group_norm_in_encoder(self) -> None:
        model = MCL(input_dims=3)
        norm_layers = [m for m in model._encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(norm_layers) == 3

    def test_default_uses_group_norm_in_proj_head(self) -> None:
        model = MCL(input_dims=3)
        assert isinstance(model.proj_head[1], nn.GroupNorm)


class TestMCLNormChannel:
    """MCL(normalization_layer_type=CHANNEL) uses GroupNorm everywhere."""

    def test_channel_encoder(self) -> None:
        model = MCL(input_dims=3, normalization_layer_type=NormalizationLayerType.CHANNEL)
        norm_layers = [m for m in model._encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(norm_layers) == 3

    def test_channel_proj_head(self) -> None:
        model = MCL(
            input_dims=3,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
            projection_dims=64,
        )
        assert isinstance(model.proj_head[1], nn.GroupNorm)
        assert model.proj_head[1].num_channels == 64

    def test_channel_forward_batch_size_1(self) -> None:
        model = MCL(input_dims=3, normalization_layer_type=NormalizationLayerType.CHANNEL)
        x = torch.randn(1, 100, 3)
        out = model(x)
        assert out.shape == (1, 128)


class TestMCLNormBatch:
    """MCL(normalization_layer_type=BATCH) uses BatchNorm1d everywhere."""

    def test_batch_encoder(self) -> None:
        model = MCL(input_dims=3, normalization_layer_type=NormalizationLayerType.BATCH)
        bn_layers = [m for m in model._encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 3

    def test_batch_proj_head(self) -> None:
        model = MCL(input_dims=3, normalization_layer_type=NormalizationLayerType.BATCH)
        assert isinstance(model.proj_head[1], nn.BatchNorm1d)
