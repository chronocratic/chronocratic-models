"""Tests for normalization_layer_type parameter across TSTCC sub-modules.

Verifies TCCEncoder, TemporalContrast, and TSTCC accept a
normalization_layer_type parameter defaulting to CHANNEL, replacing
BatchNorm with GroupNorm/LayerNorm to avoid batch degeneracy at small batch
sizes.
"""

import torch
from torch import nn

from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC
from chronocratic.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast
from chronocratic.models.enums.layers import NormalizationLayerType


class TestTCCEncoderNormDefault:
    """TCCEncoder defaults to CHANNEL (GroupNorm)."""

    def test_default_norm_is_channel(self) -> None:
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4)
        assert isinstance(encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block3[1], nn.GroupNorm)

    def test_default_group_norm_groups(self) -> None:
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4)
        assert encoder.conv_block1[1].num_groups == 1
        assert encoder.conv_block2[1].num_groups == 1
        assert encoder.conv_block3[1].num_groups == 1

    def test_default_group_norm_channels(self) -> None:
        encoder = TCCEncoder(
            input_dims=3, conv_kernel_size=8, stride=4, encoder_channels=(32, 64), output_dims=128
        )
        assert encoder.conv_block1[1].num_channels == 32
        assert encoder.conv_block2[1].num_channels == 64
        assert encoder.conv_block3[1].num_channels == 128


class TestTCCEncoderNormExplicit:
    """TCCEncoder respects explicit CHANNEL and BATCH."""

    def test_explicit_channel(self) -> None:
        encoder = TCCEncoder(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        assert isinstance(encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block3[1], nn.GroupNorm)

    def test_explicit_batch(self) -> None:
        encoder = TCCEncoder(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.BATCH,
        )
        assert isinstance(encoder.conv_block1[1], nn.BatchNorm1d)
        assert isinstance(encoder.conv_block2[1], nn.BatchNorm1d)
        assert isinstance(encoder.conv_block3[1], nn.BatchNorm1d)


class TestTCCEncoderGradientFlow:
    """Gradients flow through GroupNorm-normalized encoder."""

    def test_gradient_flows_with_group_norm(self) -> None:
        encoder = TCCEncoder(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        data = torch.randn(1, 64, 3, requires_grad=True)
        output = encoder(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestTemporalContrastNormDefault:
    """TemporalContrast defaults to CHANNEL (LayerNorm)."""

    def test_default_norm_is_layer(self) -> None:
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6)
        assert isinstance(tc.projection_head[1], nn.LayerNorm)

    def test_default_layer_norm_dim(self) -> None:
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6)
        assert tc.projection_head[1].normalized_shape == (8,)


class TestTemporalContrastNormExplicit:
    """TemporalContrast respects explicit CHANNEL and BATCH."""

    def test_explicit_channel(self) -> None:
        tc = TemporalContrast(
            num_channels=16,
            hidden_dim=100,
            timesteps=6,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        assert isinstance(tc.projection_head[1], nn.LayerNorm)

    def test_explicit_batch(self) -> None:
        tc = TemporalContrast(
            num_channels=16,
            hidden_dim=100,
            timesteps=6,
            normalization_layer_type=NormalizationLayerType.BATCH,
        )
        assert isinstance(tc.projection_head[1], nn.BatchNorm1d)


class TestTemporalContrastGradientFlow:
    """Gradients flow through LayerNorm-normalized projection head."""

    def test_gradient_flows_with_layer_norm(self) -> None:
        tc = TemporalContrast(
            num_channels=16,
            hidden_dim=100,
            timesteps=6,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        features = torch.randn(1, 16, 32)
        nce, proj = tc(features, features)
        loss = nce + proj.sum()
        loss.backward()
        assert torch.isfinite(loss)


class TestTSTCCNormDefault:
    """TSTCC defaults to CHANNEL and passes to sub-modules."""

    def test_default_uses_group_norm_in_encoder(self) -> None:
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4)
        assert isinstance(model._encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(model._encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(model._encoder.conv_block3[1], nn.GroupNorm)

    def test_default_uses_layer_norm_in_tc(self) -> None:
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4)
        assert isinstance(model._tc_model.projection_head[1], nn.LayerNorm)


class TestTSTCCNormExplicit:
    """TSTCC respects explicit CHANNEL and BATCH."""

    def test_explicit_channel(self) -> None:
        model = TSTCC(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        assert isinstance(model._encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(model._tc_model.projection_head[1], nn.LayerNorm)

    def test_explicit_batch(self) -> None:
        model = TSTCC(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.BATCH,
        )
        assert isinstance(model._encoder.conv_block1[1], nn.BatchNorm1d)
        assert isinstance(model._tc_model.projection_head[1], nn.BatchNorm1d)


class TestTSTCCNormGradientFlow:
    """Gradients flow through TSTCC with channel normalization at batch_size=1."""

    def test_gradient_flows_at_batch_size_1(self) -> None:
        model = TSTCC(
            input_dims=3,
            conv_kernel_size=8,
            stride=4,
            normalization_layer_type=NormalizationLayerType.CHANNEL,
        )
        data = torch.randn(1, 64, 3, requires_grad=True)
        output = model(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()
