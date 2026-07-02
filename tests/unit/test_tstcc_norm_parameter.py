"""Tests for norm parameter across TSTCC sub-modules.

Verifies TCCEncoder, TemporalContrast, and TSTCC accept a `norm` parameter
defaulting to "layer", replacing BatchNorm with GroupNorm/LayerNorm to avoid
batch degeneracy at small batch sizes. Per D-02.
"""

import pytest
import torch
from torch import nn

from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC


class TestTCCEncoderNormDefault:
    """TCCEncoder defaults to norm='layer' (GroupNorm)."""

    def test_default_norm_is_layer(self) -> None:
        """TCCEncoder with no norm arg uses GroupNorm in all 3 conv blocks."""
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4)
        # conv_block1[1], conv_block2[1], conv_block3[1] are the norm layers
        assert isinstance(encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block3[1], nn.GroupNorm)

    def test_default_group_norm_groups(self) -> None:
        """GroupNorm uses num_groups=1 in all conv blocks."""
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4)
        assert encoder.conv_block1[1].num_groups == 1
        assert encoder.conv_block2[1].num_groups == 1
        assert encoder.conv_block3[1].num_groups == 1

    def test_default_group_norm_channels(self) -> None:
        """GroupNorm channels match the expected channel counts."""
        encoder = TCCEncoder(
            input_dims=3, conv_kernel_size=8, stride=4, encoder_channels=(32, 64), output_dims=128
        )
        assert encoder.conv_block1[1].num_channels == 32
        assert encoder.conv_block2[1].num_channels == 64
        assert encoder.conv_block3[1].num_channels == 128


class TestTCCEncoderNormExplicit:
    """TCCEncoder respects explicit norm='layer' and norm='batch'."""

    def test_explicit_layer_norm(self) -> None:
        """TCCEncoder(norm='layer') uses GroupNorm."""
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4, norm="layer")
        assert isinstance(encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(encoder.conv_block3[1], nn.GroupNorm)

    def test_explicit_batch_norm(self) -> None:
        """TCCEncoder(norm='batch') uses BatchNorm1d (backward compat)."""
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4, norm="batch")
        assert isinstance(encoder.conv_block1[1], nn.BatchNorm1d)
        assert isinstance(encoder.conv_block2[1], nn.BatchNorm1d)
        assert isinstance(encoder.conv_block3[1], nn.BatchNorm1d)


class TestTCCEncoderNormValidation:
    """TCCEncoder validates the norm parameter."""

    def test_invalid_norm_raises(self) -> None:
        """TCCEncoder(norm='invalid') raises ValueError."""
        with pytest.raises(ValueError, match="norm must be"):
            TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4, norm="invalid")


class TestTCCEncoderGradientFlow:
    """Gradients flow through GroupNorm-normalized encoder."""

    def test_gradient_flows_with_group_norm(self) -> None:
        """Gradient flows through TCCEncoder with GroupNorm at batch_size=1."""
        encoder = TCCEncoder(input_dims=3, conv_kernel_size=8, stride=4, norm="layer")
        data = torch.randn(1, 64, 3, requires_grad=True)
        output = encoder(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()


class TestTemporalContrastNormDefault:
    """TemporalContrast defaults to norm='layer' (LayerNorm)."""

    def test_default_norm_is_layer(self) -> None:
        """TemporalContrast with no norm arg uses LayerNorm in projection_head."""
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6)
        # projection_head[1] is the norm layer
        assert isinstance(tc.projection_head[1], nn.LayerNorm)

    def test_default_layer_norm_dim(self) -> None:
        """LayerNorm operates on the correct dimension."""
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6)
        assert tc.projection_head[1].normalized_shape == (8,)  # num_channels // 2


class TestTemporalContrastNormExplicit:
    """TemporalContrast respects explicit norm='layer' and norm='batch'."""

    def test_explicit_layer_norm(self) -> None:
        """TemporalContrast(norm='layer') uses LayerNorm."""
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6, norm="layer")
        assert isinstance(tc.projection_head[1], nn.LayerNorm)

    def test_explicit_batch_norm(self) -> None:
        """TemporalContrast(norm='batch') uses BatchNorm1d (backward compat)."""
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6, norm="batch")
        assert isinstance(tc.projection_head[1], nn.BatchNorm1d)


class TestTemporalContrastNormValidation:
    """TemporalContrast validates the norm parameter."""

    def test_invalid_norm_raises(self) -> None:
        """TemporalContrast(norm='invalid') raises ValueError."""
        with pytest.raises(ValueError, match="norm must be"):
            TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6, norm="invalid")


class TestTemporalContrastGradientFlow:
    """Gradients flow through LayerNorm-normalized projection head."""

    def test_gradient_flows_with_layer_norm(self) -> None:
        """Gradient flows through TemporalContrast with LayerNorm at batch_size=1."""
        tc = TemporalContrast(num_channels=16, hidden_dim=100, timesteps=6, norm="layer")
        features = torch.randn(1, 16, 32)  # (B, C, L) > timestep=6
        nce, proj = tc(features, features)
        loss = nce + proj.sum()
        loss.backward()
        assert torch.isfinite(loss)


class TestTSTCCNormDefault:
    """TSTCC defaults to norm='layer' and passes to sub-modules."""

    def test_default_uses_group_norm_in_encoder(self) -> None:
        """TSTCC default constructor uses GroupNorm in encoder."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4)
        assert isinstance(model._encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(model._encoder.conv_block2[1], nn.GroupNorm)
        assert isinstance(model._encoder.conv_block3[1], nn.GroupNorm)

    def test_default_uses_layer_norm_in_tc(self) -> None:
        """TSTCC default constructor uses LayerNorm in TemporalContrast."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4)
        assert isinstance(model._tc_model.projection_head[1], nn.LayerNorm)


class TestTSTCCNormExplicit:
    """TSTCC respects explicit norm='layer' and norm='batch'."""

    def test_explicit_layer_norm(self) -> None:
        """TSTCC(norm='layer') passes layer norm to both sub-modules."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, norm="layer")
        assert isinstance(model._encoder.conv_block1[1], nn.GroupNorm)
        assert isinstance(model._tc_model.projection_head[1], nn.LayerNorm)

    def test_explicit_batch_norm(self) -> None:
        """TSTCC(norm='batch') passes batch norm to both sub-modules."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, norm="batch")
        assert isinstance(model._encoder.conv_block1[1], nn.BatchNorm1d)
        assert isinstance(model._tc_model.projection_head[1], nn.BatchNorm1d)


class TestTSTCCNormValidation:
    """TSTCC validates the norm parameter."""

    def test_invalid_norm_raises(self) -> None:
        """TSTCC(norm='invalid') raises ValueError."""
        with pytest.raises(ValueError, match="norm must be"):
            TSTCC(input_dims=3, conv_kernel_size=8, stride=4, norm="invalid")


class TestTSTCCNormGradientFlow:
    """Gradients flow through TSTCC with layer normalization at batch_size=1."""

    def test_gradient_flows_at_batch_size_1(self) -> None:
        """TSTCC encoder gradient flows at batch_size=1 with GroupNorm."""
        model = TSTCC(input_dims=3, conv_kernel_size=8, stride=4, norm="layer")
        data = torch.randn(1, 64, 3, requires_grad=True)
        output = model(data)
        output.sum().backward()
        assert data.grad is not None
        assert torch.isfinite(data.grad).all()
