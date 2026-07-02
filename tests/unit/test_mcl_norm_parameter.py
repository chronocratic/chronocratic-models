"""Tests for norm parameter in FCNEncoder and MCL.

Verifies that FCNEncoder accepts a norm parameter defaulting to "layer",
uses GroupNorm(1, ch) when norm="layer", and preserves BatchNorm1d(ch)
when norm="batch". MCL passes norm through to both FCNEncoder and proj_head.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chronocratic.models import MCL
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder


class TestFCNEncoderNormDefault:
    """FCNEncoder defaults to norm='layer' (GroupNorm)."""

    def test_default_uses_group_norm(self) -> None:
        encoder = FCNEncoder(input_dims=3)
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(norm_layers) == 3  # one per conv block
        assert len(bn_layers) == 0

    def test_default_group_norm_config(self) -> None:
        encoder = FCNEncoder(input_dims=3)
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        # GroupNorm(1, ch) for each block: 128, 256, 128
        expected_channels = [128, 256, 128]
        for nl, ch in zip(norm_layers, expected_channels, strict=True):
            assert nl.num_groups == 1
            assert nl.num_channels == ch


class TestFCNEncoderNormLayer:
    """FCNEncoder(norm='layer') uses GroupNorm."""

    def test_explicit_layer_norm_uses_group_norm(self) -> None:
        encoder = FCNEncoder(input_dims=3, norm="layer")
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(norm_layers) == 3
        assert len(bn_layers) == 0

    def test_layer_norm_forward_works_batch_size_1(self) -> None:
        """GroupNorm works at batch_size=1 without degeneracy."""
        encoder = FCNEncoder(input_dims=3, norm="layer")
        x = torch.randn(1, 100, 3)  # batch_size=1
        out = encoder(x)
        assert out.shape == (1, 128)
        assert not torch.isnan(out).any()

    def test_layer_norm_forward_gradient_batch_size_1(self) -> None:
        """Verify gradients flow correctly with batch_size=1 and GroupNorm."""
        encoder = FCNEncoder(input_dims=3, norm="layer")
        x = torch.randn(1, 100, 3, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestFCNEncoderNormBatch:
    """FCNEncoder(norm='batch') preserves BatchNorm1d (backward compat)."""

    def test_batch_norm_uses_batch_norm_1d(self) -> None:
        encoder = FCNEncoder(input_dims=3, norm="batch")
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        gn_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(bn_layers) == 3
        assert len(gn_layers) == 0

    def test_batch_norm_backward_compat(self) -> None:
        """norm='batch' preserves original layer count."""
        encoder = FCNEncoder(input_dims=3, norm="batch")
        # 3 conv blocks * 3 layers each + AdaptiveAvgPool + Flatten + Linear = 12
        assert len(encoder.layers) == 12


class TestFCNEncoderNormValidation:
    """FCNEncoder raises ValueError for invalid norm values."""

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="norm must be"):
            FCNEncoder(input_dims=3, norm="invalid")

    def test_empty_string_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="norm must be"):
            FCNEncoder(input_dims=3, norm="")


class TestFCNEncoderCustomChannelsNorm:
    """norm works with custom encoder channels."""

    def test_two_blocks_layer_norm(self) -> None:
        encoder = FCNEncoder(
            input_dims=3,
            output_dims=64,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
            norm="layer",
        )
        norm_layers = [m for m in encoder.layers if isinstance(m, nn.GroupNorm)]
        assert len(norm_layers) == 2
        assert norm_layers[0].num_channels == 64
        assert norm_layers[1].num_channels == 128

    def test_two_blocks_batch_norm(self) -> None:
        encoder = FCNEncoder(
            input_dims=3,
            output_dims=64,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
            norm="batch",
        )
        bn_layers = [m for m in encoder.layers if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) == 2
        assert bn_layers[0].num_features == 64
        assert bn_layers[1].num_features == 128


class TestMCLNormDefault:
    """MCL defaults to norm='layer' and passes it through."""

    def test_default_uses_group_norm_in_encoder(self) -> None:
        model = MCL(input_dims=3)
        norm_layers = [m for m in model._encoder.layers if isinstance(m, nn.GroupNorm)]  # noqa: SLF001
        assert len(norm_layers) == 3

    def test_default_uses_group_norm_in_proj_head(self) -> None:
        model = MCL(input_dims=3)
        # proj_head: Linear, GroupNorm, ReLU, Linear
        assert isinstance(model.proj_head[1], nn.GroupNorm)


class TestMCLNormLayer:
    """MCL(norm='layer') uses GroupNorm in encoder and proj_head."""

    def test_layer_norm_encoder(self) -> None:
        model = MCL(input_dims=3, norm="layer")
        norm_layers = [m for m in model._encoder.layers if isinstance(m, nn.GroupNorm)]  # noqa: SLF001
        assert len(norm_layers) == 3

    def test_layer_norm_proj_head(self) -> None:
        model = MCL(input_dims=3, norm="layer", projection_dims=64)
        assert isinstance(model.proj_head[1], nn.GroupNorm)
        assert model.proj_head[1].num_channels == 64

    def test_layer_norm_forward_batch_size_1(self) -> None:
        model = MCL(input_dims=3, norm="layer")
        x = torch.randn(1, 100, 3)
        out = model(x)
        assert out.shape == (1, 128)


class TestMCLNormBatch:
    """MCL(norm='batch') uses BatchNorm1d in encoder and proj_head."""

    def test_batch_norm_encoder(self) -> None:
        model = MCL(input_dims=3, norm="batch")
        bn_layers = [m for m in model._encoder.layers if isinstance(m, nn.BatchNorm1d)]  # noqa: SLF001
        assert len(bn_layers) == 3

    def test_batch_norm_proj_head(self) -> None:
        model = MCL(input_dims=3, norm="batch")
        assert isinstance(model.proj_head[1], nn.BatchNorm1d)


class TestMCLNormValidation:
    """MCL raises ValueError for invalid norm values."""

    def test_invalid_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="norm must be"):
            MCL(input_dims=3, norm="invalid")
