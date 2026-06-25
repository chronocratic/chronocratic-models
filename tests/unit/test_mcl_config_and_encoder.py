"""Tests for MCL config field renames, FCNEncoder parameterization, and sync_dist fix.

Verifies that MCLModelParameters uses canonical naming (input_dims instead of
n_in), that the FCN model accepts all config fields via **vars() unpacking,
that FCNEncoder builds conv blocks dynamically from tuple parameters, and
that training/validation steps use self._sync_dist instead of hardcoded True.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
import torch
from torch import nn

from chronocratic.models import FCN
from chronocratic.models.convolutional.standard.mcl.config import MCLModelParameters
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder


class TestMCLModelParameters:
    """MCL config uses canonical field names and correct defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(MCLModelParameters)

    def test_input_dims_required(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.input_dims == 1

    def test_missing_input_dims_raises(self) -> None:
        with pytest.raises(TypeError):
            MCLModelParameters()  # type: ignore[call-arg]

    def test_encoder_channel_defaults(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.encoder_channels == (128, 256, 128)
        assert params.encoder_kernels == (7, 5, 3)
        assert params.encoder_dilations == (2, 4, 8)

    def test_projection_dims_default(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.projection_dims == 128

    def test_sync_dist_default(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.sync_dist is False

    def test_output_dims_default(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.output_dims == 320

    def test_alpha_default(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.alpha == 1.0

    def test_learning_rate_default(self) -> None:
        params = MCLModelParameters(input_dims=1)
        assert params.learning_rate == 1e-3

    def test_no_n_in_field(self) -> None:
        field_names = {f.name for f in fields(MCLModelParameters)}
        assert "n_in" not in field_names
        assert "input_dims" in field_names

    def test_vars_produces_correct_keys(self) -> None:
        params = MCLModelParameters(input_dims=1)
        result = vars(params)
        expected_keys = {
            "input_dims",
            "output_dims",
            "alpha",
            "learning_rate",
            "encoder_channels",
            "encoder_kernels",
            "encoder_dilations",
            "projection_dims",
            "sync_dist",
        }
        assert set(result.keys()) == expected_keys

    def test_field_count(self) -> None:
        assert len(fields(MCLModelParameters)) == 9


class TestFCNConfigContract:
    """FCN(**vars(MCLModelParameters(input_dims=1))) works without errors."""

    def test_vars_unpacking_instantiates_fcn(self) -> None:
        params = MCLModelParameters(input_dims=1)
        model = FCN(**vars(params))
        assert model is not None

    def test_custom_encoder_params(self) -> None:
        params = MCLModelParameters(
            input_dims=2,
            encoder_channels=(64, 128, 64),
            encoder_kernels=(5, 3, 3),
            encoder_dilations=(1, 2, 4),
            projection_dims=64,
        )
        model = FCN(**vars(params))
        assert model is not None

    def test_encoder_shape(self) -> None:
        params = MCLModelParameters(input_dims=1)
        model = FCN(**vars(params))
        x = torch.randn(4, 1, 100)
        encoding = model.encoder(x)
        assert encoding.shape == (4, 320)


class TestFCNSyncDist:
    """training_step and validation_step use self._sync_dist, not hardcoded True."""

    def test_sync_dist_attribute_set(self) -> None:
        model = FCN(input_dims=1)
        assert model._sync_dist is False  # noqa: SLF001

    def test_sync_dist_true(self) -> None:
        model = FCN(input_dims=1, sync_dist=True)
        assert model._sync_dist is True  # noqa: SLF001


class TestFCNEncoder:
    """FCNEncoder builds conv blocks from tuple parameters."""

    def test_default_architecture_layer_count(self) -> None:
        encoder = FCNEncoder(input_dims=1, output_dims=320)
        # 3 conv blocks * 3 layers each (Conv, BN, ReLU) + AdaptiveAvgPool + Flatten + Linear = 12
        assert len(encoder.layers) == 12

    def test_default_output_shape(self) -> None:
        encoder = FCNEncoder(input_dims=1, output_dims=320)
        x = torch.randn(2, 1, 50)
        out = encoder(x)
        assert out.shape == (2, 320)

    def test_custom_channels(self) -> None:
        encoder = FCNEncoder(
            input_dims=1,
            output_dims=256,
            encoder_channels=(64, 128, 64),
            encoder_kernels=(5, 3, 3),
            encoder_dilations=(1, 2, 4),
        )
        x = torch.randn(2, 1, 50)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_two_block_encoder(self) -> None:
        encoder = FCNEncoder(
            input_dims=3,
            output_dims=128,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
        )
        # 2 conv blocks * 3 layers + AdaptiveAvgPool + Flatten + Linear = 9
        assert len(encoder.layers) == 9
        x = torch.randn(2, 3, 100)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_fcn_encoder_accepts_btc_and_is_transpose_sensitive(self) -> None:
        """FCNEncoder must accept (B, T, C) input with T != C and return (B, output_dims).

        Regression test: without the transpose(1, 2) inside forward(), Conv1d
        sees T channels instead of input_dims and raises RuntimeError.
        """
        encoder = FCNEncoder(input_dims=3, output_dims=320)
        x = torch.randn(4, 50, 3)  # (B, T, C) with T=50 != C=3
        out = encoder(x)
        assert out.shape == (4, 320)

    def test_default_padding_matches_original(self) -> None:
        """Default encoder should produce identical padding to the original hardcoded version.

        Original: Conv1d(in, 128, k=7, padding=6, d=2) -> k//2*d = 7//2*2 = 6
        Original: Conv1d(128, 256, k=5, padding=8, d=4) -> k//2*d = 5//2*4 = 8
        Original: Conv1d(256, 128, k=3, padding=8, d=8) -> k//2*d = 3//2*8 = 8
        """
        encoder = FCNEncoder(input_dims=1, output_dims=320)
        conv_layers = [m for m in encoder.layers if isinstance(m, nn.Conv1d)]
        assert len(conv_layers) == 3
        assert conv_layers[0].padding == (6,)
        assert conv_layers[1].padding == (8,)
        assert conv_layers[2].padding == (8,)


class TestProjectionHead:
    """Projection head uses configurable projection_dims."""

    def test_default_projection_dims(self) -> None:
        model = FCN(input_dims=1)
        lin_layers = [m for m in model.proj_head if isinstance(m, nn.Linear)]
        assert len(lin_layers) == 2
        # output_dims=320 -> projection_dims=128 -> projection_dims=128
        assert lin_layers[0].in_features == 320
        assert lin_layers[0].out_features == 128
        assert lin_layers[1].in_features == 128
        assert lin_layers[1].out_features == 128

    def test_custom_projection_dims(self) -> None:
        model = FCN(input_dims=1, projection_dims=64)
        lin_layers = [m for m in model.proj_head if isinstance(m, nn.Linear)]
        assert lin_layers[0].out_features == 64
        assert lin_layers[1].in_features == 64
        assert lin_layers[1].out_features == 64
