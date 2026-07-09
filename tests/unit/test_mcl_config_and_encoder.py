"""Tests for MCL config field renames, FCNEncoder parameterization, and sync_dist fix.

Verifies that MCLModelParameters uses canonical singular dimension naming
(input_dim, representation_dim, projection_dim), that the MCL model accepts
all config fields via **vars() unpacking, that FCNEncoder builds conv blocks
dynamically from tuple parameters, and that training/validation steps use
self._sync_dist instead of hardcoded True.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
import torch
from torch import nn

from chronocratic.models import MCL
from chronocratic.models.convolutional.standard.mcl.config import MCLModelParameters
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder


class TestMCLModelParameters:
    """MCL config uses singular dimension field names and correct defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(MCLModelParameters)

    def test_input_dim_required(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.input_dim == 1

    def test_missing_input_dim_raises(self) -> None:
        with pytest.raises(TypeError):
            MCLModelParameters()  # type: ignore[call-arg]

    def test_encoder_channel_defaults(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.encoder_channels == (128, 256, 128)
        assert params.encoder_kernels == (7, 5, 3)
        assert params.encoder_dilations == (2, 4, 8)

    def test_projection_dim_default(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.projection_dim == 128

    def test_sync_dist_default(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.sync_dist is False

    def test_representation_dim_default(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.representation_dim == 128

    def test_alpha_default(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.alpha == 1.0

    def test_learning_rate_default(self) -> None:
        params = MCLModelParameters(input_dim=1)
        assert params.learning_rate == 1e-3

    def test_no_n_in_field(self) -> None:
        field_names = {f.name for f in fields(MCLModelParameters)}
        assert "n_in" not in field_names
        assert "input_dim" in field_names

    def test_singular_dimension_names(self) -> None:
        """Config fields use singular dimension names (no _dims suffix)."""
        field_names = {f.name for f in fields(MCLModelParameters)}
        # These should exist (singular)
        assert "input_dim" in field_names
        assert "representation_dim" in field_names
        assert "projection_dim" in field_names
        # Plural forms should NOT exist
        assert "input_dims" not in field_names
        assert "output_dims" not in field_names
        assert "projection_dims" not in field_names

    def test_vars_produces_correct_keys(self) -> None:
        params = MCLModelParameters(input_dim=1)
        result = vars(params)
        expected_keys = {
            "input_dim",
            "representation_dim",
            "alpha",
            "learning_rate",
            "encoder_channels",
            "encoder_kernels",
            "encoder_dilations",
            "projection_dim",
            "sync_dist",
            "normalization_layer_type",
        }
        assert set(result.keys()) == expected_keys

    def test_field_count(self) -> None:
        assert len(fields(MCLModelParameters)) == 10


class TestMCLConfigContract:
    """MCL(**vars(MCLModelParameters(input_dim=1))) works without errors."""

    def test_vars_unpacking_instantiates_mcl(self) -> None:
        params = MCLModelParameters(input_dim=1)
        model = MCL(**vars(params))
        assert model is not None

    def test_custom_encoder_params(self) -> None:
        params = MCLModelParameters(
            input_dim=2,
            encoder_channels=(64, 128, 64),
            encoder_kernels=(5, 3, 3),
            encoder_dilations=(1, 2, 4),
            projection_dim=64,
        )
        model = MCL(**vars(params))
        assert model is not None

    def test_encoder_shape(self) -> None:
        params = MCLModelParameters(input_dim=1)
        model = MCL(**vars(params))
        x = torch.randn(4, 100, 1)
        encoding = model.encoder(x)
        assert encoding.shape == (4, 128)


class TestMCLRepresentationDim:
    """MCL exposes representation_dim property returning encode() output width."""

    def test_representation_dim_property_exists(self) -> None:
        model = MCL(input_dim=1)
        assert hasattr(model, "representation_dim")

    def test_representation_dim_returns_encode_output_width(self) -> None:
        model = MCL(input_dim=1, representation_dim=128)
        assert model.representation_dim == 128

    def test_representation_dim_matches_encoder_output(self) -> None:
        model = MCL(input_dim=1, representation_dim=64)
        assert model.representation_dim == 64
        x = torch.randn(2, 50, 1)
        encoding = model.encoder(x)
        assert encoding.shape[1] == model.representation_dim

    def test_representation_dim_custom_value(self) -> None:
        model = MCL(input_dim=1, representation_dim=256)
        assert model.representation_dim == 256

    def test_representation_dim_is_readonly_property(self) -> None:
        """representation_dim is a property, not a settable attribute."""
        model = MCL(input_dim=1)
        with pytest.raises((AttributeError, TypeError)):
            model.representation_dim = 999  # type: ignore[misc]


class TestMCLSyncDist:
    """training_step and validation_step use self._sync_dist, not hardcoded True."""

    def test_sync_dist_attribute_set(self) -> None:
        model = MCL(input_dim=1)
        assert model._sync_dist is False

    def test_sync_dist_true(self) -> None:
        model = MCL(input_dim=1, sync_dist=True)
        assert model._sync_dist is True


class TestFCNEncoder:
    """FCNEncoder builds conv blocks from tuple parameters using singular names."""

    def test_default_architecture_layer_count(self) -> None:
        encoder = FCNEncoder(input_dim=1, representation_dim=128)
        # 3 conv blocks * 3 layers each (Conv, BN, ReLU) + AdaptiveAvgPool + Flatten + Linear = 12
        assert len(encoder.layers) == 12

    def test_default_output_shape(self) -> None:
        encoder = FCNEncoder(input_dim=1, representation_dim=128)
        x = torch.randn(2, 50, 1)  # (B, T, C) with T=50, C=1
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_custom_channels(self) -> None:
        encoder = FCNEncoder(
            input_dim=1,
            representation_dim=256,
            encoder_channels=(64, 128, 64),
            encoder_kernels=(5, 3, 3),
            encoder_dilations=(1, 2, 4),
        )
        x = torch.randn(2, 50, 1)  # (B, T, C) with T=50, C=1
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_two_block_encoder(self) -> None:
        encoder = FCNEncoder(
            input_dim=3,
            representation_dim=128,
            encoder_channels=(64, 128),
            encoder_kernels=(5, 3),
            encoder_dilations=(1, 2),
        )
        # 2 conv blocks * 3 layers + AdaptiveAvgPool + Flatten + Linear = 9
        assert len(encoder.layers) == 9
        x = torch.randn(2, 100, 3)  # (B, T, C) with T=100, C=3
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_fcn_encoder_accepts_btc_and_is_transpose_sensitive(self) -> None:
        """FCNEncoder must accept (B, T, C) input with T != C and return (B, representation_dim).

        Regression test: without the transpose(1, 2) inside forward(), Conv1d
        sees T channels instead of input_dim and raises RuntimeError.
        """
        encoder = FCNEncoder(input_dim=3, representation_dim=128)
        x = torch.randn(4, 50, 3)  # (B, T, C) with T=50 != C=3
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_default_padding_matches_original(self) -> None:
        """Default encoder should produce identical padding to the original hardcoded version.

        Original: Conv1d(in, 128, k=7, padding=6, d=2) -> k//2*d = 7//2*2 = 6
        Original: Conv1d(128, 256, k=5, padding=8, d=4) -> k//2*d = 5//2*4 = 8
        Original: Conv1d(256, 128, k=3, padding=8, d=8) -> k//2*d = 3//2*8 = 8
        """
        encoder = FCNEncoder(input_dim=1, representation_dim=128)
        conv_layers = [m for m in encoder.layers if isinstance(m, nn.Conv1d)]
        assert len(conv_layers) == 3
        assert conv_layers[0].padding == (6,)
        assert conv_layers[1].padding == (8,)
        assert conv_layers[2].padding == (8,)

    def test_encoder_stores_representation_dim(self) -> None:
        """FCNEncoder stores representation_dim as an attribute."""
        encoder = FCNEncoder(input_dim=1, representation_dim=64)
        assert encoder.representation_dim == 64


class TestProjectionHead:
    """Projection head uses configurable projection_dim (singular)."""

    def test_default_projection_dim(self) -> None:
        model = MCL(input_dim=1)
        lin_layers = [m for m in model.proj_head if isinstance(m, nn.Linear)]
        assert len(lin_layers) == 2
        # representation_dim=128 -> projection_dim=128 -> projection_dim=128
        assert lin_layers[0].in_features == 128
        assert lin_layers[0].out_features == 128
        assert lin_layers[1].in_features == 128
        assert lin_layers[1].out_features == 128

    def test_custom_projection_dim(self) -> None:
        model = MCL(input_dim=1, projection_dim=64)
        lin_layers = [m for m in model.proj_head if isinstance(m, nn.Linear)]
        assert lin_layers[0].out_features == 64
        assert lin_layers[1].in_features == 64
        assert lin_layers[1].out_features == 64

    def test_projection_head_in_features_matches_representation_dim(self) -> None:
        """First Linear layer in proj_head receives representation_dim as input."""
        model = MCL(input_dim=1, representation_dim=256, projection_dim=64)
        lin_layers = [m for m in model.proj_head if isinstance(m, nn.Linear)]
        assert lin_layers[0].in_features == 256
        assert lin_layers[0].out_features == 64
