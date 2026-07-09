"""Tests for RecurrentAE dimension params and representation_dim property.

Verifies:
- Config uses input_dim (singular), not input_dims.
- Model accepts input_dim (singular) kwarg.
- Model exposes representation_dim property returning layers[-1].
- layers tuple preserved unchanged.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.recurrent.recurrentae.config import RecurrentAutoEncoderModelParameters
from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder


class TestConfigInputDimRename:
    """Config dataclass uses input_dim (singular)."""

    def test_config_has_input_dim_field(self) -> None:
        """RecurrentAutoEncoderModelParameters has input_dim field."""
        config = RecurrentAutoEncoderModelParameters(input_dim=3)
        assert config.input_dim == 3

    def test_config_does_not_have_input_dims(self) -> None:
        """Old plural field name input_dims should not exist."""
        with pytest.raises(TypeError):
            RecurrentAutoEncoderModelParameters(input_dims=3)

    def test_config_layers_preserved(self) -> None:
        """layers tuple field stays unchanged."""
        config = RecurrentAutoEncoderModelParameters(input_dim=3, layers=(64, 32, 16))
        assert config.layers == (64, 32, 16)

    def test_config_default_layers(self) -> None:
        """Default layers value is (16, 8)."""
        config = RecurrentAutoEncoderModelParameters(input_dim=5)
        assert config.layers == (16, 8)


class TestModelInputDimRename:
    """Model __init__ accepts input_dim (singular)."""

    def test_model_accepts_input_dim(self) -> None:
        """RecurrentAutoEncoder constructor accepts input_dim."""
        model = RecurrentAutoEncoder(input_dim=3, layers=(8,))
        assert isinstance(model, RecurrentAutoEncoder)

    def test_model_does_not_accept_input_dims(self) -> None:
        """Old plural param name input_dims should not be accepted."""
        with pytest.raises(TypeError):
            RecurrentAutoEncoder(input_dims=3, layers=(8,))

    def test_input_dim_stored_correctly(self) -> None:
        """Model stores input_dim value correctly."""
        model = RecurrentAutoEncoder(input_dim=5, layers=(16,))
        # Check the stored value via the private attribute
        assert model._input_dim == 5

    def test_encoder_output_shape_uses_input_dim(self) -> None:
        """Encoder output feature dim matches layers[-1], decoder reconstructs input_dim."""
        model = RecurrentAutoEncoder(input_dim=7, layers=(32, 16))
        x = torch.randn(2, 10, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestRepresentationDimProperty:
    """representation_dim property returns layers[-1]."""

    def test_representation_dim_single_layer(self) -> None:
        """representation_dim equals the single layer size."""
        model = RecurrentAutoEncoder(input_dim=3, layers=(8,))
        assert model.representation_dim == 8

    def test_representation_dim_multi_layer(self) -> None:
        """representation_dim equals the last layer size."""
        model = RecurrentAutoEncoder(input_dim=3, layers=(64, 32, 16))
        assert model.representation_dim == 16

    def test_representation_dim_default_layers(self) -> None:
        """representation_dim equals 8 with default layers (16, 8)."""
        model = RecurrentAutoEncoder(input_dim=3)
        assert model.representation_dim == 8

    def test_representation_dim_matches_encode_output(self) -> None:
        """representation_dim matches the feature dim of encode() output."""
        model = RecurrentAutoEncoder(input_dim=5, layers=(32, 16))
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            encoded = model.encode(x, batch_size=2, num_workers=0)
        assert encoded.shape[-1] == model.representation_dim

    def test_representation_dim_matches_encoder_output_feature_dim(self) -> None:
        """representation_dim matches the encoder's output feature dimension."""
        model = RecurrentAutoEncoder(input_dim=5, layers=(32, 16))
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            enc_out = model.encoder(x)
        assert enc_out.shape[2] == model.representation_dim


class TestLayerBuilderFunctions:
    """Layer builder functions use input_dim param."""

    def test_build_encoder_uses_input_dim(self) -> None:
        """_build_encoder accepts input_dim parameter name."""
        from chronocratic.models.recurrent.recurrentae.layers import (
            _build_encoder,
            _prepare_dropout,
        )

        dropout = _prepare_dropout(0.0, 1)
        encoder = _build_encoder(rnn_cls=torch.nn.LSTM, input_dim=3, layers=(8,), dropout=dropout)
        assert isinstance(encoder, torch.nn.Sequential)

    def test_build_decoder_uses_input_dim(self) -> None:
        """_build_decoder accepts input_dim parameter name."""
        from chronocratic.models.recurrent.recurrentae.layers import (
            _build_decoder,
            _prepare_dropout,
        )

        dropout = _prepare_dropout(0.0, 1)
        decoder = _build_decoder(rnn_cls=torch.nn.LSTM, input_dim=3, layers=(8,), dropout=dropout)
        assert isinstance(decoder, torch.nn.Sequential)
