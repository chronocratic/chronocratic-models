"""Tests for TimeNet canonical parameter names (input_dim, hidden_dim).

Verifies that the config and model use singular dimension names (D-01)
and that the model exposes a representation_dim property (D-03).
"""

from __future__ import annotations

import torch

import pytest

from chronocratic.models.recurrent.timenet.config import TimeNetModelParameters
from chronocratic.models.recurrent.timenet.model import TimeNet


class TestTimeNetConfigRename:
    """TimeNetModelParameters uses singular dimension names."""

    def test_config_uses_canonical_names(self) -> None:
        """Config should accept input_dim, hidden_dim (singular)."""
        config = TimeNetModelParameters(hidden_dim=64, depth=1, input_dim=1, dropout_rate=0.1)
        assert config.input_dim == 1
        assert config.depth == 1
        assert config.dropout_rate == 0.1
        assert config.hidden_dim == 64

    def test_config_defaults(self) -> None:
        """Defaults: hidden_dim=64, depth=3, input_dim required, dropout_rate=0.4, learning_rate=5e-3."""
        config = TimeNetModelParameters(input_dim=1)
        assert config.input_dim == 1
        assert config.dropout_rate == 0.4
        assert config.hidden_dim == 64
        assert config.depth == 3
        assert config.learning_rate == 5e-3


class TestTimeNetVarsContract:
    """vars(config) should pass directly to TimeNet.__init__."""

    def test_vars_passesthrough(self) -> None:
        """TimeNet(**vars(config)) must not raise TypeError."""
        config = TimeNetModelParameters(input_dim=1, hidden_dim=64, depth=1)
        model = TimeNet(**vars(config))
        assert model is not None


class TestTimeNetInternalAttrs:
    """Model internal attributes use singular dimension names."""

    def test_internal_depth_and_dropout_rate(self) -> None:
        """model._depth and model._dropout_rate are set correctly."""
        model = TimeNet(hidden_dim=64, depth=2, input_dim=1, dropout_rate=0.2)
        assert model._depth == 2
        assert model._dropout_rate == 0.2
        assert model._input_dim == 1

    def test_internal_hidden_dim(self) -> None:
        """model._hidden_dim matches constructor argument."""
        model = TimeNet(hidden_dim=128, depth=1, input_dim=1)
        assert model._hidden_dim == 128


class TestTimeNetRepresentationDim:
    """TimeNet exposes representation_dim property returning hidden_dim (D-03)."""

    def test_representation_dim_returns_hidden_dim(self) -> None:
        """representation_dim should equal hidden_dim."""
        model = TimeNet(hidden_dim=96, depth=1, input_dim=1)
        assert model.representation_dim == 96

    def test_representation_dim_matches_encode_output_vector(self) -> None:
        """representation_dim should equal the feature dim of encode() VECTOR output."""
        hidden = 48
        model = TimeNet(hidden_dim=hidden, depth=1, input_dim=1)
        batch = torch.randn(2, 10, 1)  # (B, T, C)
        encoded = model.encode(batch)
        # encode VECTOR shape: (B, D)
        assert encoded.shape[-1] == model.representation_dim


class TestTimeNetModelConstructor:
    """TimeNet.__init__ accepts singular dimension names."""

    def test_model_accepts_singular_params(self) -> None:
        """Model should accept input_dim and hidden_dim."""
        model = TimeNet(input_dim=3, hidden_dim=32, depth=2, dropout_rate=0.1)
        assert model._input_dim == 3
        assert model._hidden_dim == 32
