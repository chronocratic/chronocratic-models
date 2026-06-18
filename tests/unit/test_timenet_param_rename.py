"""Tests for TimeNet canonical parameter names (input_dims, depth, dropout_rate).

Verifies that the config and model use the renamed fields after D-02.
"""

from __future__ import annotations

import pytest

from chronocratic.models.recurrent.timenet.config import TimeNetModelParameters
from chronocratic.models.recurrent.timenet.model import TimeNet


class TestTimeNetConfigRename:
    """TimeNetModelParameters uses canonical names."""

    def test_config_uses_canonical_names(self) -> None:
        """Config should accept input_dims, depth, dropout_rate."""
        config = TimeNetModelParameters(
            hidden_dims=64, depth=1, input_dims=1, dropout_rate=0.1
        )
        assert config.input_dims == 1
        assert config.depth == 1
        assert config.dropout_rate == 0.1
        assert config.hidden_dims == 64

    def test_config_defaults(self) -> None:
        """input_dims defaults to 1, dropout_rate defaults to 0.1."""
        config = TimeNetModelParameters(hidden_dims=64, depth=1)
        assert config.input_dims == 1
        assert config.dropout_rate == 0.1
        assert config.hidden_dims == 64
        assert config.depth == 1


class TestTimeNetVarsContract:
    """vars(config) should pass directly to TimeNet.__init__."""

    def test_vars_passesthrough(self) -> None:
        """TimeNet(**vars(config)) must not raise TypeError."""
        config = TimeNetModelParameters(hidden_dims=64, depth=1)
        model = TimeNet(**vars(config))
        assert model is not None


class TestTimeNetInternalAttrs:
    """Model internal attributes use canonical names."""

    def test_internal_depth_and_dropout_rate(self) -> None:
        """model._depth and model._dropout_rate are set correctly."""
        model = TimeNet(hidden_dims=64, depth=2, input_dims=1, dropout_rate=0.2)
        assert model._depth == 2
        assert model._dropout_rate == 0.2
        assert model._input_dims == 1
