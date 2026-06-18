"""Tests for Series2Vec config defaults.

Verifies that Series2VecModelParameters can be instantiated with only
input_dims and that all five formerly-required fields carry sensible
defaults taken from the reference repository.
"""

import pytest

from chronocratic.models.convolutional.standard.series2vec.config import Series2VecModelParameters
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


class TestSeries2VecConfigDefaults:
    """Series2VecModelParameters default value tests."""

    def test_instantiate_with_only_input_dims(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg is not None

    def test_embedding_dims_default(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg.embedding_dims == 16

    def test_num_heads_default(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg.num_heads == 8

    def test_feedforward_dims_default(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg.feedforward_dims == 256

    def test_representation_dims_default(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg.representation_dims == 320

    def test_dropout_rate_default(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        assert cfg.dropout_rate == 0.01


class TestSeries2VecModelInstantiation:
    """Series2Vec model integrates with config defaults."""

    def test_model_instantiates_with_config_defaults(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        model = Series2Vec(**vars(cfg))
        assert model is not None

    def test_model_representation_dim(self) -> None:
        cfg = Series2VecModelParameters(input_dims=1)
        model = Series2Vec(**vars(cfg))
        assert model.representation_dim == 2 * cfg.representation_dims
