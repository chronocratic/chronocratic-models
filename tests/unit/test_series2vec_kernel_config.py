"""Tests for Series2Vec kernel configuration parameter flow.

Verifies that temporal_kernel_size, spatial_kernel_size, and
representation_kernel_size flow correctly from config through network
to encoder.

Requirements: D-18
"""

import pytest

from chronocratic.models.convolutional.standard.series2vec.config import Series2VecModelParameters
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec


class TestConfigHasTemporalKernelSize:
    """Series2VecModelParameters has temporal_kernel_size field (D-18)."""

    def test_config_has_temporal_kernel_size(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert hasattr(cfg, "temporal_kernel_size")

    def test_config_has_spatial_kernel_size(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert hasattr(cfg, "spatial_kernel_size")

    def test_config_has_representation_kernel_size(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert hasattr(cfg, "representation_kernel_size")


class TestConfigDefaults:
    """Default values for kernel parameters (D-18)."""

    def test_temporal_kernel_size_default(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert cfg.temporal_kernel_size == 8

    def test_spatial_kernel_size_default(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert cfg.spatial_kernel_size is None

    def test_representation_kernel_size_default(self) -> None:
        cfg = Series2VecModelParameters(input_dim=2)
        assert cfg.representation_kernel_size == 3


class TestKernelFlowConfigToEncoder:
    """Creating Series2Vec with custom kernels passes them to DisjoinEncoder (D-18)."""

    def test_custom_temporal_kernel_reaches_encoder(self) -> None:
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            representation_dim=16,
            temporal_kernel_size=5,
            num_heads=2,
            feedforward_dim=32,
        )
        # The encoder's temporal_CNN should use kernel_size=5
        actual_kernel = model.network.embed_layer.temporal_CNN[0].kernel_size[1]
        assert actual_kernel == 5

    def test_custom_spatial_kernel_reaches_encoder(self) -> None:
        model = Series2Vec(
            input_dim=4,
            embedding_dim=8,
            representation_dim=16,
            spatial_kernel_size=2,
            temporal_kernel_size=4,
            num_heads=2,
            feedforward_dim=32,
        )
        # The encoder's spatial_CNN should use kernel_size=(2, 1)
        actual_kernel = model.network.embed_layer.spatial_CNN[0].kernel_size[0]
        assert actual_kernel == 2

    def test_custom_representation_kernel_reaches_encoder(self) -> None:
        model = Series2Vec(
            input_dim=2,
            embedding_dim=8,
            representation_dim=16,
            representation_kernel_size=5,
            temporal_kernel_size=4,
            num_heads=2,
            feedforward_dim=32,
        )
        # The encoder's rep_CNN should use kernel_size=5
        actual_kernel = model.network.embed_layer.rep_CNN[0].kernel_size[0]
        assert actual_kernel == 5


class TestSpatialKernelValidates:
    """spatial_kernel_size > input_dim raises ValueError (D-03)."""

    def test_spatial_kernel_exceeds_input_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            Series2Vec(
                input_dim=2,
                embedding_dim=8,
                representation_dim=16,
                spatial_kernel_size=5,
                temporal_kernel_size=4,
                num_heads=2,
                feedforward_dim=32,
            )

    def test_spatial_kernel_equal_to_input_dim_ok(self) -> None:
        # Should not raise — equal is fine
        model = Series2Vec(
            input_dim=4,
            embedding_dim=8,
            representation_dim=16,
            spatial_kernel_size=4,
            temporal_kernel_size=4,
            num_heads=2,
            feedforward_dim=32,
        )
        assert model is not None
