"""Tests for augmentation parameter config dataclasses.

Verifies that CropShiftAugmentationParameters,
CosTRandomFunctionAugmentationParameters, and
AutoTCLNeuralNetworkAugmentationParameters instantiate correctly
with proper defaults and field types (CFG-02).

Also tests primitive parameter dataclasses (JitterParameters, ScalingParameters,
PermutationParameters) from augmentation/primitives.py.
"""

import pytest

from tscollection.models.augmentation import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
    CropShiftAugmentationParameters,
    JitterParameters,
    PermutationParameters,
    ScalingParameters,
)

# --------------------------------------------------------------------------- #
# CropShiftAugmentationParameters
# --------------------------------------------------------------------------- #


class TestCropShiftAugmentationParameters:
    """CropShiftAugmentationParameters defaults and fields."""

    def test_default_temporal_unit(self) -> None:
        params = CropShiftAugmentationParameters()
        assert params.temporal_unit == 0

    def test_custom_temporal_unit(self) -> None:
        params = CropShiftAugmentationParameters(temporal_unit=3)
        assert params.temporal_unit == 3

    def test_vars_produces_expected_keys(self) -> None:
        params = CropShiftAugmentationParameters()
        d = vars(params)
        assert 'temporal_unit' in d


# --------------------------------------------------------------------------- #
# CosTRandomFunctionAugmentationParameters
# --------------------------------------------------------------------------- #


class TestCosTRandomFunctionAugmentationParameters:
    """CosTRandomFunctionAugmentationParameters defaults and fields."""

    def test_sigma_has_default(self) -> None:
        params = CosTRandomFunctionAugmentationParameters()
        assert params.sigma == 0.1

    def test_sigma_and_default_p(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        assert params.sigma == 0.1
        assert params.p == 0.5

    def test_custom_sigma_and_p(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.2, p=0.8)
        assert params.sigma == 0.2
        assert params.p == 0.8

    def test_vars_produces_expected_keys(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        d = vars(params)
        assert 'sigma' in d
        assert 'p' in d


# --------------------------------------------------------------------------- #
# AutoTCLNeuralNetworkAugmentationParameters
# --------------------------------------------------------------------------- #


class TestAutoTCLNeuralNetworkAugmentationParameters:
    """AutoTCLNeuralNetworkAugmentationParameters fields and defaults."""

    def test_required_fields(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        assert params.input_dims == 1
        assert params.output_dims == 320

    def test_input_dims_required(self) -> None:
        with pytest.raises(TypeError):
            AutoTCLNeuralNetworkAugmentationParameters(output_dims=320)  # type: ignore[call-arg]

    def test_output_dims_has_default(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        assert params.output_dims == 16

    def test_default_fields(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        assert params.kernel_sizes == [3, 5, 7]
        assert params.hidden_dims == 64
        assert params.feature_extractor_depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3
        assert params.num_augmentation_channels == 1
        assert params.gumbel_bias == 0.001
        assert params.zeta == 1.0
        assert params.gamma_zeta == 0.05
        assert params.hard_mask is True

    def test_vars_produces_encoder_kwargs(self) -> None:
        """vars() output matches AutoTCLAugmentationTimeSeriesEncoder constructor."""
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        d = vars(params)
        expected_keys = {
            'input_dims', 'output_dims', 'kernel_sizes', 'hidden_dims',
            'feature_extractor_depth', 'dropout_rate', 'conv_kernel_size',
            'mask_mode', 'num_augmentation_channels', 'gumbel_bias',
            'zeta', 'gamma_zeta', 'hard_mask',
        }
        assert set(d.keys()) == expected_keys


# --------------------------------------------------------------------------- #
# Import path resolution
# --------------------------------------------------------------------------- #


class TestConfigImports:
    """Verify config dataclasses are importable from expected paths."""

    def test_import_from_barrel(self) -> None:
        from tscollection.models.augmentation import (  # noqa: F401
            AutoTCLNeuralNetworkAugmentationParameters,
            CosTRandomFunctionAugmentationParameters,
            CropShiftAugmentationParameters,
        )


# --------------------------------------------------------------------------- #
# Primitive parameter dataclasses
# --------------------------------------------------------------------------- #


class TestJitterParameters:
    """JitterParameters defaults and fields."""

    def test_default_sigma(self) -> None:
        params = JitterParameters()
        assert params.sigma == 0.1

    def test_default_p(self) -> None:
        params = JitterParameters()
        assert params.p == 1.0

    def test_custom_sigma(self) -> None:
        params = JitterParameters(sigma=0.5)
        assert params.sigma == 0.5

    def test_custom_p(self) -> None:
        params = JitterParameters(p=0.3)
        assert params.p == 0.3

    def test_vars_produces_expected_keys(self) -> None:
        params = JitterParameters()
        d = vars(params)
        assert 'sigma' in d
        assert 'p' in d


class TestScalingParameters:
    """ScalingParameters defaults and fields."""

    def test_default_values(self) -> None:
        params = ScalingParameters()
        assert params.sigma == 0.1
        assert params.mean == 1.0
        assert params.p == 1.0
        assert params.per_sample is False
        assert params.channel_dim == 1

    def test_custom_sigma(self) -> None:
        params = ScalingParameters(sigma=0.5, mean=2.0, per_sample=True)
        assert params.sigma == 0.5
        assert params.mean == 2.0
        assert params.per_sample is True

    def test_vars_produces_expected_keys(self) -> None:
        params = ScalingParameters()
        d = vars(params)
        assert 'sigma' in d
        assert 'mean' in d
        assert 'p' in d
        assert 'per_sample' in d
        assert 'channel_dim' in d


class TestPermutationParameters:
    """PermutationParameters defaults and fields."""

    def test_default_values(self) -> None:
        params = PermutationParameters()
        assert params.max_segments == 5
        assert params.time_dim == -1

    def test_custom_max_segments(self) -> None:
        params = PermutationParameters(max_segments=10)
        assert params.max_segments == 10

    def test_vars_produces_expected_keys(self) -> None:
        params = PermutationParameters()
        d = vars(params)
        assert 'max_segments' in d
        assert 'time_dim' in d


class TestPrimitiveImports:
    """Verify primitive config dataclasses are importable from the barrel."""

    def test_import_primitive_params_from_barrel(self) -> None:
        from tscollection.models.augmentation import (  # noqa: F401
            JitterParameters,
            PermutationParameters,
            ScalingParameters,
        )
