"""Tests for augmentation parameter config dataclasses.

Verifies that CropShiftAugmentationParameters,
CosTRandomFunctionAugmentationParameters, and
AutoTCLNeuralNetworkAugmentationParameters instantiate correctly
with proper defaults and field types (CFG-02).
"""

import pytest

from tscollection.models.augmentation.config import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
    CropShiftAugmentationParameters,
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

    def test_sigma_required(self) -> None:
        with pytest.raises(TypeError):
            CosTRandomFunctionAugmentationParameters()  # type: ignore[call-arg]

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

    def test_output_dims_required(self) -> None:
        with pytest.raises(TypeError):
            AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)  # type: ignore[call-arg]

    def test_default_fields(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        assert params.kernel_sizes == []
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

    def test_import_from_config_module(self) -> None:
        from tscollection.models.augmentation.config import (  # noqa: F401
            AutoTCLNeuralNetworkAugmentationParameters,
            CosTRandomFunctionAugmentationParameters,
            CropShiftAugmentationParameters,
        )

    def test_import_from_barrel(self) -> None:
        from tscollection.models.augmentation import (  # noqa: F401
            AutoTCLNeuralNetworkAugmentationParameters,
            CosTRandomFunctionAugmentationParameters,
            CropShiftAugmentationParameters,
        )
