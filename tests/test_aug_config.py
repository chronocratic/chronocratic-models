"""Tests for augmentation parameter config dataclasses.

Verifies that CropShiftAugmentationParameters,
CosTRandomFunctionAugmentationParameters, and
AutoTCLNeuralNetworkAugmentationParameters instantiate correctly
with proper defaults and field types (CFG-02).

Also tests primitive parameter dataclasses (JitterParameters, ScalingParameters,
PermutationParameters) from augmentation/primitives.py.
"""


from chronocratic.models.augmentation import (
    JitterParameters,
    PermutationParameters,
    ScalingParameters,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
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
        params = CropShiftAugmentationParameters(temporal_unit=2)
        assert params.temporal_unit == 2

# --------------------------------------------------------------------------- #
# CosTRandomFunctionAugmentationParameters
# --------------------------------------------------------------------------- #


class TestCosTRandomFunctionAugmentationParameters:
    """CosTRandomFunctionAugmentationParameters defaults and fields."""

    def test_default_sigma(self) -> None:
        params = CosTRandomFunctionAugmentationParameters()
        assert params.sigma == 0.1

    def test_custom_sigma(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.2)
        assert params.sigma == 0.2

    def test_custom_sigma_and_p(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.2, p=0.8)
        assert params.sigma == 0.2
        assert params.p == 0.8

    def test_default_p(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        assert params.p == 0.5

# --------------------------------------------------------------------------- #
# AutoTCLNeuralNetworkAugmentationParameters
# --------------------------------------------------------------------------- #


class TestAutoTCLNeuralNetworkAugmentationParameters:
    """AutoTCLNeuralNetworkAugmentationParameters defaults and fields."""

    def test_default_encoder_kwargs(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        assert params.encoder_kwargs is None

    def test_encoder_kwargs_not_none(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, encoder_kwargs={'kernel_sizes': [3, 5]}
        )
        assert params.encoder_kwargs == {'kernel_sizes': [3, 5]}

    def test_input_dims_is_int(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        assert isinstance(params.input_dims, int)

    def test_encoder_kwargs_is_dict(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, encoder_kwargs={'kernel_sizes': [3, 5]}
        )
        assert isinstance(params.encoder_kwargs, dict)

    def test_encoder_kwargs_none_by_default(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        assert params.encoder_kwargs is None

# --------------------------------------------------------------------------- #
# JitterParameters
# --------------------------------------------------------------------------- #


class TestJitterParameters:
    """JitterParameters defaults and fields."""

    def test_default_sigma(self) -> None:
        params = JitterParameters()
        assert params.sigma == 0.1

    def test_custom_sigma(self) -> None:
        params = JitterParameters(sigma=0.1)
        assert params.sigma == 0.1

# --------------------------------------------------------------------------- #
# ScalingParameters
# --------------------------------------------------------------------------- #


class TestScalingParameters:
    """ScalingParameters defaults and fields."""

    def test_default_sigma(self) -> None:
        params = ScalingParameters()
        assert params.sigma == 0.1

    def test_custom_sigma(self) -> None:
        params = ScalingParameters(sigma=0.1)
        assert params.sigma == 0.1

# --------------------------------------------------------------------------- #
# PermutationParameters
# --------------------------------------------------------------------------- #


class TestPermutationParameters:
    """PermutationParameters defaults and fields."""

    def test_default_max_segments(self) -> None:
        params = PermutationParameters()
        assert params.max_segments == 5

    def test_custom_max_segments(self) -> None:
        params = PermutationParameters(max_segments=3)
        assert params.max_segments == 3

    def test_default_time_dim(self) -> None:
        params = PermutationParameters()
        assert params.time_dim == -1

    def test_custom_time_dim(self) -> None:
        params = PermutationParameters(max_segments=3, time_dim=1)
        assert params.time_dim == 1

# --------------------------------------------------------------------------- #
# Barrel exports
# --------------------------------------------------------------------------- #


class TestBarrelExports:
    """Verify that parameter dataclasses are importable from the barrel."""

    def test_primitive_params_importable_from_barrel(self) -> None:
        from chronocratic.models.augmentation import (
            JitterParameters,
            PermutationParameters,
            ScalingParameters,
        )

        assert JitterParameters is not None
        assert PermutationParameters is not None
        assert ScalingParameters is not None

    def test_per_model_params_importable_from_source(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentationParameters,
        )
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
        )

        assert AutoTCLNeuralNetworkAugmentationParameters is not None
        assert CosTRandomFunctionAugmentationParameters is not None
        assert CropShiftAugmentationParameters is not None

    def test_auto_tcl_params_defaults(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        assert params.input_dims == 1
        assert params.encoder_kwargs is None
