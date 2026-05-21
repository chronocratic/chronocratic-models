"""Tests for model configuration dataclasses.

Verifies instantiation, inheritance hierarchy, field defaults, and
vars() unpacking for all config dataclasses.
"""

from abc import ABC
from dataclasses import fields, is_dataclass

import pytest

from tscollection.models.config import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    DilatedCNNModelParameters,
    ModelParameters,
    TS2VecModelParameters,
)
from tscollection.models.encoders.masking import MaskMode


class TestModelParametersBase:
    """Test ModelParameters is an ABC with no fields."""

    def test_is_abc(self) -> None:
        assert issubclass(ModelParameters, ABC)

    def test_is_dataclass(self) -> None:
        assert is_dataclass(ModelParameters)

    def test_no_fields(self) -> None:
        assert len(fields(ModelParameters)) == 0

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ModelParameters()  # type: ignore[call-arg]


class TestDilatedCNNModelParameters:
    """Test DilatedCNNModelParameters base class."""

    def test_requires_input_dims(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.input_dims == 1

    def test_default_hidden_dims(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.hidden_dims == 64

    def test_default_output_dims(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.output_dims == 320

    def test_default_depth(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.depth == 10

    def test_default_dropout_rate(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.dropout_rate == 0.1

    def test_default_conv_kernel_size(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.conv_kernel_size == 3

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(DilatedCNNModelParameters, ModelParameters)


class TestTS2VecModelParameters:
    """Test TS2VecModelParameters inherits from DilatedCNNModelParameters."""

    def test_requires_only_input_dims(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.input_dims == 1

    def test_inherits_from_dilated_cnn(self) -> None:
        assert issubclass(TS2VecModelParameters, DilatedCNNModelParameters)

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(TS2VecModelParameters, ModelParameters)

    def test_default_mask_mode(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.mask_mode == MaskMode.BINOMIAL

    def test_default_learning_rate(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.learning_rate == 1e-3

    def test_default_max_train_length(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.max_train_length is None

    def test_default_temporal_unit(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.temporal_unit == 0

    def test_default_sync_dist(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.sync_dist is False

    def test_inherited_defaults(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3

    def test_vars_produces_correct_keys(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        result = vars(params)
        expected_keys = {
            'input_dims',
            'hidden_dims',
            'output_dims',
            'depth',
            'dropout_rate',
            'conv_kernel_size',
            'mask_mode',
            'learning_rate',
            'max_train_length',
            'temporal_unit',
            'sync_dist',
        }
        assert set(result.keys()) == expected_keys


class TestCoSTModelParameters:
    """Test CoSTModelParameters inherits directly from ModelParameters."""

    def test_requires_input_dims_and_sequence_length(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.input_dims == 1
        assert params.sequence_length == 100

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(CoSTModelParameters, ModelParameters)

    def test_not_subclass_of_dilated_cnn(self) -> None:
        assert not issubclass(CoSTModelParameters, DilatedCNNModelParameters)

    def test_default_kernel_sizes(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.kernel_sizes == []

    def test_default_kernel_sizes_isolation(self) -> None:
        p1 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p2 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p1.kernel_sizes.append(2)
        assert p2.kernel_sizes == []

    def test_default_max_train_length(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.max_train_length == 201

    def test_default_hidden_dims(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.hidden_dims == 64

    def test_default_output_dims(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.output_dims == 320

    def test_default_depth(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.depth == 10

    def test_default_dropout_rate(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.dropout_rate == 0.1

    def test_default_mask_mode(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.mask_mode == MaskMode.BINOMIAL

    def test_default_learning_rate(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.learning_rate == 1e-3

    def test_default_seasonal_loss_weight(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.seasonal_loss_weight == 0.1

    def test_default_queue_size(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.queue_size == 65536

    def test_default_momentum(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.momentum == 0.999

    def test_default_temperature(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.temperature == 0.07

    def test_default_sync_dist(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.sync_dist is False

    def test_vars_produces_correct_keys(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        result = vars(params)
        expected_keys = {
            'input_dims',
            'sequence_length',
            'kernel_sizes',
            'max_train_length',
            'hidden_dims',
            'output_dims',
            'depth',
            'dropout_rate',
            'mask_mode',
            'learning_rate',
            'seasonal_loss_weight',
            'queue_size',
            'momentum',
            'temperature',
            'sync_dist',
        }
        assert set(result.keys()) == expected_keys


class TestAutoTCLModelParameters:
    """Test AutoTCLModelParameters inherits from DilatedCNNModelParameters."""

    def test_requires_only_input_dims(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.input_dims == 1

    def test_inherits_from_dilated_cnn(self) -> None:
        assert issubclass(AutoTCLModelParameters, DilatedCNNModelParameters)

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(AutoTCLModelParameters, ModelParameters)

    def test_default_kernel_sizes(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.kernel_sizes == []

    def test_default_kernel_sizes_isolation(self) -> None:
        p1 = AutoTCLModelParameters(input_dims=1)
        p2 = AutoTCLModelParameters(input_dims=1)
        p1.kernel_sizes.append(2)
        assert p2.kernel_sizes == []

    def test_default_mask_mode(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.mask_mode == MaskMode.BINOMIAL

    def test_default_learning_rate(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.learning_rate == 1e-3

    def test_default_max_train_length(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.max_train_length is None

    def test_default_sync_dist(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.sync_dist is False

    def test_inherited_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3

    def test_vars_produces_correct_keys(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        result = vars(params)
        expected_keys = {
            'input_dims',
            'kernel_sizes',
            'hidden_dims',
            'output_dims',
            'depth',
            'dropout_rate',
            'conv_kernel_size',
            'mask_mode',
            'learning_rate',
            'max_train_length',
            'sync_dist',
        }
        assert set(result.keys()) == expected_keys


class TestAllExports:
    """Test that __all__ exposes all expected classes."""

    def test_all_exports(self) -> None:
        from tscollection.models import config

        expected = {
            'AutoTCLModelParameters',
            'CoSTModelParameters',
            'DilatedCNNModelParameters',
            'ModelParameters',
            'TS2VecModelParameters',
        }
        assert set(config.__all__) == expected
