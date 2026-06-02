"""Tests for model configuration dataclasses.

Verifies instantiation, field defaults, and vars() unpacking for all
config dataclasses.
"""

from dataclasses import fields, is_dataclass

import pytest

from tscollection.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from tscollection.models.convolutional.dilated.cost.config import CoSTModelParameters
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode
from tscollection.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters


class TestTS2VecModelParameters:
    """Test TS2VecModelParameters fields and defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(TS2VecModelParameters)

    def test_requires_only_input_dims(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.input_dims == 1

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

    def test_encoder_field_defaults(self) -> None:
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

    def test_field_count(self) -> None:
        assert len(fields(TS2VecModelParameters)) == 11


class TestCoSTModelParameters:
    """Test CoSTModelParameters fields and defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(CoSTModelParameters)

    def test_requires_input_dims_and_sequence_length(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.input_dims == 1
        assert params.sequence_length == 100

    def test_missing_input_dims_raises(self) -> None:
        with pytest.raises(TypeError):
            CoSTModelParameters(sequence_length=100)  # type: ignore[call-arg]

    def test_missing_sequence_length_raises(self) -> None:
        with pytest.raises(TypeError):
            CoSTModelParameters(input_dims=1)  # type: ignore[call-arg]

    def test_default_kernel_sizes(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.kernel_sizes == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_default_kernel_sizes_isolation(self) -> None:
        p1 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p2 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p1.kernel_sizes.append(256)
        assert 256 not in p2.kernel_sizes

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

    def test_field_count(self) -> None:
        assert len(fields(CoSTModelParameters)) == 15


class TestAutoTCLModelParameters:
    """Test AutoTCLModelParameters fields and defaults."""

    def test_is_dataclass(self) -> None:
        assert is_dataclass(AutoTCLModelParameters)

    def test_requires_only_input_dims(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.input_dims == 1

    def test_default_kernel_sizes(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.kernel_sizes == [3, 5, 7]

    def test_default_kernel_sizes_isolation(self) -> None:
        p1 = AutoTCLModelParameters(input_dims=1)
        p2 = AutoTCLModelParameters(input_dims=1)
        p1.kernel_sizes.append(9)
        assert 9 not in p2.kernel_sizes

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

    def test_encoder_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3

    def test_meta_learning_rate_default(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.meta_learning_rate == 1e-2

    def test_local_loss_weight_default(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.local_loss_weight == 0.1

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
            'meta_learning_rate',
            'local_loss_weight',
            'sync_dist',
        }
        assert set(result.keys()) == expected_keys

    def test_field_count(self) -> None:
        assert len(fields(AutoTCLModelParameters)) == 13


class TestNoModelParameters:
    """ModelParameters has been removed."""

    def test_import_fails(self) -> None:
        with pytest.raises(ImportError):
            from tscollection.models.config import ModelParameters  # noqa: F401, PLC0415
