"""Tests for per-model config hierarchy (Phase 4, Plan 2).

Verifies that config classes live in their own module files with
correct tiered inheritance:
    - dilated/config.py: DilatedCNNModelParameters
    - ts2vec/config.py: TS2VecModelParameters -> DilatedCNNModelParameters
    - cost/config.py: CoSTModelParameters -> ModelParameters (NOT DilatedCNN)
    - autotcl/config.py: AutoTCLModelParameters -> DilatedCNNModelParameters
"""

import pytest

from tscollection.models.config import ModelParameters
from tscollection.models.convolutional.dilated.config import DilatedCNNModelParameters
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode
from tscollection.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters
from tscollection.models.convolutional.dilated.cost.config import CoSTModelParameters
from tscollection.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters


class TestDilatedCNNModelParametersLocation:
    """DilatedCNNModelParameters lives in dilated/config.py."""

    def test_module_location(self) -> None:
        assert (
            DilatedCNNModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.config'
        )

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(DilatedCNNModelParameters, ModelParameters)


class TestTS2VecModelParametersLocation:
    """TS2VecModelParameters lives in ts2vec/config.py."""

    def test_module_location(self) -> None:
        assert (
            TS2VecModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.ts2vec.config'
        )

    def test_inherits_from_dilated_cnn(self) -> None:
        assert issubclass(TS2VecModelParameters, DilatedCNNModelParameters)

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(TS2VecModelParameters, ModelParameters)


class TestCoSTModelParametersLocation:
    """CoSTModelParameters lives in cost/config.py."""

    def test_module_location(self) -> None:
        assert (
            CoSTModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.cost.config'
        )

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(CoSTModelParameters, ModelParameters)

    def test_not_subclass_of_dilated_cnn(self) -> None:
        assert not issubclass(CoSTModelParameters, DilatedCNNModelParameters)


class TestAutoTCLModelParametersLocation:
    """AutoTCLModelParameters lives in autotcl/config.py."""

    def test_module_location(self) -> None:
        assert (
            AutoTCLModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.autotcl.config'
        )

    def test_inherits_from_dilated_cnn(self) -> None:
        assert issubclass(AutoTCLModelParameters, DilatedCNNModelParameters)

    def test_inherits_from_model_parameters(self) -> None:
        assert issubclass(AutoTCLModelParameters, ModelParameters)


class TestDilatedCNNConfigDefaults:
    """DilatedCNNModelParameters field defaults match source."""

    def test_field_defaults(self) -> None:
        params = DilatedCNNModelParameters(input_dims=1)
        assert params.input_dims == 1
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3


class TestTS2VecConfigDefaults:
    """TS2VecModelParameters field defaults match source."""

    def test_own_field_defaults(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.mask_mode == MaskMode.BINOMIAL
        assert params.learning_rate == 1e-3
        assert params.max_train_length is None
        assert params.temporal_unit == 0
        assert params.sync_dist is False

    def test_inherited_field_defaults(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3


class TestCoSTConfigDefaults:
    """CoSTModelParameters field defaults match source."""

    def test_required_fields(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.input_dims == 1
        assert params.sequence_length == 100

    def test_max_train_length_is_int(self) -> None:
        """CoST max_train_length defaults to 201, not None."""
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.max_train_length == 201

    def test_kernel_sizes_isolated(self) -> None:
        """Mutable default (list) must not be shared across instances."""
        p1 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p2 = CoSTModelParameters(input_dims=1, sequence_length=100)
        p1.kernel_sizes.append(2)
        assert p2.kernel_sizes == []

    def test_other_defaults(self) -> None:
        params = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.mask_mode == MaskMode.BINOMIAL
        assert params.learning_rate == 1e-3
        assert params.seasonal_loss_weight == 0.1
        assert params.queue_size == 65536
        assert params.momentum == 0.999
        assert params.temperature == 0.07
        assert params.sync_dist is False


class TestAutoTCLConfigDefaults:
    """AutoTCLModelParameters field defaults match source."""

    def test_own_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.kernel_sizes == []
        assert params.mask_mode == MaskMode.BINOMIAL
        assert params.learning_rate == 1e-3
        assert params.max_train_length is None
        assert params.sync_dist is False

    def test_kernel_sizes_isolated(self) -> None:
        """Mutable default (list) must not be shared across instances."""
        p1 = AutoTCLModelParameters(input_dims=1)
        p2 = AutoTCLModelParameters(input_dims=1)
        p1.kernel_sizes.append(2)
        assert p2.kernel_sizes == []

    def test_inherited_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3


class TestConfigAllExports:
    """Each config module exposes its class via __all__."""

    def test_dilated_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.config as mod

        assert 'DilatedCNNModelParameters' in mod.__all__

    def test_ts2vec_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.ts2vec.config as mod

        assert 'TS2VecModelParameters' in mod.__all__

    def test_cost_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.cost.config as mod

        assert 'CoSTModelParameters' in mod.__all__

    def test_autotcl_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.autotcl.config as mod

        assert 'AutoTCLModelParameters' in mod.__all__
