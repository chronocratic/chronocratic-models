"""Tests for per-model config locations and module exports.

Verifies that config classes live in their own module files:
    - ts2vec/config.py: TS2VecModelParameters
    - cost/config.py: CoSTModelParameters
    - autotcl/config.py: AutoTCLModelParameters
"""

import pytest

from tscollection.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from tscollection.models.convolutional.dilated.cost.config import CoSTModelParameters
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode
from tscollection.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters


class TestTS2VecModelParametersLocation:
    """TS2VecModelParameters lives in ts2vec/config.py."""

    def test_module_location(self) -> None:
        assert (
            TS2VecModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.ts2vec.config'
        )

    def test_no_parent_class(self) -> None:
        assert TS2VecModelParameters.__bases__ == (object,)


class TestCoSTModelParametersLocation:
    """CoSTModelParameters lives in cost/config.py."""

    def test_module_location(self) -> None:
        assert (
            CoSTModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.cost.config'
        )

    def test_no_parent_class(self) -> None:
        assert CoSTModelParameters.__bases__ == (object,)


class TestAutoTCLModelParametersLocation:
    """AutoTCLModelParameters lives in autotcl/config.py."""

    def test_module_location(self) -> None:
        assert (
            AutoTCLModelParameters.__module__
            == 'tscollection.models.convolutional.dilated.autotcl.config'
        )

    def test_no_parent_class(self) -> None:
        assert AutoTCLModelParameters.__bases__ == (object,)


class TestTS2VecConfigDefaults:
    """TS2VecModelParameters field defaults match model init."""

    def test_own_field_defaults(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.mask_mode == MaskMode.BINOMIAL
        assert params.learning_rate == 1e-3
        assert params.max_train_length is None
        assert params.temporal_unit == 0
        assert params.sync_dist is False

    def test_encoder_field_defaults(self) -> None:
        params = TS2VecModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3


class TestCoSTConfigDefaults:
    """CoSTModelParameters field defaults match model init."""

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
        p1.kernel_sizes.append(256)
        assert 256 not in p2.kernel_sizes

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
    """AutoTCLModelParameters field defaults match model init."""

    def test_own_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.kernel_sizes == [3, 5, 7]
        assert params.mask_mode == MaskMode.BINOMIAL
        assert params.learning_rate == 1e-3
        assert params.max_train_length is None
        assert params.sync_dist is False

    def test_kernel_sizes_isolated(self) -> None:
        """Mutable default (list) must not be shared across instances."""
        p1 = AutoTCLModelParameters(input_dims=1)
        p2 = AutoTCLModelParameters(input_dims=1)
        p1.kernel_sizes.append(9)
        assert 9 not in p2.kernel_sizes

    def test_encoder_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.hidden_dims == 64
        assert params.output_dims == 320
        assert params.depth == 10
        assert params.dropout_rate == 0.1
        assert params.conv_kernel_size == 3

    def test_training_field_defaults(self) -> None:
        params = AutoTCLModelParameters(input_dims=1)
        assert params.meta_learning_rate == 1e-2
        assert params.local_loss_weight == 0.1


class TestConfigAllExports:
    """Each config module exposes its class via __all__."""

    def test_ts2vec_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.ts2vec.config as mod  # noqa: PLC0415

        assert 'TS2VecModelParameters' in mod.__all__

    def test_cost_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.cost.config as mod  # noqa: PLC0415

        assert 'CoSTModelParameters' in mod.__all__

    def test_autotcl_config_all(self) -> None:
        import tscollection.models.convolutional.dilated.autotcl.config as mod  # noqa: PLC0415

        assert 'AutoTCLModelParameters' in mod.__all__


class TestNoDilatedCNNConfig:
    """DilatedCNNModelParameters no longer exists."""

    def test_import_fails(self) -> None:
        with pytest.raises(ImportError):
            from tscollection.models.convolutional.dilated.config import (  # noqa: F401, PLC0415
                DilatedCNNModelParameters,
            )
