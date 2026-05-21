"""Integration tests for from_config() factory methods on all three models.

Verifies that each model can be instantiated from its typed config dataclass,
that config fields propagate correctly to __init__ attributes, and that
augmentation kwargs pass through as additional_kwargs.

Also verifies the correct mixin inheritance for each model class.
"""

from tscollection.models._abstract import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)
from tscollection.models._augmentation.enums import (
    AutoTCLAugmentationMode,
    CoSTAugmentationMode,
    TS2VecAugmentationMode,
)
from tscollection.models.autotcl.model import AutoTCL
from tscollection.models.config import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    TS2VecModelParameters,
)
from tscollection.models.cost.model import CoST
from tscollection.models.ts2vec.model import TS2Vec


class TestFromConfigInstantiation:
    """Test that from_config() returns valid model instances."""

    def test_ts2vec_from_config_returns_instance(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec.from_config(
            config,
            augmentation_mode=TS2VecAugmentationMode.CROP_SHIFT,
            augmentation_method_params={},
        )
        assert isinstance(model, TS2Vec)

    def test_cost_from_config_returns_instance(self) -> None:
        config = CoSTModelParameters(input_dims=1, sequence_length=100)
        model = CoST.from_config(
            config,
            augmentation_mode=CoSTAugmentationMode.RANDOM_FUNCTIONS,
            augmentation_method_params={'sigma': 0.1},
        )
        assert isinstance(model, CoST)

    def test_autotcl_from_config_returns_instance(self) -> None:
        config = AutoTCLModelParameters(input_dims=1)
        # AutoTCL's neural network augmentation needs encoder params too;
        # provide them as additional_kwargs alongside the config unpack.
        model = AutoTCL.from_config(
            config,
            augmentation_mode=AutoTCLAugmentationMode.NEURAL_NETWORK,
            augmentation_method_params={
                'input_dims': 1,
                'output_dims': 320,
                'kernel_sizes': [],
            },
        )
        assert isinstance(model, AutoTCL)


class TestFromConfigAttributePropagation:
    """Test that config fields propagate correctly to model attributes."""

    def test_from_config_propagates_model_params(self) -> None:
        config = TS2VecModelParameters(
            input_dims=3,
            hidden_dims=128,
            output_dims=256,
            depth=8,
            dropout_rate=0.2,
        )
        model = TS2Vec.from_config(
            config,
            augmentation_mode=TS2VecAugmentationMode.CROP_SHIFT,
            augmentation_method_params={},
        )
        assert model.hparams.input_dims == 3
        assert model.hparams.hidden_dims == 128
        assert model.hparams.output_dims == 256
        assert model.hparams.depth == 8
        assert model.hparams.dropout_rate == 0.2

    def test_from_config_passes_additional_kwargs(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec.from_config(
            config,
            augmentation_mode=TS2VecAugmentationMode.CROP_SHIFT,
            augmentation_method_params={},
        )
        # augmentation_mode is passed as additional_kwarg and stored in __init__
        assert model.hparams.augmentation_mode == TS2VecAugmentationMode.CROP_SHIFT


class TestMixinInheritance:
    """Test that each model inherits the correct encoding mixin."""

    def test_ts2vec_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(TS2Vec, PoolingEncodingMixin)

    def test_cost_inherits_decomposition_encoding_mixin(self) -> None:
        assert issubclass(CoST, DecompositionEncodingMixin)

    def test_autotcl_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(AutoTCL, PoolingEncodingMixin)
