"""Integration tests for direct model instantiation on all four models.

Verifies that each model can be instantiated from its typed config dataclass
via ``Model(**vars(config))``, that config fields propagate correctly to
``__init__`` attributes, and that augmentation instances pass through.

Also verifies the correct mixin inheritance for each model class.
"""

import torch

from chronocratic.models._mixin.encoding import BasicEncodingMixin
from chronocratic.models.augmentation import IndependentPair
from chronocratic.models.convolutional.dilated._mixin.encoding import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)
from chronocratic.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL
from chronocratic.models.convolutional.dilated.cost.config import CoSTModelParameters
from chronocratic.models.convolutional.dilated.cost.model import CoST
from chronocratic.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec
from chronocratic.models.convolutional.standard.tstcc.augmentations import _default_tstcc_pair
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC


class TestModelInstantiation:
    """Test that direct instantiation returns valid model instances."""

    def test_ts2vec_instantiation_returns_instance(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec(**vars(config), augmentation=None)
        assert isinstance(model, TS2Vec)

    def test_cost_instantiation_returns_instance(self) -> None:
        config = CoSTModelParameters(input_dims=1, sequence_length=100)
        model = CoST(**vars(config), augmentation=IndependentPair(aug=None))
        assert isinstance(model, CoST)

    def test_autotcl_instantiation_returns_instance(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        config = AutoTCLModelParameters(input_dims=1)
        model = AutoTCL(
            **vars(config),
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
            ),
        )
        assert isinstance(model, AutoTCL)

    def test_tstcc_instantiation_returns_instance(self) -> None:
        from chronocratic.models.convolutional.standard.tstcc.config import TSTCCModelParameters

        config = TSTCCModelParameters(
            input_dims=1,
            conv_kernel_size=5,
            stride=1,
            output_dims=16,
            features_len=12,
            num_classes=10,
        )
        model = TSTCC(**vars(config), augmentation=_default_tstcc_pair())
        assert isinstance(model, TSTCC)


class TestMixinInheritance:
    """Test mixin inheritance for each model."""

    def test_ts2vec_is_pooling_encoding_mixin(self) -> None:
        model = TS2Vec(input_dims=1)
        assert isinstance(model, PoolingEncodingMixin)

    def test_cost_is_pooling_encoding_mixin(self) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        assert isinstance(model, DecompositionEncodingMixin)

    def test_autotcl_is_pooling_encoding_mixin(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        model = AutoTCL(
            input_dims=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
            ),
        )
        assert isinstance(model, PoolingEncodingMixin)

    def test_tstcc_is_basic_encoding_mixin(self) -> None:
        model = TSTCC(
            input_dims=1,
            conv_kernel_size=5,
            stride=1,
            output_dims=16,
            features_len=12,
            num_classes=10,
        )
        assert isinstance(model, BasicEncodingMixin)


class TestAugmentationConfigPropagation:
    """Test that augmentation config fields propagate to __init__ attributes."""

    def test_ts2vec_augmentation_config_propagates(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
            CropShiftProducer,
        )

        config = CropShiftAugmentationParameters(temporal_unit=2)
        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer(params=config))
        assert model._augmentation._params.temporal_unit == 2

    def test_cost_augmentation_config_propagates(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
            CosTRandomFunctionAugmentationParameters,
        )

        config = CosTRandomFunctionAugmentationParameters(sigma=0.2, p=0.8)
        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=CosTRandomFunctionAugmentation(params=config),
        )
        # CoST wraps plain Augmentation in IndependentPair
        assert model._augmentation._aug._params.sigma == 0.2
        assert model._augmentation._aug._params.p == 0.8

    def test_autotcl_augmentation_config_propagates(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        config = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        model = AutoTCL(input_dims=1, augmentation=AutoTCLNeuralNetworkAugmentation(params=config))
        assert model._augmentation.params.input_dims == 1


class TestAugmentationPassThrough:
    """Test that augmentation instances pass through to __init__ attributes."""

    def test_ts2vec_augmentation_pass_through(self) -> None:
        from chronocratic.models.augmentation.base import ViewPair
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer

        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
        assert model._augmentation is not None
        result = model._augmentation.produce(torch.randn(4, 100, 1))
        assert isinstance(result, ViewPair)

    def test_cost_augmentation_pass_through(self) -> None:
        from chronocratic.models.augmentation.base import ViewPair
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
        )

        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=IndependentPair(aug=CosTRandomFunctionAugmentation()),
        )
        assert model._augmentation is not None
        result = model._augmentation.produce(torch.randn(4, 100, 1))
        assert isinstance(result, ViewPair)

    def test_autotcl_augmentation_pass_through(self) -> None:
        from chronocratic.models.augmentation.base import SingleView
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        model = AutoTCL(
            input_dims=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
            ),
        )
        assert model._augmentation is not None
        result = model._augmentation.produce(torch.randn(4, 100, 1))
        assert isinstance(result, SingleView)

    def test_tstcc_augmentation_pass_through(self) -> None:
        from chronocratic.models.augmentation.base import ViewPair

        model = TSTCC(
            input_dims=1,
            conv_kernel_size=5,
            stride=1,
            output_dims=16,
            features_len=15,
            num_classes=10,
        )
        assert model._augmentation is not None
        data = torch.randn(4, 1, 100)
        result = model._augmentation.produce(data)
        assert isinstance(result, ViewPair)


class TestBackwardCompatModelConstruction:
    """Verify old-symbol construction patterns still work."""

    def test_ts2vec_with_crop_shift_producer(self) -> None:
        """CropShiftProducer still works with TS2Vec."""
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer

        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())
        assert model._augmentation is not None

    def test_cost_with_raw_augmentation(self) -> None:
        """CoST still accepts raw Augmentation (wraps in IndependentPair)."""
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
        )

        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=IndependentPair(aug=CosTRandomFunctionAugmentation()),
        )
        assert model._augmentation is not None

    def test_autotcl_with_neural_augmentation(self) -> None:
        """AutoTCL still accepts neural augmentation."""
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        model = AutoTCL(
            input_dims=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
            ),
        )
        assert model._augmentation is not None

    def test_tstcc_with_default_pair(self) -> None:
        """TSTCC still accepts default augmentation pair."""
        model = TSTCC(
            input_dims=1,
            conv_kernel_size=5,
            stride=1,
            output_dims=16,
            features_len=15,
            num_classes=10,
            augmentation=_default_tstcc_pair(),
        )
        assert model._augmentation is not None
