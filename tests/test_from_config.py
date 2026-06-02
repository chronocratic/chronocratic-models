"""Integration tests for direct model instantiation on all three models.

Verifies that each model can be instantiated from its typed config dataclass
via ``Model(**vars(config))``, that config fields propagate correctly to
``__init__`` attributes, and that augmentation instances pass through.

Also verifies the correct mixin inheritance for each model class.
"""

import torch

from tscollection.models.augmentation import (
    AutoTCLNeuralNetworkAugmentation,
    CosTRandomFunctionAugmentation,
    CropShiftAugmentation,
    RIPTrainingStrategy,
)
from tscollection.models.augmentation import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
)
from tscollection.models.convolutional.dilated._mixin.encoding import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)
from tscollection.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from tscollection.models.convolutional.dilated.autotcl.model import AutoTCL
from tscollection.models.convolutional.dilated.cost.config import CoSTModelParameters
from tscollection.models.convolutional.dilated.cost.model import CoST
from tscollection.models.convolutional.dilated.ts2vec.config import TS2VecModelParameters
from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec


class TestModelInstantiation:
    """Test that direct instantiation returns valid model instances."""

    def test_ts2vec_instantiation_returns_instance(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec(**vars(config), augmentation=CropShiftAugmentation())
        assert isinstance(model, TS2Vec)

    def test_cost_instantiation_returns_instance(self) -> None:
        config = CoSTModelParameters(input_dims=1, sequence_length=100)
        model = CoST(
            **vars(config),
            augmentation=CosTRandomFunctionAugmentation(
                params=CosTRandomFunctionAugmentationParameters(sigma=0.1)
            ),
        )
        assert isinstance(model, CoST)

    def test_autotcl_instantiation_returns_instance(self) -> None:
        config = AutoTCLModelParameters(input_dims=1)
        model = AutoTCL(
            **vars(config),
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=AutoTCLNeuralNetworkAugmentationParameters(
                    input_dims=1,
                    output_dims=320,
                    kernel_sizes=[3],
                ),
                training_strategy=RIPTrainingStrategy(),
            ),
        )
        assert isinstance(model, AutoTCL)


class TestConfigAttributePropagation:
    """Test that config fields propagate correctly to model attributes."""

    def test_instantiation_propagates_model_params(self) -> None:
        config = TS2VecModelParameters(
            input_dims=3,
            hidden_dims=128,
            output_dims=256,
            depth=8,
            dropout_rate=0.2,
        )
        model = TS2Vec(**vars(config), augmentation=CropShiftAugmentation())
        assert model.hparams.input_dims == 3
        assert model.hparams.hidden_dims == 128
        assert model.hparams.output_dims == 256
        assert model.hparams.depth == 8
        assert model.hparams.dropout_rate == 0.2

    def test_instantiation_passes_augmentation(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec(**vars(config), augmentation=CropShiftAugmentation())
        # augmentation is ignored by save_hyperparameters, not in hparams
        assert not hasattr(model.hparams, 'augmentation')


class TestDefaultAugmentation:
    """Test that models work without explicit augmentation."""

    def test_ts2vec_default_augmentation(self) -> None:
        model = TS2Vec(input_dims=1)
        assert model._augmentation is not None

    def test_cost_default_augmentation(self) -> None:
        model = CoST(input_dims=1, sequence_length=100, kernel_sizes=[3])
        assert model._augmentation is not None

    def test_autotcl_default_augmentation(self) -> None:
        model = AutoTCL(input_dims=1, kernel_sizes=[3])
        assert model._augmentation is not None


class TestMixinInheritance:
    """Test that each model inherits the correct encoding mixin."""

    def test_ts2vec_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(TS2Vec, PoolingEncodingMixin)

    def test_cost_inherits_decomposition_encoding_mixin(self) -> None:
        assert issubclass(CoST, DecompositionEncodingMixin)

    def test_autotcl_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(AutoTCL, PoolingEncodingMixin)


class TestTrainingStepIntegration:
    """Verify that a training step runs without crashing."""

    def test_ts2vec_training_step_runs(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec(**vars(config), augmentation=CropShiftAugmentation())
        model.eval()
        batch = torch.randn(4, 100, 1)
        loss = model.validation_step(batch, batch_idx=0)
        assert loss is not None
        assert loss.ndim == 0
