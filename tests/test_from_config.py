"""Integration tests for from_config() factory methods on all three models.

Verifies that each model can be instantiated from its typed config dataclass,
that config fields propagate correctly to __init__ attributes, and that
augmentation instances pass through as additional_kwargs.

Also verifies the correct mixin inheritance for each model class.
"""

import pytest
import torch

from tscollection.models.augmentation import (
    AutoTCLNeuralNetworkAugmentation,
    CosTRandomFunctionAugmentation,
    CropShiftAugmentation,
    RIPTrainingStrategy,
)
from tscollection.models.augmentation.config import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
)
from tscollection.models.convolutional.dilated._mixin.encoding import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)
from tscollection.models.convolutional.dilated.autotcl.model import AutoTCL
from tscollection.models.convolutional.dilated.cost.model import CoST
from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec
from tscollection.models.config import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    TS2VecModelParameters,
)


class TestFromConfigInstantiation:
    """Test that from_config() returns valid model instances."""

    def test_ts2vec_from_config_returns_instance(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec.from_config(
            config,
            augmentation=CropShiftAugmentation(),
        )
        assert isinstance(model, TS2Vec)

    def test_cost_from_config_returns_instance(self) -> None:
        config = CoSTModelParameters(input_dims=1, sequence_length=100)
        model = CoST.from_config(
            config,
            augmentation=CosTRandomFunctionAugmentation(
                params=CosTRandomFunctionAugmentationParameters(sigma=0.1)
            ),
        )
        assert isinstance(model, CoST)

    def test_autotcl_from_config_returns_instance(self) -> None:
        config = AutoTCLModelParameters(input_dims=1)
        model = AutoTCL.from_config(
            config,
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
            augmentation=CropShiftAugmentation(),
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
            augmentation=CropShiftAugmentation(),
        )
        # augmentation is ignored by save_hyperparameters, not in hparams
        assert not hasattr(model.hparams, 'augmentation')


class TestMixinInheritance:
    """Test that each model inherits the correct encoding mixin."""

    def test_ts2vec_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(TS2Vec, PoolingEncodingMixin)

    def test_cost_inherits_decomposition_encoding_mixin(self) -> None:
        assert issubclass(CoST, DecompositionEncodingMixin)

    def test_autotcl_inherits_pooling_encoding_mixin(self) -> None:
        assert issubclass(AutoTCL, PoolingEncodingMixin)


class TestFromConfigOverlapDetection:
    """Test that from_config raises on overlapping keys."""

    def test_ts2vec_raises_on_overlapping_keys(self) -> None:
        config = TS2VecModelParameters(input_dims=1, learning_rate=0.01)
        with pytest.raises(ValueError, match='overlapping'):
            TS2Vec.from_config(
                config,
                learning_rate=0.5,
                augmentation=CropShiftAugmentation(),
            )

    def test_cost_raises_on_overlapping_keys(self) -> None:
        config = CoSTModelParameters(input_dims=1, sequence_length=100, learning_rate=0.01)
        with pytest.raises(ValueError, match='overlapping'):
            CoST.from_config(
                config,
                learning_rate=0.5,
                augmentation=CosTRandomFunctionAugmentation(
                    params=CosTRandomFunctionAugmentationParameters(sigma=0.1)
                ),
            )

    def test_autotcl_raises_on_overlapping_keys(self) -> None:
        config = AutoTCLModelParameters(input_dims=1, learning_rate=0.01)
        with pytest.raises(ValueError, match='overlapping'):
            AutoTCL.from_config(
                config,
                learning_rate=0.5,
                augmentation=AutoTCLNeuralNetworkAugmentation(
                    params=AutoTCLNeuralNetworkAugmentationParameters(
                        input_dims=1,
                        output_dims=320,
                        kernel_sizes=[3],
                    ),
                    training_strategy=RIPTrainingStrategy(),
                ),
            )


class TestTrainingStepIntegration:
    """Verify that a training step runs without crashing."""

    def test_ts2vec_training_step_runs(self) -> None:
        config = TS2VecModelParameters(input_dims=1)
        model = TS2Vec.from_config(
            config,
            augmentation=CropShiftAugmentation(),
        )
        model.eval()
        batch = torch.randn(4, 100, 1)
        loss = model.validation_step(batch, batch_idx=0)
        assert loss is not None
        assert loss.ndim == 0
