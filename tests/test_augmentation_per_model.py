"""Tests for per-model augmentation modules.

Verifies that concrete augmentations live in their model directories:
    - ts2vec/augmentation.py: CropShiftProducer + CropShiftAugmentationParameters
    - cost/augmentation.py: CosTRandomFunctionAugmentation + CosTRandomFunctionAugmentationParameters
    - autotcl/augmentation/: methods.py, training.py, __init__.py
"""

import torch

from chronocratic.models.augmentation.base import (
    AugmentationTrainingStrategy,
)

# --------------------------------------------------------------------------- #
# Task 1: ts2vec/augmentation.py
# --------------------------------------------------------------------------- #


class TestTS2VecAugmentationModule:
    """CropShiftProducer lives in ts2vec/augmentation.py."""

    def test_import_from_ts2vec(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        assert CropShiftProducer is not None

    def test_import_params_from_ts2vec(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
        )

        assert CropShiftAugmentationParameters is not None

    def test_crop_shift_satisfies_producer_protocol(self) -> None:
        from chronocratic.models.augmentation.base import (
            AlignedPair as AlignedPairType,
        )
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        aug = CropShiftProducer()
        result = aug.produce(torch.randn(2, 100, 3))
        assert isinstance(result, AlignedPairType)

    def test_crop_shift_produce_returns_aligned_pair(self) -> None:
        from chronocratic.models.augmentation.base import AlignedPair
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        aug = CropShiftProducer()
        data = torch.randn(2, 100, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)
        assert result.first.shape[2] == 3
        assert result.second.shape[2] == 3
        assert isinstance(result.overlap_length, int)

    def test_crop_shift_with_params(self) -> None:
        from chronocratic.models.augmentation.base import AlignedPair
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
            CropShiftProducer,
        )

        aug = CropShiftProducer(params=CropShiftAugmentationParameters(temporal_unit=1))
        data = torch.randn(2, 100, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)

    def test_crop_shift_params_defaults(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
        )

        params = CropShiftAugmentationParameters()
        assert params.temporal_unit == 0

    def test_crop_shift_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        assert (
            CropShiftProducer.__module__
            == 'chronocratic.models.convolutional.dilated.ts2vec.augmentation'
        )

    def test_params_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftAugmentationParameters,
        )

        assert (
            CropShiftAugmentationParameters.__module__
            == 'chronocratic.models.convolutional.dilated.ts2vec.augmentation'
        )

    def test_all_exports(self) -> None:
        import chronocratic.models.convolutional.dilated.ts2vec.augmentation as mod

        assert 'CropShiftAugmentationParameters' in mod.__all__
        assert 'CropShiftProducer' in mod.__all__


# --------------------------------------------------------------------------- #
# Task 1: cost/augmentation.py
# --------------------------------------------------------------------------- #


class TestCoSTAugmentationModule:
    """CosTRandomFunctionAugmentation lives in cost/augmentation.py."""

    def test_import_from_cost(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
        )

        assert CosTRandomFunctionAugmentation is not None

    def test_import_params_from_cost(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentationParameters,
        )

        assert CosTRandomFunctionAugmentationParameters is not None

    def test_cost_aug_is_augmentation_method(self) -> None:
        """CoST aug still has .augment() for backward compat, but inherits from Augmentation Protocol."""
        from chronocratic.models.augmentation.base import Augmentation
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
            CosTRandomFunctionAugmentationParameters,
        )

        # CosTRandomFunctionAugmentation implements the Augmentation Protocol
        # (structural typing) and provides augment() for backward compatibility
        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        aug = CosTRandomFunctionAugmentation(params=params)
        assert isinstance(aug, Augmentation)

    def test_cost_aug_satisfies_augmentation_protocol(self) -> None:
        from chronocratic.models.augmentation.base import Augmentation
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
            CosTRandomFunctionAugmentationParameters,
        )

        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        aug = CosTRandomFunctionAugmentation(params=params)
        assert isinstance(aug, Augmentation)

    def test_cost_augment_returns_tensor(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
            CosTRandomFunctionAugmentationParameters,
        )

        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        aug = CosTRandomFunctionAugmentation(params=params)
        data = torch.randn(2, 50, 3)
        result = aug.augment(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape

    def test_cost_params_required_sigma(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentationParameters,
        )

        params = CosTRandomFunctionAugmentationParameters(sigma=0.2)
        assert params.sigma == 0.2
        assert params.p == 0.5

    def test_cost_dict_params_compat(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
        )

        aug = CosTRandomFunctionAugmentation(params={'sigma': 0.1, 'p': 0.5})
        data = torch.randn(2, 50, 3)
        result = aug.augment(data)
        assert isinstance(result, torch.Tensor)

    def test_cost_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentation,
        )

        assert (
            CosTRandomFunctionAugmentation.__module__
            == 'chronocratic.models.convolutional.dilated.cost.augmentation'
        )

    def test_cost_params_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.cost.augmentation import (
            CosTRandomFunctionAugmentationParameters,
        )

        assert (
            CosTRandomFunctionAugmentationParameters.__module__
            == 'chronocratic.models.convolutional.dilated.cost.augmentation'
        )

    def test_cost_all_exports(self) -> None:
        import chronocratic.models.convolutional.dilated.cost.augmentation as mod

        assert 'CosTRandomFunctionAugmentation' in mod.__all__
        assert 'CosTRandomFunctionAugmentationParameters' in mod.__all__


# --------------------------------------------------------------------------- #
# Task 2: autotcl/augmentation/ package
# --------------------------------------------------------------------------- #


class TestAutoTCLAugmentationMethods:
    """AutoTCLNeuralNetworkAugmentation lives in autotcl/augmentation/methods.py."""

    def test_import_from_autotcl_methods(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
        )

        assert AutoTCLNeuralNetworkAugmentation is not None

    def test_import_params_from_autotcl_methods(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        assert AutoTCLNeuralNetworkAugmentationParameters is not None

    def test_is_trainable_augmentation(self) -> None:
        from chronocratic.models.augmentation.base import TrainableAugmentationProducer
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
        )

        assert issubclass(AutoTCLNeuralNetworkAugmentation, TrainableAugmentationProducer)

    def test_constructor_with_dataclass(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        aug = AutoTCLNeuralNetworkAugmentation(params=params)
        assert isinstance(aug, AutoTCLNeuralNetworkAugmentation)

    def test_has_trainable_params(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        aug = AutoTCLNeuralNetworkAugmentation(params=params)
        assert len(list(aug.parameters())) > 0

    def test_augment_returns_views(self) -> None:
        from chronocratic.models.augmentation.base import SingleView
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, output_dims=320, kernel_sizes=[3]
        )
        aug = AutoTCLNeuralNetworkAugmentation(params=params)
        aug.eval()
        data = torch.randn(2, 100, 1)
        with torch.no_grad():
            result = aug.augment(data)
        assert isinstance(result, SingleView)

    def test_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentation,
        )

        assert (
            AutoTCLNeuralNetworkAugmentation.__module__
            == 'chronocratic.models.convolutional.dilated.autotcl.augmentation.methods'
        )

    def test_params_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        assert (
            AutoTCLNeuralNetworkAugmentationParameters.__module__
            == 'chronocratic.models.convolutional.dilated.autotcl.augmentation.methods'
        )


class TestAutoTCLTrainingStrategies:
    """Training strategies live in autotcl/augmentation/training.py."""

    def test_import_rip_strategy(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            RIPTrainingStrategy,
        )

        assert RIPTrainingStrategy is not None

    def test_import_adversarial_strategy(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            AdversarialTrainingStrategy,
        )

        assert AdversarialTrainingStrategy is not None

    def test_rip_is_training_strategy(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            RIPTrainingStrategy,
        )

        assert issubclass(RIPTrainingStrategy, AugmentationTrainingStrategy)

    def test_adversarial_is_training_strategy(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            AdversarialTrainingStrategy,
        )

        assert issubclass(AdversarialTrainingStrategy, AugmentationTrainingStrategy)

    def test_rip_compute_loss_scalar(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            RIPTrainingStrategy,
        )

        strategy = RIPTrainingStrategy()
        x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_factor = torch.rand(2, 10, 3)
        loss = strategy.compute_loss(
            x_embeddings=x_emb, aug_x_embeddings=aug_x_emb, augmentation_factor=aug_factor
        )
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_adversarial_compute_loss_scalar(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            AdversarialTrainingStrategy,
        )

        strategy = AdversarialTrainingStrategy()
        x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_factor = torch.rand(2, 10, 3)
        loss = strategy.compute_loss(
            x_embeddings=x_emb, aug_x_embeddings=aug_x_emb, augmentation_factor=aug_factor
        )
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_rip_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            RIPTrainingStrategy,
        )

        assert (
            RIPTrainingStrategy.__module__
            == 'chronocratic.models.convolutional.dilated.autotcl.augmentation.training'
        )

    def test_adversarial_module_location(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
            AdversarialTrainingStrategy,
        )

        assert (
            AdversarialTrainingStrategy.__module__
            == 'chronocratic.models.convolutional.dilated.autotcl.augmentation.training'
        )


class TestAutoTCLAugmentationBarrel:
    """autotcl/augmentation/__init__.py re-exports all symbols."""

    def test_barrel_exports_methods(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation import (
            AutoTCLNeuralNetworkAugmentation,
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        assert AutoTCLNeuralNetworkAugmentation is not None
        assert AutoTCLNeuralNetworkAugmentationParameters is not None

    def test_barrel_exports_strategies(self) -> None:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation import (
            AdversarialTrainingStrategy,
            RIPTrainingStrategy,
        )

        assert AdversarialTrainingStrategy is not None
        assert RIPTrainingStrategy is not None

    def test_barrel_all_list(self) -> None:
        import chronocratic.models.convolutional.dilated.autotcl.augmentation as mod

        expected = {
            'AutoTCLNeuralNetworkAugmentation',
            'AutoTCLNeuralNetworkAugmentationParameters',
            'RIPTrainingStrategy',
            'AdversarialTrainingStrategy',
        }
        assert expected.issubset(set(mod.__all__))


# --------------------------------------------------------------------------- #
# Backward compatibility: old imports still work
# --------------------------------------------------------------------------- #


class TestBackwardCompatibility:
    """Barrel import paths still work after per-model migration."""

    def test_barrel_still_exports(self) -> None:
        from chronocratic.models.augmentation import (
            AdversarialTrainingStrategy,
            AutoTCLNeuralNetworkAugmentation,
            CosTRandomFunctionAugmentation,
            CropShiftProducer,
            RIPTrainingStrategy,
        )

        assert CropShiftProducer is not None
        assert CosTRandomFunctionAugmentation is not None
        assert AutoTCLNeuralNetworkAugmentation is not None
        assert RIPTrainingStrategy is not None
        assert AdversarialTrainingStrategy is not None
