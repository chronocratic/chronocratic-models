"""Tests for the augmentation ABC hierarchy and training strategies.

Verifies that TrainingViews, AugmentationMethod, TrainableAugmentation,
AugmentationTrainingStrategy, RIPTrainingStrategy, and AdversarialTrainingStrategy
behave correctly according to decisions D-01 through D-03.
"""

import pytest
import torch

from tscollection.models.augmentation import (
    AdversarialTrainingStrategy,
    AugmentationMethod,
    AugmentationTrainingStrategy,
    RIPTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)


# --------------------------------------------------------------------------- #
# TrainingViews
# --------------------------------------------------------------------------- #


class TestTrainingViews:
    """TrainingViews dataclass field access and structure."""

    def test_views_tuple_access(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        views = TrainingViews(views=(t1, t2), metadata={'k': 'v'})
        assert views.views[0].shape == (2, 10, 4)
        assert views.views[1].shape == (2, 10, 4)

    def test_metadata_access(self) -> None:
        views = TrainingViews(views=(torch.randn(1, 5, 2),), metadata={'k': 'v'})
        assert views.metadata['k'] == 'v'


# --------------------------------------------------------------------------- #
# AugmentationMethod ABC
# --------------------------------------------------------------------------- #


class TestAugmentationMethod:
    """AugmentationMethod is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AugmentationMethod()  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# TrainableAugmentation ABC
# --------------------------------------------------------------------------- #


class TestTrainableAugmentation:
    """TrainableAugmentation is abstract and inherits nn.Module."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            TrainableAugmentation(training_strategy=RIPTrainingStrategy())


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy ABC
# --------------------------------------------------------------------------- #


class TestAugmentationTrainingStrategy:
    """AugmentationTrainingStrategy is abstract."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AugmentationTrainingStrategy()  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# should_train default
# --------------------------------------------------------------------------- #


class TestShouldTrain:
    """Default should_train returns True."""

    def test_should_train_default_true(self) -> None:
        strategy = RIPTrainingStrategy()
        assert strategy.should_train(epoch=0, batch_idx=0) is True


# --------------------------------------------------------------------------- #
# RIPTrainingStrategy
# --------------------------------------------------------------------------- #


class TestRIPTrainingStrategy:
    """RIPTrainingStrategy.compute_loss produces a scalar tensor requiring grad."""

    def test_compute_loss_returns_scalar(self) -> None:
        strategy = RIPTrainingStrategy()
        x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_factor = torch.rand(2, 10, 3)

        loss = strategy.compute_loss(
            x_embeddings=x_emb,
            aug_x_embeddings=aug_x_emb,
            augmentation_factor=aug_factor,
        )
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad

    def test_loss_equivalence_to_original_auto_tcl(self) -> None:
        """RIPTrainingStrategy must match original AutoTCL loss for identical inputs.

        calculate_regular_consistency uses internal randomness (torch.randint),
        so we patch it to return a deterministic value to ensure both calls
        produce identical consistency terms.
        """
        from unittest.mock import patch
        from torch.nn import functional as F
        from tscollection.models.losses import maximum_mean_discrepancy_with_gaussian_kernel_loss

        torch.manual_seed(42)
        x_emb = torch.randn(2, 10, 32)
        aug_x_emb = torch.randn(2, 10, 32)
        aug_factor = torch.rand(2, 10, 3)

        # Original AutoTCL loss computation (from model.py:195-216)
        consistency_weight = 0.001
        regularization_weight = 0.001
        regularization_threshold = 0.4

        # Compute original loss components
        vx_distance = maximum_mean_discrepancy_with_gaussian_kernel_loss(x_emb, aug_x_emb)
        fixed_consistency = torch.tensor(0.5)
        regularization_loss = F.relu(
            torch.sum(aug_factor, dim=-1).mean() - regularization_threshold
        )
        original_loss = (
            vx_distance
            + regularization_weight * regularization_loss
            + consistency_weight * fixed_consistency
        )

        # New strategy loss computation with patched consistency
        strategy = RIPTrainingStrategy(
            consistency_weight=consistency_weight,
            regularization_weight=regularization_weight,
            regularization_threshold=regularization_threshold,
        )
        # Patch at the source module so lazy import in compute_loss returns fixed value
        with patch(
            'tscollection.models.cnn.dilated.autotcl.utils.calculate_regular_consistency',
            return_value=fixed_consistency,
        ):
            new_loss = strategy.compute_loss(
                x_embeddings=x_emb,
                aug_x_embeddings=aug_x_emb,
                augmentation_factor=aug_factor,
            )

        assert torch.allclose(original_loss, new_loss, atol=1e-6)


# --------------------------------------------------------------------------- #
# AdversarialTrainingStrategy
# --------------------------------------------------------------------------- #


class TestAdversarialTrainingStrategy:
    """AdversarialTrainingStrategy.compute_loss produces a scalar tensor requiring grad."""

    def test_compute_loss_returns_scalar(self) -> None:
        strategy = AdversarialTrainingStrategy()
        x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_x_emb = torch.randn(2, 10, 32, requires_grad=True)
        aug_factor = torch.rand(2, 10, 3)

        loss = strategy.compute_loss(
            x_embeddings=x_emb,
            aug_x_embeddings=aug_x_emb,
            augmentation_factor=aug_factor,
        )
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad

    def test_loss_equivalence_to_original_auto_tcl(self) -> None:
        """AdversarialTrainingStrategy must match original AutoTCL loss."""
        from tscollection.models.losses import info_nce_loss

        torch.manual_seed(42)
        x_emb = torch.randn(2, 10, 32)
        aug_x_emb = torch.randn(2, 10, 32)
        aug_factor = torch.rand(2, 10, 3)

        # Original AutoTCL adversarial loss (from model.py:218-222)
        original_loss = -1 * info_nce_loss(x_emb, aug_x_emb, temperature=1.0)

        # New strategy loss computation
        strategy = AdversarialTrainingStrategy()
        new_loss = strategy.compute_loss(
            x_embeddings=x_emb,
            aug_x_embeddings=aug_x_emb,
            augmentation_factor=aug_factor,
        )

        assert torch.allclose(original_loss, new_loss, atol=1e-6)


# --------------------------------------------------------------------------- #
# Concrete augmentations (Task 4)
# --------------------------------------------------------------------------- #


class TestCropShiftAugmentation:
    """CropShiftAugmentation returns TrainingViews with correct structure."""

    @pytest.fixture(autouse=True)
    def _imports(self) -> None:
        """Lazy import to avoid circular dependency at module load time."""
        from tscollection.models.augmentation.strategies import CropShiftAugmentation
        self.aug_cls = CropShiftAugmentation  # type: ignore[attr-defined]

    def test_augment_returns_training_views(self) -> None:
        aug = self.aug_cls()  # type: ignore[attr-defined]
        data = torch.randn(2, 100, 3)
        result = aug.augment(data)
        assert isinstance(result, TrainingViews)
        assert len(result.views) == 2
        assert 'crop_length' in result.metadata

    def test_augment_with_params(self) -> None:
        from tscollection.models.augmentation.config import CropShiftAugmentationParameters

        aug = self.aug_cls(  # type: ignore[attr-defined]
            params=CropShiftAugmentationParameters(temporal_unit=1)
        )
        data = torch.randn(2, 100, 3)
        result = aug.augment(data)
        assert isinstance(result, TrainingViews)

    def test_augment_with_temporal_unit_kwarg(self) -> None:
        aug = self.aug_cls()  # type: ignore[attr-defined]
        data = torch.randn(2, 100, 3)
        result = aug.augment(data, temporal_unit=5)
        assert isinstance(result, TrainingViews)


class TestCosTRandomFunctionAugmentation:
    """CosTRandomFunctionAugmentation returns TrainingViews with single view."""

    def test_augment_returns_training_views(self) -> None:
        from tscollection.models.augmentation.strategies import CosTRandomFunctionAugmentation
        from tscollection.models.augmentation.config import CosTRandomFunctionAugmentationParameters

        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        aug = CosTRandomFunctionAugmentation(params=params)
        data = torch.randn(2, 50, 3)
        result = aug.augment(data)
        assert isinstance(result, TrainingViews)
        assert len(result.views) == 1


class TestAutoTCLNeuralNetworkAugmentation:
    """AutoTCLNeuralNetworkAugmentation constructor and augment behavior."""

    def test_constructor_accepts_dataclass(self) -> None:
        from tscollection.models.augmentation.strategies import AutoTCLNeuralNetworkAugmentation
        from tscollection.models.augmentation.config import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        strategy = RIPTrainingStrategy()
        aug = AutoTCLNeuralNetworkAugmentation(
            params=params, training_strategy=strategy
        )
        assert isinstance(aug, AutoTCLNeuralNetworkAugmentation)

    def test_has_trainable_params(self) -> None:
        from tscollection.models.augmentation.strategies import AutoTCLNeuralNetworkAugmentation
        from tscollection.models.augmentation.config import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        strategy = RIPTrainingStrategy()
        aug = AutoTCLNeuralNetworkAugmentation(
            params=params, training_strategy=strategy
        )
        param_count = len(list(aug.parameters()))
        assert param_count > 0

    def test_augment_returns_training_views(self) -> None:
        from tscollection.models.augmentation.strategies import AutoTCLNeuralNetworkAugmentation
        from tscollection.models.augmentation.config import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, output_dims=320, kernel_sizes=[3]
        )
        strategy = RIPTrainingStrategy()
        aug = AutoTCLNeuralNetworkAugmentation(
            params=params, training_strategy=strategy
        )
        aug.eval()
        data = torch.randn(2, 100, 1)
        with torch.no_grad():
            result = aug.augment(data)
        assert isinstance(result, TrainingViews)
        assert len(result.views) == 1


class TestLazyImport:
    """CropShift lazy import resolves at runtime."""

    def test_lazy_import_works(self) -> None:
        from tscollection.models.augmentation.strategies import CropShiftAugmentation

        aug = CropShiftAugmentation()
        data = torch.randn(2, 100, 3)
        result = aug.augment(data)
        assert isinstance(result, TrainingViews)
