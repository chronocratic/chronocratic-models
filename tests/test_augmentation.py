"""Tests for the augmentation ABC hierarchy, new producer contract, and training strategies.

Verifies that AugmentationTrainingStrategy, RIPTrainingStrategy,
and AdversarialTrainingStrategy behave correctly. Tests new Augmentation Protocol,
AugmentationProducer[ViewSet] contract, ViewSet types (SingleView, ViewPair,
AlignedPair), and CropShiftProducer.

Legacy symbols (TrainingViews, AugmentationMethod, TrainableAugmentation) were
removed — their tests have been deleted.
"""

import pytest
import torch

from chronocratic.models.augmentation import (
    AlignedPair,
    Augmentation,
    AugmentationTrainingStrategy,
    SingleView,
    SingleViewProducer,
    ViewPair,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
    AdversarialTrainingStrategy,
    RIPTrainingStrategy,
)
from chronocratic.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentation,
    CosTRandomFunctionAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
    CropShiftAugmentationParameters,
    CropShiftProducer,
)

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
        """RIPTrainingStrategy must match original AutoTCL loss for identical inputs."""
        from unittest.mock import patch

        from torch.nn import functional as F

        from chronocratic.models.convolutional.dilated.autotcl.losses import (
            maximum_mean_discrepancy_with_gaussian_kernel_loss,
        )

        torch.manual_seed(42)
        x_emb = torch.randn(2, 10, 32)
        aug_x_emb = torch.randn(2, 10, 32)
        aug_factor = torch.rand(2, 10, 3)

        consistency_weight = 0.001
        regularization_weight = 0.001
        regularization_threshold = 0.4

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

        strategy = RIPTrainingStrategy(
            consistency_weight=consistency_weight,
            regularization_weight=regularization_weight,
            regularization_threshold=regularization_threshold,
        )
        with patch(
            'chronocratic.models.convolutional.dilated.autotcl.utils.calculate_regular_consistency',
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
        from chronocratic.models.convolutional.dilated.autotcl.losses import info_nce_loss

        torch.manual_seed(42)
        x_emb = torch.randn(2, 10, 32)
        aug_x_emb = torch.randn(2, 10, 32)
        aug_factor = torch.rand(2, 10, 3)

        original_loss = -1 * info_nce_loss(x_emb, aug_x_emb, temperature=1.0)

        strategy = AdversarialTrainingStrategy()
        new_loss = strategy.compute_loss(
            x_embeddings=x_emb,
            aug_x_embeddings=aug_x_emb,
            augmentation_factor=aug_factor,
        )

        assert torch.allclose(original_loss, new_loss, atol=1e-6)


# --------------------------------------------------------------------------- #
# Concrete augmentations
# --------------------------------------------------------------------------- #


class TestCropShiftProducer:
    """CropShiftProducer returns AlignedPair via .produce()."""

    def test_produce_returns_aligned_pair(self) -> None:
        aug = CropShiftProducer()
        data = torch.randn(2, 100, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)
        assert result.first.shape[2] == 3
        assert result.second.shape[2] == 3
        assert isinstance(result.overlap_length, int)

    def test_produce_with_params(self) -> None:
        aug = CropShiftProducer(
            params=CropShiftAugmentationParameters(temporal_unit=1)
        )
        data = torch.randn(2, 100, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)

    def test_produce_with_temporal_unit_kwarg(self) -> None:
        aug = CropShiftProducer(
            params=CropShiftAugmentationParameters(temporal_unit=5)
        )
        data = torch.randn(2, 1000, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)


class TestCosTRandomFunctionAugmentation:
    """CosTRandomFunctionAugmentation supports Augmentation Protocol."""

    def test_call_returns_tensor(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        aug = CosTRandomFunctionAugmentation(params=params)
        data = torch.randn(2, 50, 3)
        result = aug(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape


class TestAutoTCLNeuralNetworkAugmentation:
    """AutoTCLNeuralNetworkAugmentation constructor and augment behavior."""

    def test_constructor_accepts_dataclass(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        strategy = RIPTrainingStrategy()
        aug = AutoTCLNeuralNetworkAugmentation(
            params=params, training_strategy=strategy
        )
        assert isinstance(aug, AutoTCLNeuralNetworkAugmentation)

    def test_has_trainable_params(self) -> None:
        params = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=320)
        strategy = RIPTrainingStrategy()
        aug = AutoTCLNeuralNetworkAugmentation(
            params=params, training_strategy=strategy
        )
        param_count = len(list(aug.parameters()))
        assert param_count > 0

    def test_augment_returns_training_views(self) -> None:
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
        assert isinstance(result, SingleView)


class TestLazyImport:
    """CropShift lazy import resolves at runtime."""

    def test_lazy_import_works(self) -> None:
        aug = CropShiftProducer()
        data = torch.randn(2, 100, 3)
        result = aug.produce(data)
        assert isinstance(result, AlignedPair)


# --------------------------------------------------------------------------- #
# New contract — ViewSet types
# --------------------------------------------------------------------------- #


class TestViewSetTypes:
    """ViewSet dataclass types (SingleView, ViewPair, AlignedPair)."""

    def test_single_view_creation(self) -> None:
        t = torch.randn(2, 10, 4)
        sv = SingleView(view=t)
        assert sv.view is t

    def test_single_view_is_frozen(self) -> None:
        sv = SingleView(view=torch.randn(2, 10, 4))
        with pytest.raises(Exception):
            sv.view = torch.randn(3, 5, 2)  # type: ignore[attr-defined]

    def test_view_pair_creation(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        vp = ViewPair(first=t1, second=t2)
        assert vp.first is t1
        assert vp.second is t2

    def test_aligned_pair_extends_view_pair(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        ap = AlignedPair(first=t1, second=t2, overlap_length=8)
        assert isinstance(ap, ViewPair)
        assert ap.overlap_length == 8

    def test_aligned_pair_is_frozen(self) -> None:
        ap = AlignedPair(
            first=torch.randn(2, 10, 4),
            second=torch.randn(2, 10, 4),
            overlap_length=5,
        )
        with pytest.raises(Exception):
            ap.overlap_length = 10  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# New contract — Augmentation Protocol
# --------------------------------------------------------------------------- #


class TestAugmentationProtocol:
    """Augmentation Protocol structural checks."""

    def test_primitive_jitter_satisfies_protocol(self) -> None:
        from chronocratic.models.augmentation import Jitter

        aug = Jitter()
        data = torch.randn(2, 10, 4)
        result = aug(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape

    def test_augmentation_protocol_is_runtime_checkable(self) -> None:
        from chronocratic.models.augmentation import Jitter

        aug = Jitter()
        assert isinstance(aug, Augmentation)


# --------------------------------------------------------------------------- #
# New contract — SingleViewProducer
# --------------------------------------------------------------------------- #


class TestSingleViewProducer:
    """SingleViewProducer wraps one Augmentation, returns SingleView."""

    def test_produce_returns_single_view(self) -> None:
        from chronocratic.models.augmentation import Jitter

        producer = SingleViewProducer(aug=Jitter())
        data = torch.randn(2, 10, 4)
        result = producer.produce(data)
        assert isinstance(result, SingleView)
        assert result.view.shape == data.shape
