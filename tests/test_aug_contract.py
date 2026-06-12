"""Tests for new augmentation contract types in base.py.

Verifies Augmentation Protocol, AugmentationProducer Protocol,
SingleView/ViewPair/AlignedPair dataclasses, and
TrainableAugmentationProducer nominal ABC.
"""

import pytest
import torch
from dataclasses import is_dataclass
from torch import nn
from torch.optim import AdamW
from typing import Protocol

from tscollection.models.augmentation.base import (
    Augmentation,
    AugmentationProducer,
    AugmentationTrainingStrategy,
    AlignedPair,
    SingleView,
    TrainableAugmentationProducer,
    ViewPair,
)

# --------------------------------------------------------------------------- #
# Augmentation Protocol
# --------------------------------------------------------------------------- #


class TestAugmentationProtocol:
    """Augmentation Protocol is structural with __call__: Tensor -> Tensor."""

    def test_is_protocol(self) -> None:
        """Augmentation is a Protocol subclass."""
        assert issubclass(Augmentation, Protocol)

    def test_concrete_class_satisfies_protocol(self) -> None:
        """A class with __call__(Tensor) -> Tensor satisfies the Protocol structurally."""

        class IdentityAug:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x

        # Augmentation is not @runtime_checkable; verify structural conformance
        # by checking the concrete class has the required method.
        assert hasattr(IdentityAug, '__call__')
        aug = IdentityAug()
        x = torch.randn(2, 10, 4)
        result = aug(x)
        assert isinstance(result, torch.Tensor)


# --------------------------------------------------------------------------- #
# SingleView
# --------------------------------------------------------------------------- #


class TestSingleView:
    """SingleView is a frozen dataclass with view: torch.Tensor."""

    def test_is_frozen_dataclass(self) -> None:
        """SingleView is a frozen dataclass."""
        assert is_dataclass(SingleView)

    def test_view_field(self) -> None:
        """SingleView has a 'view' field of type torch.Tensor."""
        tensor = torch.randn(2, 10, 4)
        sv = SingleView(view=tensor)
        assert isinstance(sv.view, torch.Tensor)
        assert torch.equal(sv.view, tensor)

    def test_frozen(self) -> None:
        """SingleView is immutable (frozen)."""
        sv = SingleView(view=torch.randn(1, 5, 2))
        with pytest.raises(Exception):
            sv.view = torch.randn(1, 5, 2)  # type: ignore


# --------------------------------------------------------------------------- #
# ViewPair
# --------------------------------------------------------------------------- #


class TestViewPair:
    """ViewPair is a frozen dataclass with first and second fields."""

    def test_is_frozen_dataclass(self) -> None:
        """ViewPair is a frozen dataclass."""
        assert is_dataclass(ViewPair)

    def test_fields(self) -> None:
        """ViewPair has 'first' and 'second' fields of type torch.Tensor."""
        first = torch.randn(2, 10, 4)
        second = torch.randn(2, 10, 4)
        vp = ViewPair(first=first, second=second)
        assert isinstance(vp.first, torch.Tensor)
        assert isinstance(vp.second, torch.Tensor)
        assert torch.equal(vp.first, first)
        assert torch.equal(vp.second, second)

    def test_frozen(self) -> None:
        """ViewPair is immutable (frozen)."""
        vp = ViewPair(first=torch.randn(1, 5, 2), second=torch.randn(1, 5, 2))
        with pytest.raises(Exception):
            vp.first = torch.randn(1, 5, 2)  # type: ignore


# --------------------------------------------------------------------------- #
# AlignedPair
# --------------------------------------------------------------------------- #


class TestAlignedPair:
    """AlignedPair extends ViewPair with overlap_length: int."""

    def test_extends_viewpair(self) -> None:
        """AlignedPair is a subclass of ViewPair."""
        assert issubclass(AlignedPair, ViewPair)

    def test_overlap_length_field(self) -> None:
        """AlignedPair has 'overlap_length' field of type int."""
        first = torch.randn(2, 10, 4)
        second = torch.randn(2, 10, 4)
        ap = AlignedPair(first=first, second=second, overlap_length=5)
        assert ap.overlap_length == 5
        assert isinstance(ap.overlap_length, int)

    def test_has_first_and_second(self) -> None:
        """AlignedPair inherits 'first' and 'second' from ViewPair."""
        first = torch.randn(2, 10, 4)
        second = torch.randn(2, 10, 4)
        ap = AlignedPair(first=first, second=second, overlap_length=5)
        assert torch.equal(ap.first, first)
        assert torch.equal(ap.second, second)

    def test_frozen(self) -> None:
        """AlignedPair is immutable (frozen)."""
        ap = AlignedPair(
            first=torch.randn(1, 5, 2),
            second=torch.randn(1, 5, 2),
            overlap_length=5,
        )
        with pytest.raises(Exception):
            ap.overlap_length = 10  # type: ignore


# --------------------------------------------------------------------------- #
# AugmentationProducer Protocol
# --------------------------------------------------------------------------- #


class TestAugmentationProducer:
    """AugmentationProducer[V] is a covariant Protocol with produce(Tensor) -> V."""

    def test_is_protocol(self) -> None:
        """AugmentationProducer is a Protocol subclass."""
        assert issubclass(AugmentationProducer, Protocol)

    def test_covariant_type_parameter(self) -> None:
        """A class with produce(Tensor) -> SingleView satisfies AugmentationProducer structurally."""

        class _TestProducer:
            def produce(self, x: torch.Tensor) -> SingleView:
                return SingleView(view=x)

        # AugmentationProducer is not @runtime_checkable; verify structural conformance
        assert hasattr(_TestProducer, 'produce')
        producer = _TestProducer()
        x = torch.randn(2, 10, 4)
        result = producer.produce(x)
        assert isinstance(result, SingleView)


# --------------------------------------------------------------------------- #
# TrainableAugmentationProducer (nominal ABC)
# --------------------------------------------------------------------------- #


class TestTrainableAugmentationProducer:
    """TrainableAugmentationProducer is a nominal ABC + nn.Module."""

    def test_is_abc_and_nn_module(self) -> None:
        """TrainableAugmentationProducer is a subclass of nn.Module and is abstract."""
        assert issubclass(TrainableAugmentationProducer, nn.Module)
        with pytest.raises(TypeError):
            TrainableAugmentationProducer(training_strategy=make_strategy())  # type: ignore[type-abstract]

    def test_has_abstract_produce(self) -> None:
        """TrainableAugmentationProducer declares produce() as abstract."""
        assert 'produce' in TrainableAugmentationProducer.__abstractmethods__

    def test_has_abstract_train_step(self) -> None:
        """TrainableAugmentationProducer declares train_step() as abstract."""
        assert 'train_step' in TrainableAugmentationProducer.__abstractmethods__

    def test_configure_optimizer_returns_adamw(self) -> None:
        """configure_optimizer() returns an AdamW optimizer."""

        class _ConcreteProducer(TrainableAugmentationProducer):
            def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
                super().__init__(training_strategy=training_strategy)
                self._dummy_param = nn.Linear(4, 4)  # gives .parameters() something

            def produce(self, x: torch.Tensor) -> SingleView:
                return SingleView(view=x)

            def train_step(
                self, x: torch.Tensor, encoder: nn.Module, batch_idx: int
            ) -> torch.Tensor | None:
                return None

        producer = _ConcreteProducer(training_strategy=make_strategy())
        optimizer = producer.configure_optimizer(lr=0.001)
        assert isinstance(optimizer, AdamW)

    def test_should_train_delegates_to_strategy(self) -> None:
        """should_train_augmentation() delegates to training strategy."""

        class _ConcreteProducer(TrainableAugmentationProducer):
            def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
                super().__init__(training_strategy=training_strategy)
                self._dummy_param = nn.Linear(4, 4)

            def produce(self, x: torch.Tensor) -> SingleView:
                return SingleView(view=x)

            def train_step(
                self, x: torch.Tensor, encoder: nn.Module, batch_idx: int
            ) -> torch.Tensor | None:
                return None

        strategy = make_strategy(training_ratio_step=2)
        producer = _ConcreteProducer(training_strategy=strategy)
        assert producer.should_train_augmentation(epoch=0, batch_idx=0) is True
        assert producer.should_train_augmentation(epoch=1, batch_idx=0) is False
        assert producer.should_train_augmentation(epoch=2, batch_idx=0) is True


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy (retained unchanged)
# --------------------------------------------------------------------------- #


class TestAugmentationTrainingStrategyRetained:
    """AugmentationTrainingStrategy is still importable and unchanged."""

    def test_is_abstract(self) -> None:
        """AugmentationTrainingStrategy is still abstract."""
        with pytest.raises(TypeError):
            AugmentationTrainingStrategy()

    def test_has_compute_loss_abstract(self) -> None:
        """AugmentationTrainingStrategy still has abstract compute_loss."""
        assert 'compute_loss' in AugmentationTrainingStrategy.__abstractmethods__

    def test_has_should_train(self) -> None:
        """AugmentationTrainingStrategy still has should_train method."""
        assert hasattr(AugmentationTrainingStrategy, 'should_train')


# --------------------------------------------------------------------------- #
# Backward compatibility: existing symbols still importable
# --------------------------------------------------------------------------- #


class TestBackwardCompatibility:
    """Existing symbols remain importable from base.py."""

    def test_training_views_importable(self) -> None:
        """TrainingViews is still importable."""
        from tscollection.models.augmentation.base import TrainingViews

        assert TrainingViews is not None

    def test_augmentation_method_importable(self) -> None:
        """AugmentationMethod is still importable."""
        from tscollection.models.augmentation.base import AugmentationMethod

        assert AugmentationMethod is not None

    def test_trainable_augmentation_importable(self) -> None:
        """TrainableAugmentation is still importable."""
        from tscollection.models.augmentation.base import TrainableAugmentation

        assert TrainableAugmentation is not None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_strategy(training_ratio_step: int = 1):
    """Create a minimal AugmentationTrainingStrategy for tests."""

    class _TestStrategy(AugmentationTrainingStrategy):
        def __init__(self, training_ratio_step: int = 1) -> None:
            super().__init__(training_ratio_step=training_ratio_step)

        def compute_loss(
            self,
            x_embeddings: torch.Tensor,
            aug_x_embeddings: torch.Tensor,
            augmentation_factor: torch.Tensor,
        ) -> torch.Tensor:
            return torch.tensor(0.0)

    return _TestStrategy(training_ratio_step=training_ratio_step)
