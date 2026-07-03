"""Tests for maybe_* helpers in augmentation/trainable_support.py.

Covers:
- maybe_train_augmentation returns None for non-trainable producers.
- maybe_configure_augmentation_optimizer returns None for non-trainable producers.
- maybe_train_augmentation delegates to train_step when producer is trainable.
- maybe_configure_augmentation_optimizer delegates to configure_optimizer when trainable.
- Both helpers use isinstance(x, TrainableAugmentationProducer) as the sole gate.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW

from chronocratic.models.augmentation.base import (
    AugmentationProducer,
    AugmentationTrainingStrategy,
    SingleView,
    TrainableAugmentationProducer,
)
from chronocratic.models.augmentation.primitives import Jitter
from chronocratic.models.augmentation.producers import SingleViewProducer
from chronocratic.models.augmentation.trainable_support import (
    maybe_configure_augmentation_optimizer,
    maybe_train_augmentation,
)


class _DummyStrategy(AugmentationTrainingStrategy):
    """Minimal training strategy for test doubles."""

    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0.0)


class _NoTrainStrategy(AugmentationTrainingStrategy):
    """Strategy that never trains."""

    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0.0)

    def should_train(self, epoch: int, batch_idx: int) -> bool:
        return False


class _DummyTrainableProducer(TrainableAugmentationProducer):
    """Minimal trainable producer for tests."""

    def __init__(self, strategy: AugmentationTrainingStrategy) -> None:
        super().__init__(training_strategy=strategy)
        self._dummy = nn.Linear(4, 4)
        self._train_step_called = False
        self._configure_called = False
        self._last_loss = torch.tensor(1.5)

    def produce(self, x: torch.Tensor) -> SingleView:
        return SingleView(view=x)

    def train_step(
        self, x: torch.Tensor, encoder: nn.Module, batch_idx: int
    ) -> torch.Tensor | None:
        self._train_step_called = True
        return self._last_loss

    def configure_optimizer(self, lr: float) -> AdamW:
        self._configure_called = True
        return AdamW(self.parameters(), lr=lr)


class TestMaybeTrainAugmentationNonTrainable:
    """maybe_train_augmentation returns None for non-trainable producers."""

    def test_returns_none_for_single_view_producer(self) -> None:
        """SingleViewProducer(Jitter) is not trainable."""
        producer: AugmentationProducer[SingleView] = SingleViewProducer(aug=Jitter())
        x = torch.randn(2, 10, 3)
        encoder = nn.Linear(3, 8)

        result = maybe_train_augmentation(producer, x=x, encoder=encoder, epoch=0, batch_idx=0)

        assert result is None

    def test_returns_none_for_stateless_producer(self) -> None:
        """Any stateless producer returns None."""
        producer = SingleViewProducer(aug=Jitter())
        assert not isinstance(producer, TrainableAugmentationProducer)

        result = maybe_train_augmentation(
            producer, x=torch.randn(1, 5, 2), encoder=nn.Linear(2, 4), epoch=1, batch_idx=0
        )

        assert result is None


class TestMaybeConfigureOptimizerNonTrainable:
    """maybe_configure_augmentation_optimizer returns None for non-trainable producers."""

    def test_returns_none_for_single_view_producer(self) -> None:
        """SingleViewProducer(Jitter) is not trainable."""
        producer: AugmentationProducer[SingleView] = SingleViewProducer(aug=Jitter())

        result = maybe_configure_augmentation_optimizer(producer, lr=0.001)

        assert result is None

    def test_returns_none_for_stateless_producer(self) -> None:
        """Any stateless producer returns None."""
        producer = SingleViewProducer(aug=Jitter())

        result = maybe_configure_augmentation_optimizer(producer, lr=0.01)

        assert result is None


class TestMaybeTrainAugmentationTrainable:
    """maybe_train_augmentation delegates to trainable producer when gate passes."""

    def test_delegates_to_train_step(self) -> None:
        """When producer is TrainableAugmentationProducer and should_train is True."""
        strategy = _DummyStrategy()
        producer = _DummyTrainableProducer(strategy)
        x = torch.randn(2, 10, 3)
        encoder = nn.Linear(3, 8)

        result = maybe_train_augmentation(producer, x=x, encoder=encoder, epoch=0, batch_idx=0)

        assert result is not None
        assert producer._train_step_called

    def test_skips_train_step_when_should_train_false(self) -> None:
        """When should_train_augmentation returns False, train_step is not called."""
        producer = _DummyTrainableProducer(_NoTrainStrategy())
        x = torch.randn(2, 10, 3)
        encoder = nn.Linear(3, 8)

        result = maybe_train_augmentation(producer, x=x, encoder=encoder, epoch=5, batch_idx=0)

        assert result is None
        assert not producer._train_step_called

    def test_returns_loss_from_train_step(self) -> None:
        """The loss returned by train_step is propagated."""
        producer = _DummyTrainableProducer(_DummyStrategy())
        producer._last_loss = torch.tensor(2.7)
        x = torch.randn(2, 10, 3)
        encoder = nn.Linear(3, 8)

        result = maybe_train_augmentation(producer, x=x, encoder=encoder, epoch=0, batch_idx=0)

        assert result is not None
        torch.testing.assert_close(result, torch.tensor(2.7))


class TestMaybeConfigureOptimizerTrainable:
    """maybe_configure_augmentation_optimizer delegates to trainable producer."""

    def test_delegates_to_configure_optimizer(self) -> None:
        producer = _DummyTrainableProducer(_DummyStrategy())

        result = maybe_configure_augmentation_optimizer(producer, lr=0.001)

        assert result is not None
        assert producer._configure_called


class TestIsinstanceGate:
    """Both helpers use isinstance(x, TrainableAugmentationProducer) as sole gate."""

    def test_gate_check_on_non_trainable(self) -> None:
        """Verify isinstance check is the discriminator."""
        producer = SingleViewProducer(aug=Jitter())

        assert not isinstance(producer, TrainableAugmentationProducer)
        assert (
            maybe_train_augmentation(
                producer, x=torch.randn(1, 5, 2), encoder=nn.Linear(2, 4), epoch=0, batch_idx=0
            )
            is None
        )
        assert maybe_configure_augmentation_optimizer(producer, lr=0.001) is None

    def test_gate_check_on_trainable(self) -> None:
        producer = _DummyTrainableProducer(_DummyStrategy())

        assert isinstance(producer, TrainableAugmentationProducer)
        assert maybe_configure_augmentation_optimizer(producer, lr=0.001) is not None
        result = maybe_train_augmentation(
            producer, x=torch.randn(1, 5, 2), encoder=nn.Linear(2, 4), epoch=0, batch_idx=0
        )
        assert result is not None
