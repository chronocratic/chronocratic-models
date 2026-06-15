"""Tests for the Seeded decorator in augmentation/decorators.py.

Covers:
- Seeded wraps a stateless producer and returns the same ViewSet type.
- Seeded with a fixed seed produces identical output on repeated calls.
- Seeded is Generic[V] (accepts any AugmentationProducer).
- Seeded raises TypeError when inner is TrainableAugmentationProducer.
"""

import pytest
import torch
from torch import nn

from chronocratic.models.augmentation.base import (
    AugmentationTrainingStrategy,
    SingleView,
    TrainableAugmentationProducer,
    ViewPair,
)
from chronocratic.models.augmentation.decorators import Seeded
from chronocratic.models.augmentation.primitives import Jitter
from chronocratic.models.augmentation.producers import (
    IndependentPair,
    SingleViewProducer,
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


class _DummyTrainableProducer(TrainableAugmentationProducer):
    """Minimal trainable producer for tests."""

    def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
        super().__init__(training_strategy=training_strategy)
        self._dummy = nn.Linear(4, 4)

    def produce(self, x: torch.Tensor) -> SingleView:
        return SingleView(view=x)

    def train_step(
        self, x: torch.Tensor, encoder: nn.Module, batch_idx: int
    ) -> torch.Tensor | None:
        return None


class TestSeededWrapsProducer:
    """Seeded wraps a stateless producer and preserves ViewSet type."""

    def test_seeded_returns_same_viewset_type(self) -> None:
        """Seeded wrapping SingleViewProducer still returns SingleView."""
        producer = SingleViewProducer(aug=Jitter())
        seeded = Seeded(inner=producer, seed=42)
        x = torch.randn(2, 10, 3)
        result = seeded.produce(x)

        assert isinstance(result, SingleView)

    def test_seeded_wraps_pair_producer(self) -> None:
        """Seeded wrapping IndependentPair returns ViewPair."""
        producer = IndependentPair(aug=Jitter())
        seeded = Seeded(inner=producer, seed=42)
        x = torch.randn(2, 10, 3)
        result = seeded.produce(x)

        assert isinstance(result, ViewPair)


class TestSeededDeterminism:
    """Seeded with fixed seed produces identical output on repeated calls."""

    def test_fixed_seed_produces_identical_output(self) -> None:
        """Two calls with the same seed return identical tensors."""
        producer = SingleViewProducer(aug=Jitter())
        seeded = Seeded(inner=producer, seed=123)
        x = torch.ones(2, 5, 3)

        result_a = seeded.produce(x)
        result_b = seeded.produce(x)

        assert isinstance(result_a, SingleView)
        assert isinstance(result_b, SingleView)
        torch.testing.assert_close(result_a.view, result_b.view)

    def test_different_seeds_produce_different_output(self) -> None:
        """Different seeds generally produce different outputs."""
        producer = SingleViewProducer(aug=Jitter())
        seeded_1 = Seeded(inner=producer, seed=1)
        seeded_2 = Seeded(inner=producer, seed=999)
        x = torch.ones(2, 5, 3)

        result_1 = seeded_1.produce(x)
        result_2 = seeded_2.produce(x)

        assert isinstance(result_1, SingleView)
        assert isinstance(result_2, SingleView)
        assert not torch.allclose(result_1.view, result_2.view)

    def test_seeded_does_not_leak_random_state(self) -> None:
        """Seeded uses fork_rng so outer random state is preserved."""
        producer = SingleViewProducer(aug=Jitter())
        seeded = Seeded(inner=producer, seed=42)
        x = torch.randn(2, 5, 3)

        torch.manual_seed(777)
        before_tensor = torch.randn(1)

        seeded.produce(x)

        torch.manual_seed(777)
        after_tensor = torch.randn(1)

        torch.testing.assert_close(before_tensor, after_tensor)


class TestSeededGeneric:
    """Seeded is Generic[V] and accepts any AugmentationProducer."""

    def test_seeded_accepts_augmentation_producer(self) -> None:
        """Seeded inner parameter type is AugmentationProducer."""
        producer = SingleViewProducer(aug=Jitter())
        assert hasattr(producer, "produce")
        assert callable(producer.produce)

        seeded = Seeded(inner=producer, seed=42)
        assert seeded is not None

    def test_seeded_with_pair_producer(self) -> None:
        """Seeded works with producers that return ViewPair (Generic[V])."""
        producer = IndependentPair(aug=Jitter())
        seeded = Seeded(inner=producer, seed=42)
        x = torch.randn(2, 5, 3)
        result = seeded.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape


class TestSeededTrainableGuard:
    """Seeded.__init__ raises TypeError if inner is TrainableAugmentationProducer."""

    def test_seeded_raises_for_trainable_producer(self) -> None:
        """Seeded must not wrap TrainableAugmentationProducer (SPEC §4.6)."""
        trainable = _DummyTrainableProducer(training_strategy=_DummyStrategy())
        with pytest.raises(TypeError, match="TrainableAugmentationProducer"):
            Seeded(inner=trainable, seed=42)

    def test_seeded_isinstance_gate(self) -> None:
        """Guard uses isinstance(inner, TrainableAugmentationProducer)."""
        trainable = _DummyTrainableProducer(training_strategy=_DummyStrategy())
        assert isinstance(trainable, TrainableAugmentationProducer)

        with pytest.raises(TypeError):
            Seeded(inner=trainable, seed=42)

    def test_seeded_allows_stateless_producer(self) -> None:
        """Seeded accepts stateless (non-trainable) producers."""
        producer = SingleViewProducer(aug=Jitter())
        assert not isinstance(producer, TrainableAugmentationProducer)

        seeded = Seeded(inner=producer, seed=42)
        assert seeded is not None
