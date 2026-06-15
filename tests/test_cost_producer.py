"""TDD tests for CoST producer integration.

Verifies CosTRandomFunctionAugmentation satisfies the Augmentation Protocol
(__call__: Tensor -> Tensor) and CoST accepts AugmentationProducer[ViewPair]
with IndependentPair as default, replacing the old double-augment pattern.
"""

from collections.abc import Callable

import torch

from chronocratic.models.augmentation.base import (
    Augmentation,
    ViewPair,
)
from chronocratic.models.augmentation.producers import IndependentPair
from chronocratic.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentation,
    CosTRandomFunctionAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.cost.model import CoST


class TestCosTRandomFunctionAugmentationProtocol:
    """CosTRandomFunctionAugmentation satisfies Augmentation Protocol."""

    def test_has_call_method(self) -> None:
        aug = CosTRandomFunctionAugmentation()
        assert callable(aug)

    def test_satisfies_augmentation_protocol(self) -> None:
        aug = CosTRandomFunctionAugmentation()
        assert isinstance(aug, Augmentation)

    def test_call_returns_same_shape_tensor(self) -> None:
        aug = CosTRandomFunctionAugmentation(sigma=0.1)
        x = torch.randn(4, 100, 3)
        result = aug(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_with_sigma_zero_returns_unchanged(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=0.0, p=1.0)
        aug = CosTRandomFunctionAugmentation(params=params)
        x = torch.randn(2, 50, 1)
        result = aug(x)
        torch.testing.assert_close(result, x)

    def test_call_with_p_zero_returns_unchanged(self) -> None:
        params = CosTRandomFunctionAugmentationParameters(sigma=1.0, p=0.0)
        aug = CosTRandomFunctionAugmentation(params=params)
        x = torch.randn(2, 50, 1)
        result = aug(x)
        torch.testing.assert_close(result, x)


class TestCoSTProducerIntegration:
    """CoST accepts AugmentationProducer[ViewPair] and uses IndependentPair."""

    def test_cost_accepts_independent_pair(self) -> None:
        aug = CosTRandomFunctionAugmentation()
        producer = IndependentPair(aug=aug)
        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=producer,
        )
        assert isinstance(model, CoST)

    def test_cost_default_is_independent_pair(self) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        # Default augmentation should be an IndependentPair
        pair = model._augmentation.produce(torch.randn(2, 100, 1))  # noqa: SLF001
        assert isinstance(pair, ViewPair)
        assert pair.first.shape == (2, 100, 1)
        assert pair.second.shape == (2, 100, 1)

    def test_cost_produce_returns_view_pair(self) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        x = torch.randn(2, 100, 1)
        pair = model._augmentation.produce(x)  # noqa: SLF001
        assert hasattr(pair, 'first')
        assert hasattr(pair, 'second')


class TestCoSTTrainingWithProducer:
    """CoST trains 5 steps with finite loss using producer contract."""

    def test_cost_trains_5_steps_with_producer(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
        finite_losses: Callable[..., None],
    ) -> None:
        aug = CosTRandomFunctionAugmentation()
        producer = IndependentPair(aug=aug)
        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=producer,
        )
        losses = train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
        finite_losses(losses, expected_min=5)

    def test_cost_trains_5_steps_with_default_augmentation(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
        finite_losses: Callable[..., None],
    ) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        losses = train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
        finite_losses(losses, expected_min=5)


class TestCoSTSeededEquivalence:
    """Seeded CoST produces identical loss sequence (SC-7 numerical equivalence)."""

    def test_seeded_cost_produces_identical_loss_sequence(
        self, train_steps: Callable[..., list[torch.Tensor]]
    ) -> None:
        torch.manual_seed(42)
        model1 = CoST(input_dims=1, sequence_length=100)
        torch.manual_seed(42)
        model2 = CoST(input_dims=1, sequence_length=100)

        losses1 = train_steps(
            model=model1, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123
        )
        losses2 = train_steps(
            model=model2, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123
        )

        assert len(losses1) == len(losses2) == 5
        for _i, (l1, l2) in enumerate(zip(losses1, losses2, strict=True)):
            torch.testing.assert_close(l1, l2, rtol=1e-5, atol=1e-5)
