"""TDD tests for CoST producer integration.

Verifies CosTRandomFunctionAugmentation satisfies the Augmentation Protocol
(__call__: Tensor -> Tensor) and CoST accepts AugmentationProducer[ViewPair]
with IndependentPair as default, replacing the old double-augment pattern.
"""

from __future__ import annotations

import math

import pytest
import torch

from tscollection.models.augmentation.base import (
    Augmentation,
    AugmentationProducer,
    ViewPair,
)
from tscollection.models.augmentation.producers import IndependentPair
from tscollection.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentation,
    CosTRandomFunctionAugmentationParameters,
)
from tscollection.models.convolutional.dilated.cost.model import CoST


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
        aug = CosTRandomFunctionAugmentation(params=CosTRandomFunctionAugmentationParameters(sigma=0.0, p=1.0))
        x = torch.randn(2, 50, 1)
        result = aug(x)
        torch.testing.assert_close(result, x)

    def test_call_with_p_zero_returns_unchanged(self) -> None:
        aug = CosTRandomFunctionAugmentation(params=CosTRandomFunctionAugmentationParameters(sigma=1.0, p=0.0))
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
        pair = model._augmentation.produce(torch.randn(2, 100, 1))
        assert isinstance(pair, ViewPair)
        assert pair.first.shape == (2, 100, 1)
        assert pair.second.shape == (2, 100, 1)

    def test_cost_produce_returns_view_pair(self) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        x = torch.randn(2, 100, 1)
        pair = model._augmentation.produce(x)
        assert hasattr(pair, 'first')
        assert hasattr(pair, 'second')


class TestCoSTTrainingWithProducer:
    """CoST trains 5 steps with finite loss using producer contract."""

    def test_cost_trains_5_steps_with_producer(self) -> None:
        aug = CosTRandomFunctionAugmentation()
        producer = IndependentPair(aug=aug)
        model = CoST(
            input_dims=1,
            sequence_length=100,
            augmentation=producer,
        )
        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, 'Loss must be a scalar tensor'
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )

    def test_cost_trains_5_steps_with_default_augmentation(self) -> None:
        model = CoST(input_dims=1, sequence_length=100)
        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)
        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )


class TestCoSTSeededEquivalence:
    """Seeded CoST produces identical loss sequence (SC-7 numerical equivalence)."""

    def test_seeded_cost_produces_identical_loss_sequence(self) -> None:
        torch.manual_seed(42)
        model1 = CoST(input_dims=1, sequence_length=100)
        torch.manual_seed(42)
        model2 = CoST(input_dims=1, sequence_length=100)

        losses1 = _train_steps(model=model1, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123)
        losses2 = _train_steps(model=model2, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123)

        assert len(losses1) == len(losses2) == 5
        for i, (l1, l2) in enumerate(zip(losses1, losses2, strict=True)):
            torch.testing.assert_close(l1, l2, rtol=1e-5, atol=1e-5)


def _train_steps(
    model: CoST,
    batch_size: int,
    seq_length: int,
    input_dims: int,
    num_steps: int,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Run ``num_steps`` training steps via a minimal Lightning Trainer."""
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, TensorDataset

    if seed is not None:
        torch.manual_seed(seed)

    data = torch.randn(batch_size * num_steps, seq_length, input_dims)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Collect losses manually
    collected: list[torch.Tensor] = []

    class _LossCollector(pl.Callback):
        def on_train_batch_end(self, *args, **kwargs) -> None:  # noqa: ARG002
            collected.append(model._last_loss.clone() if hasattr(model, '_last_loss') else torch.tensor(0.0))

    # Patch training_step to capture loss
    original_step = model.training_step

    def patched_step(batch, batch_idx):
        loss = original_step(batch, batch_idx)
        if loss is not None:
            collected.append(loss.clone())
        return loss

    model.training_step = patched_step  # type: ignore[method-assign]

    trainer = pl.Trainer(
        accelerator='cpu',
        max_steps=num_steps,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)

    return collected
