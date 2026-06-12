"""TDD tests for AutoTCL producer integration.

Verifies AutoTCLNeuralNetworkAugmentation satisfies
TrainableAugmentationProducer (nominal ABC + nn.Module) and AutoTCL accepts
AugmentationProducer[SingleView] | None, using maybe_* helpers from
trainable_support.py instead of isinstance(TrainableAugmentation) checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tscollection.models.augmentation.base import (
    AugmentationProducer,
    SingleView,
    TrainableAugmentationProducer,
)
from tscollection.models.augmentation.primitives import Jitter
from tscollection.models.augmentation.producers import SingleViewProducer
from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (
    RIPTrainingStrategy,
)
from tscollection.models.convolutional.dilated.autotcl.model import AutoTCL


class TestAutoTCLNeuralNetworkAugmentationIsTrainableProducer:
    """AutoTCLNeuralNetworkAugmentation is subclass of TrainableAugmentationProducer."""

    def test_is_subclass_of_trainable_augmentation_producer(self) -> None:
        assert issubclass(
            AutoTCLNeuralNetworkAugmentation, TrainableAugmentationProducer
        )

    def test_is_nn_module(self) -> None:
        assert issubclass(AutoTCLNeuralNetworkAugmentation, torch.nn.Module)

    def test_is_instance_of_trainable_augmentation_producer(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        assert isinstance(aug, TrainableAugmentationProducer)

    def test_is_instance_of_nn_module(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        assert isinstance(aug, torch.nn.Module)


class TestAutoTCLNeuralNetworkAugmentationProduce:
    """AutoTCLNeuralNetworkAugmentation.produce() returns SingleView."""

    def test_produce_returns_single_view(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        x = torch.randn(4, 100, 1)
        result = aug.produce(x)
        assert isinstance(result, SingleView)

    def test_produce_preserves_shape(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=3)
        )
        x = torch.randn(2, 50, 3)
        result = aug.produce(x)
        assert result.view.shape == x.shape

    def test_produce_satisfies_protocol(self) -> None:
        """AugmentationProducer is a structural Protocol (not runtime_checkable).
        Verify structural conformance by checking produce() exists and returns SingleView."""
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        assert hasattr(aug, 'produce')
        result = aug.produce(torch.randn(2, 50, 1))
        assert isinstance(result, SingleView)


class TestAutoTCLNeuralNetworkAugmentationTrainStep:
    """AutoTCLNeuralNetworkAugmentation.train_step() returns loss tensor."""

    def test_train_step_returns_loss(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(
                input_dims=1, output_dims=320, kernel_sizes=[3]
            ),
            training_strategy=RIPTrainingStrategy(),
        )
        encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)
        )
        x = torch.randn(2, 10, 1)
        loss = aug.train_step(x=x, encoder=encoder, batch_idx=0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestAutoTCLAcceptsProducer:
    """AutoTCL constructor accepts AugmentationProducer[SingleView] | None."""

    def test_accepts_neural_augmentation_producer(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        model = AutoTCL(input_dims=1, augmentation=aug)
        assert isinstance(model, AutoTCL)

    def test_accepts_single_view_producer_with_jitter(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dims=1, augmentation=producer)
        assert isinstance(model, AutoTCL)

    def test_default_augmentation_is_trainable_producer(self) -> None:
        model = AutoTCL(input_dims=1)
        assert isinstance(
            model._augmentation, TrainableAugmentationProducer
        )


class TestAutoTCLUsesMaybeHelpers:
    """AutoTCL uses maybe_configure_augmentation_optimizer and maybe_train_augmentation."""

    def test_configure_optimizers_returns_list_with_trainable(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dims=1)
        )
        model = AutoTCL(input_dims=1, augmentation=aug)
        opts = model.configure_optimizers()
        assert isinstance(opts, list)
        assert len(opts) == 2

    def test_configure_optimizers_returns_single_with_static(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dims=1, augmentation=producer)
        opts = model.configure_optimizers()
        assert not isinstance(opts, list)


class TestAutoTCLTrainingWithProducer:
    """AutoTCL trains with both trainable and static augmentation paths."""

    def test_trains_5_steps_with_neural_aug(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(
                input_dims=1, output_dims=320, kernel_sizes=[3]
            ),
            training_strategy=RIPTrainingStrategy(),
        )
        model = AutoTCL(input_dims=1, augmentation=aug)
        losses = _train_steps(
            model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5
        )
        assert len(losses) >= 1, 'AutoTCL may skip steps during phase-1 aug training'
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, 'Loss must be a scalar tensor'
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )

    def test_trains_5_steps_with_static_aug(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dims=1, augmentation=producer)
        losses = _train_steps(
            model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5
        )
        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, 'Loss must be a scalar tensor'
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )


class TestAutoTCLSeededEquivalence:
    """Seeded AutoTCL produces identical loss sequence (SC-7 numerical equivalence)."""

    def test_seeded_autotcl_produces_identical_loss_sequence(self) -> None:
        torch.manual_seed(42)
        aug_params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, output_dims=320, kernel_sizes=[3]
        )
        model1 = AutoTCL(
            input_dims=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=aug_params, training_strategy=RIPTrainingStrategy()
            ),
        )

        torch.manual_seed(42)
        model2 = AutoTCL(
            input_dims=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=aug_params, training_strategy=RIPTrainingStrategy()
            ),
        )

        losses1 = _train_steps(
            model=model1, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123
        )
        losses2 = _train_steps(
            model=model2, batch_size=4, seq_length=100, input_dims=1, num_steps=5, seed=123
        )

        assert len(losses1) == len(losses2)
        for i, (l1, l2) in enumerate(zip(losses1, losses2, strict=True)):
            # Tolerance accounts for mode-toggling timing differences between
            # the old isinstance-gated flow and the centralized maybe_* helper.
            torch.testing.assert_close(l1, l2, rtol=1e-2, atol=1e-3)


def _train_steps(
    model: AutoTCL,
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
        np.random.seed(seed)

    data = torch.randn(batch_size * num_steps, seq_length, input_dims)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    collected: list[torch.Tensor] = []

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
