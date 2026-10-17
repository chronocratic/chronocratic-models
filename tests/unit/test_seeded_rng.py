"""Training randomness must follow the global seed set by ``seed_everything``."""

from collections.abc import Callable

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models.augmentation.decorators import Seeded
from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL
from chronocratic.models.convolutional.dilated.cost.model import CoST
from chronocratic.models.convolutional.dilated.encoders.masking import (
    generate_binomial_mask,
    generate_continuous_mask,
)
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec
from chronocratic.models.utils.utils import process_sample_length


def _crop(*, sample: torch.Tensor) -> torch.Tensor:
    return process_sample_length(sample=sample, max_sample_length=16)


def _binomial_mask(*, sample: torch.Tensor) -> torch.Tensor:
    return generate_binomial_mask(batch_size=sample.size(0), seq_length=sample.size(1))


def _continuous_mask(*, sample: torch.Tensor) -> torch.Tensor:
    return generate_continuous_mask(batch_size=sample.size(0), seq_length=sample.size(1))


def _crop_shift(*, sample: torch.Tensor) -> torch.Tensor:
    # A fresh producer per draw, as a fresh training run would build one.
    pair = CropShiftProducer().produce(sample)
    return torch.cat([pair.first, pair.second], dim=1)


@pytest.mark.parametrize("draw", [_crop, _binomial_mask, _continuous_mask, _crop_shift])
def test_draw_repeats_under_same_seed(draw: Callable[..., torch.Tensor]) -> None:
    """Two draws after the same seed are identical; a different seed changes them."""
    sample = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256, 1)

    def seeded_draws(*, seed: int) -> list[torch.Tensor]:
        seed_everything(seed=seed, verbose=False)
        return [draw(sample=sample) for _ in range(3)]

    first = seeded_draws(seed=42)
    second = seeded_draws(seed=42)
    other = seeded_draws(seed=7)

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(
        a.shape == c.shape and torch.equal(a, c) for a, c in zip(first, other, strict=True)
    )


def test_seeded_producer_ignores_outer_state() -> None:
    """``Seeded`` repeats its views whatever the outer torch state is."""
    producer = Seeded(inner=CropShiftProducer(), seed=3)
    sample = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64, 1)

    torch.manual_seed(0)
    first = producer.produce(sample)
    torch.manual_seed(1)
    second = producer.produce(sample)

    assert first.overlap_length == second.overlap_length
    assert torch.equal(first.first, second.first)
    assert torch.equal(first.second, second.second)


def _build_ts2vec() -> pl.LightningModule:
    return TS2Vec(input_dim=2, max_train_length=32)


def _build_autotcl() -> pl.LightningModule:
    return AutoTCL(input_dim=2, max_train_length=32)


def _build_cost() -> pl.LightningModule:
    return CoST(input_dim=2, sequence_length=64, max_train_length=32)


@pytest.mark.parametrize("build", [_build_ts2vec, _build_autotcl, _build_cost])
def test_training_repeats_under_same_seed(build: Callable[[], pl.LightningModule]) -> None:
    """Two short fits after the same seed end with identical weights."""

    def fit_weights() -> list[torch.Tensor]:
        seed_everything(seed=42, verbose=False)
        data = torch.randn(8, 64, 2)
        loader = DataLoader(TensorDataset(data), batch_size=4)
        model = build()
        trainer = pl.Trainer(
            max_steps=3,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model=model, train_dataloaders=loader)
        return [p.detach().clone() for p in model.parameters()]

    first = fit_weights()
    second = fit_weights()

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
