"""Regression test: SimCLR with default settings learns on MINMAX-scaled data.

On data MINMAX-scaled to ``(0, 1)`` the per-series std is about 0.1, so the
absolute ``Jitter(0.8)`` of the strong view is several times the signal. With
``instance_normalize=False`` and the reference pair, this small SimCLR sits at
chance (tail gap within 1% of ``ln(2B - 1)``); the defaults reach about 40%
below it. This test trains the defaults on synthetic data with those
properties and asserts the loss moves clearly below chance (#99).

It guards ``instance_normalize`` only. Permutation hurts SimCLR without
moving its loss toward chance (the encoder learns position-invariant features
instead), so no loss check can see it; ``tests/unit/test_simclr.py`` asserts
that the default pair has none.
"""

import math

import torch

from chronocratic.models.convolutional.standard.simclr.model import SimCLR

_SEQUENCE_LENGTH = 128
_BATCH_SIZE = 16
_CLASS_COUNT = 4
_STEP_COUNT = 200
_TAIL_STEP_COUNT = 30
# Fraction of the chance loss the tail gap must fall below. The reference pair
# on raw MINMAX data reaches at most 1%; the defaults reach about 40%.
_MIN_REL_GAP = 0.1


def _bump_dataset(*, sample_count: int, generator: torch.Generator) -> torch.Tensor:
    """Build class-structured, non-stationary series, MINMAX-scaled to ``(0, 1)``.

    Each series is a Gaussian bump at a class-specific position, plus a random
    linear trend and white noise. The whole dataset is then min-max scaled with
    global statistics, so every value is positive and the per-series std is
    small.

    Args:
        sample_count: Number of series to generate.
        generator: Random generator for reproducibility.

    Returns:
        Tensor of shape ``(sample_count, _SEQUENCE_LENGTH, 1)``.
    """
    time = torch.linspace(0.0, 1.0, _SEQUENCE_LENGTH)
    labels = torch.randint(0, _CLASS_COUNT, (sample_count, 1), generator=generator)
    centers = (labels + 1) / (_CLASS_COUNT + 1)
    centers = centers + 0.02 * torch.randn(sample_count, 1, generator=generator)
    amplitudes = 1.0 + 0.3 * torch.randn(sample_count, 1, generator=generator)
    bumps = amplitudes * torch.exp(-(((time - centers) / 0.05) ** 2))
    trends = 0.5 * torch.randn(sample_count, 1, generator=generator) * time
    noise = 0.05 * torch.randn(sample_count, _SEQUENCE_LENGTH, generator=generator)
    series = bumps + trends + noise
    scaled = (series - series.min()) / (series.max() - series.min())
    return scaled.unsqueeze(-1)


def test_simclr_defaults_learn_on_minmax_scaled_data() -> None:
    """Mean gap-to-chance over the last steps must be clearly negative."""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    dataset = _bump_dataset(sample_count=_BATCH_SIZE * 16, generator=generator)
    model = SimCLR(
        input_dim=1,
        stem_conv_channels=8,
        encoder_stage_channels=(8, 16),
        encoder_stage_depths=(1, 1),
        encoder_stage_strides=(1, 2),
        projection_hidden_dim=32,
        projection_dim=16,
    )
    model.train()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    chance = math.log(2 * _BATCH_SIZE - 1)

    gaps: list[float] = []
    for _ in range(_STEP_COUNT):
        indices = torch.randint(0, len(dataset), (_BATCH_SIZE,), generator=generator)
        loss = model._compute_loss(dataset[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gaps.append(loss.item() - chance)

    tail_gap = sum(gaps[-_TAIL_STEP_COUNT:]) / _TAIL_STEP_COUNT
    assert tail_gap < -_MIN_REL_GAP * chance, (
        f"SimCLR stayed at chance: tail gap {tail_gap:.4f}, first gap {gaps[0]:.4f}, "
        f"chance {chance:.4f}"
    )
