"""Regression test: TS-TCC with default settings learns on MINMAX-scaled data.

The HAR-tuned augmentation pair (absolute ``Jitter(0.8)`` + ``Permutation(8)``)
destroyed every bit of information shared by the two views on rbs_4-style data
(non-stationary shapes, MINMAX-scaled to ``(0, 1)``), so the loss sat exactly at
chance and the encoder collapsed to a constant embedding. This test trains the
defaults on synthetic data with the same properties and asserts the loss moves
clearly below chance.
"""

import torch

from chronocratic.models.convolutional.standard.tstcc.model import (
    _chance_loss,
    _COLLAPSE_REL_TOL,
    TSTCC,
)

_SEQUENCE_LENGTH = 128
_BATCH_SIZE = 16
_CLASS_COUNT = 4
_STEP_COUNT = 200
_TAIL_STEP_COUNT = 30


def _bump_dataset(*, sample_count: int, generator: torch.Generator) -> torch.Tensor:
    """Build class-structured, non-stationary series, MINMAX-scaled to ``(0, 1)``.

    Each series is a Gaussian bump at a class-specific position, plus a random
    linear trend and white noise. The whole dataset is then min-max scaled with
    global statistics, like the rbs_4 datamodule, so every value is positive and
    the per-series std is small.

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


def test_tstcc_defaults_learn_on_minmax_scaled_data() -> None:
    """Mean gap-to-chance over the last steps must be clearly negative."""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    dataset = _bump_dataset(sample_count=_BATCH_SIZE * 16, generator=generator)
    model = TSTCC(input_dim=1, sequence_length=_SEQUENCE_LENGTH)
    model.train()
    optimizers = model.configure_optimizers()

    gaps: list[float] = []
    for _ in range(_STEP_COUNT):
        indices = torch.randint(0, len(dataset), (_BATCH_SIZE,), generator=generator)
        batch = (dataset[indices], torch.zeros(_BATCH_SIZE, dtype=torch.long))
        loss, batch_size = model._compute_loss(batch)
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        chance = _chance_loss(batch_size=batch_size, temporal_weight=1.0, contextual_weight=0.7)
        gaps.append(loss.item() - chance)

    tail_gap = sum(gaps[-_TAIL_STEP_COUNT:]) / _TAIL_STEP_COUNT
    assert tail_gap < -_COLLAPSE_REL_TOL * chance, (
        f"TS-TCC stayed at chance: tail gap {tail_gap:.4f}, first gap {gaps[0]:.4f}, "
        f"chance {chance:.4f}"
    )
