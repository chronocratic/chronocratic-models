"""Seeded synthetic time series that mimic real benchmark data.

Classification mimics UCR/UEA: each class owns a dominant frequency plus a shape
template (a Gaussian bump at a class-specific position). Forecasting mimics
ETT/electricity: trend + two seasonal periods + noise per channel.
"""

from enum import StrEnum

import torch

__all__ = ["Scaling", "make_classification", "make_forecasting_windows", "scale"]


class Scaling(StrEnum):
    """How generated series are scaled."""

    ZSCORE = "zscore"  # per series, per channel
    MINMAX = "minmax"  # global, to (0, 1)
    NONE = "none"


def make_classification(
    *,
    n_samples: int,
    seq_len: int,
    n_channels: int,
    n_classes: int,
    noise_std: float = 0.3,
    nan_pad_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate class-structured series of shape ``(N, T, C)`` with balanced labels.

    Args:
        n_samples: Number of series.
        seq_len: Length of each series.
        n_channels: Number of channels.
        n_classes: Number of classes.
        noise_std: Std of additive white noise.
        nan_pad_fraction: Fraction of series whose tail is replaced by NaN
            (variable-length simulation, as in UEA padding).
        seed: Seed controlling templates, labels, jitter and noise.

    Returns:
        ``(x, y)`` with ``x`` float32 ``(N, T, C)`` and ``y`` long ``(N,)``.
    """
    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(0.0, 1.0, seq_len)
    labels = (torch.arange(n_samples) % n_classes)[torch.randperm(n_samples, generator=generator)]
    gains = 0.5 + torch.rand(n_channels, generator=generator)  # fixed per-channel mixing

    frequencies = (2 + 2 * labels).float().unsqueeze(1)  # (N, 1) cycles per series
    centers = ((labels + 1) / (n_classes + 1)).float().unsqueeze(1)
    phase = 0.1 * torch.randn(n_samples, 1, generator=generator)
    center_jitter = 0.02 * torch.randn(n_samples, 1, generator=generator)
    wave = torch.sin(2 * torch.pi * frequencies * time + phase)
    bump = 2.0 * torch.exp(-(((time - centers - center_jitter) / 0.06) ** 2))
    base = wave + bump  # (N, T)

    noise = noise_std * torch.randn(n_samples, seq_len, n_channels, generator=generator)
    x = base.unsqueeze(-1) * gains + noise

    n_padded = int(nan_pad_fraction * n_samples)
    if n_padded > 0:
        padded_rows = torch.randperm(n_samples, generator=generator)[:n_padded]
        valid_lengths = torch.randint(
            seq_len // 2, seq_len - seq_len // 8, (n_padded,), generator=generator
        )
        for row, length in zip(padded_rows.tolist(), valid_lengths.tolist(), strict=True):
            x[row, length:] = float("nan")
    return x.float(), labels


def make_forecasting_windows(
    *, n_windows: int, seq_len: int, n_channels: int, seed: int = 0
) -> torch.Tensor:
    """Generate sliding windows over one long multichannel series.

    The long series is trend + seasonality at periods 16 and 48 + noise, with
    per-channel amplitudes and phases, like a small ETT/electricity slice.

    Args:
        n_windows: Number of windows (stride 1 over the long series).
        seq_len: Window length.
        n_channels: Number of channels.
        seed: Seed.

    Returns:
        Float32 tensor ``(n_windows, seq_len, n_channels)``.
    """
    generator = torch.Generator().manual_seed(seed)
    total = n_windows + seq_len - 1
    time = torch.arange(total).float().unsqueeze(1)  # (L, 1)
    amp_short = 1.0 + torch.rand(1, n_channels, generator=generator)
    amp_long = 0.5 + torch.rand(1, n_channels, generator=generator)
    phase = 2 * torch.pi * torch.rand(1, n_channels, generator=generator)
    slope = 0.002 * torch.randn(1, n_channels, generator=generator)
    series = (
        slope * time
        + amp_short * torch.sin(2 * torch.pi * time / 16 + phase)
        + amp_long * torch.sin(2 * torch.pi * time / 48 + phase)
        + 0.1 * torch.randn(total, n_channels, generator=generator)
    )
    return series.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous().float()


def scale(*, x: torch.Tensor, method: Scaling) -> torch.Tensor:
    """Scale ``(N, T, C)`` data, ignoring NaN padding.

    Args:
        x: Data tensor; NaNs mark padding and are preserved.
        method: Scaling method.

    Returns:
        Scaled tensor, same shape.
    """
    if method is Scaling.NONE:
        return x
    if method is Scaling.ZSCORE:
        mean = torch.nanmean(x, dim=1, keepdim=True)
        centered = x - mean
        var = torch.nanmean(centered**2, dim=1, keepdim=True)
        return centered / torch.sqrt(var + 1e-8)
    finite = x[~torch.isnan(x)]
    low, high = finite.min(), finite.max()
    return (x - low) / (high - low)
