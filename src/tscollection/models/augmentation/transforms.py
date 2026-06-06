"""Atomic augmentation primitives.

Each class subclasses :class:`AugmentationMethod` and produces a single
augmented view. Use :mod:`tscollection.models.augmentation.composition`
(``ComposeAugmentation``, ``PairedAugmentation``) to combine them.

Shape convention: primitives default to ``(batch, time, channels)`` per
the ``AugmentationMethod`` ABC. Models that use ``(batch, channels, time)``
(e.g. TS-TCC) override ``time_dim`` / ``channel_dim`` accordingly.
"""

from __future__ import annotations

__all__ = [
    'Jitter',
    'JitterParameters',
    'Permutation',
    'PermutationParameters',
    'Scaling',
    'ScalingParameters',
    'Shift',
    'ShiftParameters',
]

from dataclasses import dataclass
from typing import Any

import torch

from tscollection.models.augmentation.base import AugmentationMethod, TrainingViews


def _normalize_dim(data: torch.Tensor, dim: int) -> int:
    return dim if dim >= 0 else data.dim() + dim


def _should_apply(p: float) -> bool:
    return p >= 1.0 or torch.rand((1,)).item() < p


# --------------------------------------------------------------------------- #
# Jitter
# --------------------------------------------------------------------------- #


@dataclass
class JitterParameters:
    """Parameters for :class:`Jitter`.

    Args:
        sigma: Std of the additive Gaussian noise.
        p: Probability of applying the transform. ``1.0`` means always.
    """

    sigma: float = 0.1
    p: float = 1.0


class Jitter(AugmentationMethod):
    """Add elementwise Gaussian noise with std ``sigma``."""

    def __init__(self, params: JitterParameters | None = None) -> None:
        self._params = params if params is not None else JitterParameters()

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
        if not _should_apply(self._params.p):
            return TrainingViews(views=(data,), metadata={})
        noise = torch.randn_like(data) * self._params.sigma
        return TrainingViews(views=(data + noise,), metadata={})


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


@dataclass
class ScalingParameters:
    """Parameters for :class:`Scaling`.

    Args:
        sigma: Std of the per-channel Gaussian scale factor.
        mean: Mean of the per-channel scale factor.
        p: Probability of applying the transform.
        per_sample: If ``True``, draw an independent factor for each
            sample in the batch. If ``False``, the factor is shared
            across the batch.
        channel_dim: Dimension index of the channel axis. Defaults to
            ``-1`` for the ``(B, T, C)`` convention.
    """

    sigma: float = 0.1
    mean: float = 1.0
    p: float = 1.0
    per_sample: bool = False
    channel_dim: int = -1


class Scaling(AugmentationMethod):
    """Multiply data by a per-channel Gaussian scale factor."""

    def __init__(self, params: ScalingParameters | None = None) -> None:
        self._params = params if params is not None else ScalingParameters()

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
        if not _should_apply(self._params.p):
            return TrainingViews(views=(data,), metadata={})
        c_dim = _normalize_dim(data, self._params.channel_dim)
        shape = [1] * data.dim()
        shape[c_dim] = data.size(c_dim)
        if self._params.per_sample:
            shape[0] = data.size(0)
        factor = torch.randn(shape, device=data.device) * self._params.sigma + self._params.mean
        return TrainingViews(views=(data * factor,), metadata={})


# --------------------------------------------------------------------------- #
# Shift
# --------------------------------------------------------------------------- #


@dataclass
class ShiftParameters:
    """Parameters for :class:`Shift`.

    Args:
        sigma: Std of the per-channel additive offset.
        p: Probability of applying the transform.
        channel_dim: Dimension index of the channel axis. Defaults to
            ``-1`` for the ``(B, T, C)`` convention.
    """

    sigma: float = 0.1
    p: float = 1.0
    channel_dim: int = -1


class Shift(AugmentationMethod):
    """Add a per-channel Gaussian constant offset."""

    def __init__(self, params: ShiftParameters | None = None) -> None:
        self._params = params if params is not None else ShiftParameters()

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
        if not _should_apply(self._params.p):
            return TrainingViews(views=(data,), metadata={})
        c_dim = _normalize_dim(data, self._params.channel_dim)
        shape = [1] * data.dim()
        shape[c_dim] = data.size(c_dim)
        offset = torch.randn(shape, device=data.device) * self._params.sigma
        return TrainingViews(views=(data + offset,), metadata={})


# --------------------------------------------------------------------------- #
# Permutation
# --------------------------------------------------------------------------- #


@dataclass
class PermutationParameters:
    """Parameters for :class:`Permutation`.

    Args:
        max_segments: Upper bound on the number of segments to split
            each sample into. The actual number is drawn uniformly from
            ``[1, max_segments)`` per sample.
        time_dim: Dimension index of the time axis. Defaults to ``-2``
            for the ``(B, T, C)`` convention.
    """

    max_segments: int = 5
    time_dim: int = -2


class Permutation(AugmentationMethod):
    """Split each sample's time axis into segments and permute them."""

    def __init__(self, params: PermutationParameters | None = None) -> None:
        self._params = params if params is not None else PermutationParameters()

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
        t_dim = _normalize_dim(data, self._params.time_dim)
        batch_size = data.size(0)
        seq_len = data.size(t_dim)
        max_segments = self._params.max_segments

        result = torch.empty_like(data)
        for i in range(batch_size):
            num_segments = int(torch.randint(1, max_segments, (1,)).item())
            if num_segments > 1:
                split_points = torch.randperm(seq_len - 2)[: num_segments - 1].sort().values
                splits = torch.tensor_split(torch.arange(seq_len), split_points)
                permutation = torch.randperm(len(splits))
                warp = torch.cat([splits[int(p)] for p in permutation]).to(data.device)
                # data[i] removes the batch dim, so the time dim shifts down by 1.
                result[i] = data[i].index_select(t_dim - 1, warp)
            else:
                result[i] = data[i]
        return TrainingViews(views=(result,), metadata={})
