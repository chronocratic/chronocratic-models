"""Augmentation primitives and the paired weak/strong augmentation for TS-TCC.

Contains:

- Atomic transforms (``Jitter``, ``Scaling``, ``Permutation``) implementing
  :class:`AugmentationMethod`. Each produces a single-view
  :class:`TrainingViews`.
- :class:`ComposeAugmentation` — chains transforms sequentially on one view
  (analogous to ``torchvision.transforms.Compose``).
- :class:`TSTCCPairedAugmentation` — the concrete subclass of the abstract
  :class:`PairedAugmentation`, providing the original TS-TCC weak (scaling)
  and strong (segment-permutation + jitter) views.

TS-TCC operates on tensors of shape ``(batch, channels, time)``, so the
defaults in this module use ``channel_dim=1`` and ``time_dim=-1``.
"""

from __future__ import annotations

__all__ = [
    'ComposeAugmentation',
    'Jitter',
    'JitterParameters',
    'Permutation',
    'PermutationParameters',
    'Scaling',
    'ScalingParameters',
    'TSTCCPairedAugmentation',
]

from dataclasses import dataclass
from typing import Any

import torch

from tscollection.models.augmentation.base import AugmentationMethod, TrainingViews
from tscollection.models.augmentation.composition import PairedAugmentation


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
            ``1`` for the ``(B, C, T)`` convention used by TS-TCC.
    """

    sigma: float = 0.1
    mean: float = 1.0
    p: float = 1.0
    per_sample: bool = False
    channel_dim: int = 1


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
# Permutation
# --------------------------------------------------------------------------- #


@dataclass
class PermutationParameters:
    """Parameters for :class:`Permutation`.

    Args:
        max_segments: Upper bound on the number of segments to split
            each sample into. The actual number is drawn uniformly from
            ``[1, max_segments)`` per sample.
        time_dim: Dimension index of the time axis. Defaults to ``-1``
            for the ``(B, C, T)`` convention used by TS-TCC.
    """

    max_segments: int = 5
    time_dim: int = -1


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


# --------------------------------------------------------------------------- #
# Compose
# --------------------------------------------------------------------------- #


class ComposeAugmentation(AugmentationMethod):
    """Apply a sequence of augmentations one after another.

    Each augmentation's first view is fed as input to the next. The
    final output is a single-view :class:`TrainingViews`.
    """

    def __init__(self, augmentations: list[AugmentationMethod]) -> None:
        self._augmentations = augmentations

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401
    ) -> TrainingViews:
        current = data
        for augmentation in self._augmentations:
            current = augmentation.augment(current, **kwargs).views[0]
        return TrainingViews(views=(current,), metadata={})


# --------------------------------------------------------------------------- #
# TS-TCC's concrete paired augmentation
# --------------------------------------------------------------------------- #


class TSTCCPairedAugmentation(PairedAugmentation):
    """TS-TCC's weak/strong augmentation pair.

    - Weak view: per-(sample, channel) Gaussian scaling around ``mean=2.0``.
    - Strong view: random segment permutation followed by additive jitter.

    Either view can be replaced via constructor arguments; the defaults
    reproduce the original TS-TCC contract.
    """

    def __init__(
        self, weak: AugmentationMethod | None = None, strong: AugmentationMethod | None = None
    ) -> None:
        self._weak: AugmentationMethod = weak if weak is not None else self._default_weak()
        self._strong: AugmentationMethod = strong if strong is not None else self._default_strong()

    @staticmethod
    def _default_weak() -> AugmentationMethod:
        return Scaling(ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=1))

    @staticmethod
    def _default_strong() -> AugmentationMethod:
        return ComposeAugmentation(
            [
                Permutation(PermutationParameters(max_segments=5, time_dim=-1)),
                Jitter(JitterParameters(sigma=0.8)),
            ]
        )

    @property
    def first(self) -> AugmentationMethod:
        return self._weak

    @property
    def second(self) -> AugmentationMethod:
        return self._strong
