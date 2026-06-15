"""Model-agnostic augmentation primitives.

Extracted from ``tstcc/augmentations.py`` and reshaped to satisfy the
:model-agnostic :class:`~chronocratic.models.augmentation.base.Augmentation`
Protocol. Each primitive accepts a tensor and returns a transformed tensor
of the same shape.

Shared across all models. Imports nothing model-specific.

Exported symbols:
    - ``Jitter``, ``JitterParameters``: Additive Gaussian noise.
    - ``Scaling``, ``ScalingParameters``: Per-channel multiplicative scaling.
    - ``Permutation``, ``PermutationParameters``: Time-segment permutation.
    - ``ComposeAugmentation``: Chain primitives sequentially.
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
]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import Augmentation


def _normalize_dim(data: torch.Tensor, dim: int) -> int:
    """Convert a potentially negative dimension index to its absolute value."""
    return dim if dim >= 0 else data.dim() + dim


def _should_apply(p: float) -> bool:
    """Return ``True`` if the transform should be applied based on probability ``p``."""
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


class Jitter:
    """Add elementwise Gaussian noise with std ``sigma``.

    Satisfies the :class:`Augmentation` Protocol via ``__call__``.
    """

    def __init__(self, params: JitterParameters | None = None) -> None:
        self._params = params if params is not None else JitterParameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a jittered view of ``x``.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as ``x``, with Gaussian noise added
            (or the original tensor if the transform is skipped).
        """
        if not _should_apply(self._params.p):
            return x
        noise = torch.randn_like(x) * self._params.sigma
        return x + noise


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


class Scaling:
    """Multiply data by a per-channel Gaussian scale factor.

    Satisfies the :class:`Augmentation` Protocol via ``__call__``.
    """

    def __init__(self, params: ScalingParameters | None = None) -> None:
        self._params = params if params is not None else ScalingParameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scaled view of ``x``.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as ``x``, multiplied by a
            per-channel scale factor.
        """
        if not _should_apply(self._params.p):
            return x
        c_dim = _normalize_dim(x, self._params.channel_dim)
        shape = [1] * x.dim()
        shape[c_dim] = x.size(c_dim)
        if self._params.per_sample:
            shape[0] = x.size(0)
        factor = torch.randn(shape, device=x.device) * self._params.sigma + self._params.mean
        return x * factor


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


class Permutation:
    """Split each sample's time axis into segments and permute them.

    Satisfies the :class:`Augmentation` Protocol via ``__call__``.
    """

    def __init__(self, params: PermutationParameters | None = None) -> None:
        self._params = params if params is not None else PermutationParameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a view with per-sample time segments permuted.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as ``x``, with time-axis segments
            shuffled independently per batch element.
        """
        t_dim = _normalize_dim(x, self._params.time_dim)
        seq_len = x.size(t_dim)
        # Cannot meaningfully permute segments on short sequences
        _min_permute_len = 3
        if seq_len < _min_permute_len:
            return x.clone()
        batch_size = x.size(0)
        max_segments = self._params.max_segments

        result = torch.empty_like(x)
        for i in range(batch_size):
            num_segments = int(torch.randint(1, max_segments, (1,)).item())
            if num_segments > 1:
                split_points = torch.randperm(seq_len - 2)[: num_segments - 1].sort().values
                splits = torch.tensor_split(torch.arange(seq_len), split_points)
                permutation = torch.randperm(len(splits))
                warp = torch.cat([splits[int(p)] for p in permutation]).to(x.device)
                # x[i] removes the batch dim, so the time dim shifts down by 1.
                result[i] = x[i].index_select(t_dim - 1, warp)
            else:
                result[i] = x[i]
        return result


# --------------------------------------------------------------------------- #
# Compose
# --------------------------------------------------------------------------- #


class ComposeAugmentation:
    """Apply a sequence of augmentations one after another.

    Analogous to ``torchvision.transforms.Compose``. Each augmentation's
    output is fed as input to the next.

    Satisfies the :class:`Augmentation` Protocol via ``__call__``.
    """

    def __init__(self, augmentations: list[Augmentation]) -> None:
        """Initialize with a list of :class:`Augmentation` primitives.

        Args:
            augmentations: Ordered list of augmentation primitives to apply.
        """
        self._augmentations = augmentations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each configured augmentation sequentially.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor after all augmentations have been applied.
        """
        current = x
        for augmentation in self._augmentations:
            current = augmentation(current)
        return current
