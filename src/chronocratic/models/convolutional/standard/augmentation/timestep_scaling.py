"""Per-timestep scaling, the default weak view of both TS-TCC and SimCLR."""

from __future__ import annotations

__all__ = ["PerTimestepScaling", "PerTimestepScalingParameters"]

from dataclasses import dataclass

import torch


@dataclass
class PerTimestepScalingParameters:
    """Parameters for :class:`PerTimestepScaling`.

    The defaults follow the TS-TCC reference implementation.

    Args:
        sigma: Std of the per-timestep Gaussian scale factor.
        mean: Mean of the per-timestep Gaussian scale factor.
    """

    sigma: float = 1.1
    mean: float = 2.0


class PerTimestepScaling:
    """Multiply every timestep by its own Gaussian factor.

    The factor changes from timestep to timestep and is shared by all
    channels: on ``(B, T, C)`` input it has shape ``(B, T, 1)``.

    TS-TCC and SimCLR both use it as their default weak view. Both encoders
    normalize with ``GroupNorm`` after bias-free convolutions, which cancels
    a factor that is constant over time, such as the one drawn by
    :class:`~chronocratic.models.augmentation.primitives.Scaling`. That view
    would reach the encoder nearly unchanged. A factor that varies over time
    survives the normalization.

    Satisfies the :class:`~chronocratic.models.augmentation.base.Augmentation`
    Protocol via ``__call__``.
    """

    def __init__(self, params: PerTimestepScalingParameters | None = None) -> None:
        """Initialize the scaling augmentation.

        Args:
            params: Scale-factor distribution. Defaults to
                ``PerTimestepScalingParameters()`` (``sigma=1.1``, ``mean=2.0``).
        """
        self._params = params if params is not None else PerTimestepScalingParameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scaled view of ``x`` with one factor per timestep.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        factor_shape = (x.size(0), x.size(1), 1)
        factor = (
            torch.randn(factor_shape, device=x.device, dtype=x.dtype) * self._params.sigma
            + self._params.mean
        )
        return x * factor
