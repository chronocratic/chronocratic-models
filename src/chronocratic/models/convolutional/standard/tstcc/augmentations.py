"""TS-TCC augmentation wiring.

Defines the model-local :class:`TSTCCScaling` (upstream per-timestep scaling)
and :func:`_default_tstcc_pair`, which builds the default ``TSTCCScaling``
(weak) | ``Jitter`` (strong) pair, with segment permutation off unless
requested.

TS-TCC operates on tensors of shape ``(batch, time, channels)``, so the
builders in this module use ``time_dim=1``.
"""

from __future__ import annotations

__all__ = ["TSTCCScaling", "TSTCCScalingParameters", "_default_tstcc_pair"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
)
from chronocratic.models.augmentation.producers import RolePairProducer

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import Augmentation, AugmentationProducer, ViewPair


@dataclass
class TSTCCScalingParameters:
    """Parameters for :class:`TSTCCScaling`.

    Args:
        sigma: Std of the per-timestep Gaussian scale factor.
        mean: Mean of the per-timestep Gaussian scale factor.
    """

    sigma: float = 1.1
    mean: float = 2.0


class TSTCCScaling:
    """Multiply every timestep by an independent Gaussian factor.

    Matches upstream TS-TCC ``scaling``, which draws a factor of shape
    ``(N, T)`` and multiplies every channel by it. On ``(B, T, C)`` input the
    factor has shape ``(B, T, 1)`` and is broadcast across channels.

    The shared :class:`~chronocratic.models.augmentation.primitives.Scaling`
    draws one scalar per sample/channel instead, which TS-TCC's bias-free conv
    and GroupNorm cancel, so it left the weak view nearly the identity.

    Satisfies the :class:`Augmentation` Protocol via ``__call__``.
    """

    def __init__(self, params: TSTCCScalingParameters | None = None) -> None:
        """Initialize the scaling augmentation.

        Args:
            params: Scale-factor distribution. Defaults to
                ``TSTCCScalingParameters()`` (``sigma=1.1``, ``mean=2.0``).
        """
        self._params = params if params is not None else TSTCCScalingParameters()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-timestep scaled view of ``x``.

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


def _default_tstcc_pair(
    *, max_segments: int = 1, jitter_sigma: float = 0.8
) -> AugmentationProducer[ViewPair]:
    """Build the default TS-TCC weak/strong augmentation pair.

    The weak view applies :class:`TSTCCScaling`. The strong view applies
    :class:`Jitter`, preceded by segment :class:`Permutation` only when
    ``max_segments > 1``. Permutation is off by default because it scrambles
    position-dependent shapes, leaving the temporal and contextual losses
    with positives that share nothing. ``max_segments=8`` gives the HAR
    recipe of the original TS-TCC repository.

    Args:
        max_segments: Upper bound on permutation segments. ``1`` disables
            permutation.
        jitter_sigma: Std of the strong view's additive Gaussian noise.

    Returns:
        A producer that returns :class:`ViewPair` instances when
        :meth:`~AugmentationProducer.produce` is called.

    Raises:
        ValueError: If ``max_segments < 1``.
    """
    if max_segments < 1:
        msg = f"max_segments must be >= 1, got {max_segments}"
        raise ValueError(msg)
    jitter = Jitter(JitterParameters(sigma=jitter_sigma))
    strong: Augmentation = (
        ComposeAugmentation(
            [Permutation(PermutationParameters(max_segments=max_segments, time_dim=1)), jitter]
        )
        if max_segments > 1
        else jitter
    )
    return RolePairProducer(first=TSTCCScaling(), second=strong)
