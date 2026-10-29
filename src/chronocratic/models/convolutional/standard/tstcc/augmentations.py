"""TS-TCC augmentation wiring.

Provides :func:`_default_tstcc_pair`, which builds the default
:class:`~chronocratic.models.augmentation.primitives.TimestepScaling` (weak) |
``Jitter`` (strong) pair, with segment permutation off unless requested.

TS-TCC operates on tensors of shape ``(batch, time, channels)``, so the
builders in this module use ``time_dim=1``.
"""

from __future__ import annotations

__all__ = ["_default_tstcc_pair"]

from typing import TYPE_CHECKING

from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    TimestepScaling,
)
from chronocratic.models.augmentation.producers import RolePairProducer

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import Augmentation, AugmentationProducer, ViewPair


def _default_tstcc_pair(
    *, max_segments: int = 1, jitter_sigma: float = 0.8
) -> AugmentationProducer[ViewPair]:
    """Build the default TS-TCC weak/strong augmentation pair.

    The weak view applies :class:`TimestepScaling`. The strong view applies
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
    return RolePairProducer(first=TimestepScaling(), second=strong)
