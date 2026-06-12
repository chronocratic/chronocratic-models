"""TS-TCC augmentation wiring.

Re-exports shared primitives from :mod:`augmentation/primitives` and provides
the ``_default_tstcc_pair()`` builder function that assembles a
:class:`RolePair` producer with the original TS-TCC weak/strong defaults.

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
    '_default_tstcc_pair',
]

# Re-export shared primitives.
from tscollection.models.augmentation.base import AugmentationProducer, ViewPair
from tscollection.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)
from tscollection.models.augmentation.producers import RolePair


def _default_tstcc_pair() -> AugmentationProducer[ViewPair]:
    """Build the default TS-TCC weak/strong augmentation pair.

    Returns a :class:`RolePair` producer whose first view applies Gaussian
    scaling (weak) and whose second view applies segment permutation followed
    by jitter (strong).

    Returns:
        A producer that returns :class:`ViewPair` instances when
        :meth:`~AugmentationProducer.produce` is called.
    """
    return RolePair(
        first=Scaling(
            ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=1)
        ),
        second=ComposeAugmentation(
            [
                Permutation(PermutationParameters(max_segments=5, time_dim=-1)),
                Jitter(JitterParameters(sigma=0.8)),
            ]
        ),
    )
