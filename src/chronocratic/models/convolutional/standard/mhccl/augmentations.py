"""MHCCL augmentation wiring.

Provides ``_default_mhccl_pair()``, which assembles the view pair the reference
implementation uses. The reference vendors TS-TCC's augmentation module verbatim
and computes both views in ``dataloaderq/augmentations.py::DataTransform``::

    aug_1 = scaling(sample, sigma=1.1)
    aug_2 = jitter(permutation(sample, max_segments=8), sigma=0.8)

The constants are TS-TCC's, so this builder is presently identical to
:func:`_default_tstcc_pair`. It is sourced here rather than imported because the
two models' defaults answer to different upstreams and may diverge.

The weak view drives the query encoder and the strong view the momentum encoder,
which is the asymmetry :class:`RolePairProducer` exists to express.

MHCCL operates on tensors of shape ``(batch, time, channels)``, so the
parameters below use ``channel_dim=-1`` and ``time_dim=1``.
"""

from __future__ import annotations

__all__ = ["_default_mhccl_pair"]

from typing import TYPE_CHECKING

from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)
from chronocratic.models.augmentation.producers import RolePairProducer

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair


def _default_mhccl_pair() -> AugmentationProducer[ViewPair]:
    """Build the default MHCCL weak/strong augmentation pair.

    Returns a :class:`RolePairProducer` whose first view applies Gaussian
    scaling (weak) and whose second applies segment permutation followed by
    jitter (strong).

    One deliberate divergence, commented at its line: the scale factor is drawn
    per sample rather than once per batch.

    Returns:
        A producer that returns :class:`ViewPair` instances when
        :meth:`~AugmentationProducer.produce` is called.
    """
    return RolePairProducer(
        # DIVERGENCE: ``per_sample=True``. The reference draws one scale factor
        # and applies it to every sample, which is sound there because it
        # augments the whole training set once, offline, so the factor still
        # varies across the corpus. Under this library's per-batch producer
        # contract a literal port would redraw one factor per batch and apply it
        # uniformly within that batch — a constant rescale that supplies no
        # per-sample variation for the contrastive terms to exploit.
        first=Scaling(ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=-1)),
        second=ComposeAugmentation(
            [
                Permutation(PermutationParameters(max_segments=8, time_dim=1)),
                Jitter(JitterParameters(sigma=0.8)),
            ]
        ),
    )
