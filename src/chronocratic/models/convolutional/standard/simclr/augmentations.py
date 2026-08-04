"""SimCLR augmentation wiring.

Provides ``_default_simclr_pair()``, which assembles the view pair the
reference implementation uses. The reference computes both views in
``ULTS/data_loader/augmentations.py::DataTransform``::

    aug_1 = scaling(sample, sigma=1.1)
    aug_2 = jitter(permutation(sample, max_segments=5, seg_mode="random"), sigma=0.8)

SimCLR operates on tensors of shape ``(batch, time, channels)``, so the
parameters below use ``channel_dim=-1`` and ``time_dim=1``.
"""

from __future__ import annotations

__all__ = ["_default_simclr_pair"]

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


def _default_simclr_pair() -> AugmentationProducer[ViewPair]:
    """Build the default SimCLR weak/strong augmentation pair.

    Returns a :class:`RolePairProducer` whose first view applies Gaussian
    scaling (weak) and whose second view applies segment permutation followed
    by jitter (strong). The reference notes this pairing follows TS-TCC; the
    constants differ from :func:`_default_tstcc_pair` in two places, so they
    are sourced here rather than shared: the scale factor is centred on
    ``1.0`` (TS-TCC uses ``2.0``) and permutation draws from at most ``5``
    segments (TS-TCC uses ``8``).

    Two deliberate divergences, both on the scaling view and each commented at
    its line below.

    ``per_sample=True``. The reference draws one scale factor per channel and
    applies it to every sample. That is sound in context: it augments the
    training set once, offline, so the factor still varies across the corpus.
    Under this library's per-batch producer contract a literal port would
    instead redraw a single factor per batch and apply it uniformly within
    that batch, which is a constant rescale that a normalizing encoder largely
    removes and that supplies no per-sample variation for the loss to exploit.

    ``sigma=0.1`` rather than the reference's ``1.1``, which draws a
    substantial fraction of its scale factors negative.

    Returns:
        A producer that returns :class:`ViewPair` instances when
        :meth:`~AugmentationProducer.produce` is called.
    """
    return RolePairProducer(
        # DIVERGENCE: ``sigma=0.1`` where the reference's ``DataTransform``
        # passes ``1.1``. The scale factor is drawn from ``N(mean, sigma)``,
        # which is unbounded below, so ``sigma=1.1`` about ``mean=1.0`` places
        # roughly 18% of the distribution below zero. Those draws invert the
        # sign of the series they scale, so the pair no longer represents the
        # same underlying signal and the contrastive objective has nothing to
        # discriminate. ``0.1`` is the default of the reference's own
        # ``scaling()`` in ``data_loader/augmentations.py``, overridden only by
        # ``DataTransform``; that module documents the intended range as
        # ``[0.1, 2.0]``. The invariant that matters is ``mean - 3 * sigma > 0``,
        # asserted in ``tests/unit/test_simclr.py``.
        #
        # DIVERGENCE: ``per_sample=True``. See this function's docstring — under
        # a per-batch producer contract the reference's single shared factor
        # degenerates into a constant rescale of the batch.
        first=Scaling(ScalingParameters(sigma=0.1, mean=1.0, per_sample=True, channel_dim=-1)),
        second=ComposeAugmentation(
            [
                Permutation(PermutationParameters(max_segments=5, time_dim=1)),
                Jitter(JitterParameters(sigma=0.8)),
            ]
        ),
    )
