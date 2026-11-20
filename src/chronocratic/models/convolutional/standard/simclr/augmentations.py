"""SimCLR augmentation wiring.

Provides ``_default_simclr_pair()``, which builds the default
:class:`~chronocratic.models.convolutional.standard.augmentation.PerTimestepScaling`
(weak) | ``Jitter`` (strong) pair, with segment permutation off unless
requested. The reference computes both views in
``ULTS/data_loader/augmentations.py::DataTransform``::

    aug_1 = scaling(sample, sigma=1.1)
    aug_2 = jitter(permutation(sample, max_segments=5, seg_mode="random"), sigma=0.8)

SimCLR operates on tensors of shape ``(batch, time, channels)``, so the
builders in this module use ``time_dim=1``.
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
)
from chronocratic.models.augmentation.producers import RolePairProducer
from chronocratic.models.convolutional.standard.augmentation import PerTimestepScaling

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import Augmentation, AugmentationProducer, ViewPair


def _default_simclr_pair(
    *, max_segments: int = 1, jitter_sigma: float = 0.8
) -> AugmentationProducer[ViewPair]:
    """Build the default SimCLR weak/strong augmentation pair.

    The weak view applies :class:`PerTimestepScaling`. The strong view applies
    :class:`Jitter`, preceded by segment :class:`Permutation` only when
    ``max_segments > 1``. ``max_segments=5`` gives the reference's strong
    view.

    Two deliberate divergences from the reference:

    **Permutation is off by default.** SimCLR's loss compares the two views
    timestep by timestep (``forward`` projects every timestep and
    concatenates), so permuted segments can only be matched by features that
    look the same at every position. Training then discards *where* things
    happen in the series, which on data with position-dependent structure
    leaves the encoder below an untrained one. The loss still falls well
    below chance, so a loss-based check does not see it. On series without
    position-dependent structure permutation is harmless.

    **The weak view scales every timestep independently.** The reference's
    ``scaling()`` draws one factor per channel, shared across the series
    (and, as the reference applies it offline, across the training set).
    The encoder's ``GroupNorm`` cancels a factor that is constant over time,
    so that view is nearly the identity and the pair reduces to the raw
    series against a jittered one. :class:`PerTimestepScaling` is the
    per-timestep scaling of the TS-TCC reference, which the SimCLR reference
    says this pairing follows.

    Args:
        max_segments: Upper bound on permutation segments. ``1`` disables
            permutation.
        jitter_sigma: Std of the strong view's additive Gaussian noise. It
            is absolute, so it assumes unit-scale input, which SimCLR's
            ``instance_normalize`` default provides.

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
    return RolePairProducer(first=PerTimestepScaling(), second=strong)
