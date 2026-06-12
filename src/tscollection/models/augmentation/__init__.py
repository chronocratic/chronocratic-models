"""Augmentation package — abstract types and concrete re-exports.

This package provides the producer contract for augmentation strategies.

**Contract types**:

- :class:`Augmentation` / :class:`AugmentationProducer` /
  :class:`TrainableAugmentationProducer` from ``base.py``.
- :class:`SingleView` / :class:`ViewPair` / :class:`AlignedPair` typed
  ViewSets from ``base.py``.
- :class:`SingleViewProducer` / :class:`IndependentPair` /
  :class:`RolePair` / :class:`FullOverlapPair` from ``producers.py``.
- :class:`Seeded` decorator from ``decorators.py``.
- :func:`maybe_train_augmentation` / :func:`maybe_configure_augmentation_optimizer`
  from ``trainable_support.py``.
- Shared primitives (:class:`Jitter`, :class:`Scaling`,
  :class:`Permutation`, :class:`ComposeAugmentation`) from ``primitives.py``.
"""

from __future__ import annotations

from .base import (
    AlignedPair,
    Augmentation,
    AugmentationProducer,
    AugmentationTrainingStrategy,
    SingleView,
    TrainableAugmentationProducer,
    ViewPair,
)
from .decorators import Seeded
from .primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)
from .producers import (
    FullOverlapPair,
    IndependentPair,
    RolePair,
    SingleViewProducer,
)
from .trainable_support import (
    maybe_configure_augmentation_optimizer,
    maybe_train_augmentation,
)

__all__ = [
    'AlignedPair',
    'Augmentation',
    'AugmentationProducer',
    'AugmentationTrainingStrategy',
    'ComposeAugmentation',
    'FullOverlapPair',
    'IndependentPair',
    'Jitter',
    'JitterParameters',
    'Permutation',
    'PermutationParameters',
    'RolePair',
    'Scaling',
    'ScalingParameters',
    'Seeded',
    'SingleView',
    'SingleViewProducer',
    'TrainableAugmentationProducer',
    'ViewPair',
    'maybe_configure_augmentation_optimizer',
    'maybe_train_augmentation',
]
