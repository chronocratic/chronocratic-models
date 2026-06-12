"""Augmentation package — abstract types and concrete re-exports.

This package provides the producer contract for augmentation strategies.

**New contract**:

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

# ---------------------------------------------------------------------------
# New contract — base types and ViewSets
# ---------------------------------------------------------------------------
from .base import (
    AlignedPair,
    Augmentation,
    AugmentationProducer,
    AugmentationTrainingStrategy,
    SingleView,
    TrainableAugmentationProducer,
    ViewPair,
)

# ---------------------------------------------------------------------------
# New contract — shared primitives
# ---------------------------------------------------------------------------
from .primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)

# ---------------------------------------------------------------------------
# New contract — producer combinators
# ---------------------------------------------------------------------------
from .producers import (
    FullOverlapPair,
    IndependentPair,
    RolePair,
    SingleViewProducer,
)

# ---------------------------------------------------------------------------
# New contract — decorator and trainable helpers
# ---------------------------------------------------------------------------
from .decorators import Seeded
from .trainable_support import (
    maybe_configure_augmentation_optimizer,
    maybe_train_augmentation,
)

# ---------------------------------------------------------------------------
# Per-model concrete augmentations (re-exported for convenience)
# ---------------------------------------------------------------------------
# NOTE: cost/augmentation.py still imports deleted TrainingViews — will be
# fixed in plan 13. Its re-exports are deferred until then.
from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (
    AdversarialTrainingStrategy,
    RIPTrainingStrategy,
)
from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
    CropShiftAugmentationParameters,
    CropShiftProducer,
)


__all__ = [
    # New contract — base
    'AlignedPair',
    'Augmentation',
    'AugmentationProducer',
    'AugmentationTrainingStrategy',
    'SingleView',
    'TrainableAugmentationProducer',
    'ViewPair',
    # New contract — primitives
    'ComposeAugmentation',
    'Jitter',
    'JitterParameters',
    'Permutation',
    'PermutationParameters',
    'Scaling',
    'ScalingParameters',
    # New contract — producers
    'FullOverlapPair',
    'IndependentPair',
    'RolePair',
    'SingleViewProducer',
    # New contract — decorator / helpers
    'Seeded',
    'maybe_configure_augmentation_optimizer',
    'maybe_train_augmentation',
    # Per-model concrete augmentations (cost deferred to plan 13)
    'AdversarialTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CropShiftAugmentationParameters',
    'RIPTrainingStrategy',
]
