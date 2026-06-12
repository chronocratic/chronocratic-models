"""Augmentation package — abstract types and concrete re-exports.

This package provides the new producer contract and retains the legacy
augmentation symbols for backward compatibility.

**New contract** (preferred):

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

**Legacy symbols** (retained until plan 01-11 deletes them):

- :class:`AugmentationMethod` / :class:`TrainableAugmentation` /
  :class:`TrainingViews` / :class:`DualAugmentation`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# New contract — base types and ViewSets
# ---------------------------------------------------------------------------
from .base import (
    AlignedPair,
    Augmentation,
    AugmentationProducer,
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
# Legacy symbols (retained for backward compatibility — D-05)
# ---------------------------------------------------------------------------
from .base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)
from .dual import DualAugmentation

# ---------------------------------------------------------------------------
# Per-model concrete augmentations (re-exported for convenience)
# ---------------------------------------------------------------------------
from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (
    AdversarialTrainingStrategy,
    RIPTrainingStrategy,
)
from tscollection.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentation,
    CosTRandomFunctionAugmentationParameters,
)
from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
    CropShiftAugmentationParameters,
    CropShiftProducer,
)

# D-05: Backward compat alias — Barrel only (ts2vec/augmentation.py no longer exports it)
CropShiftAugmentation = CropShiftProducer  # type: ignore[misc]

__all__ = [
    # New contract — base
    'AlignedPair',
    'Augmentation',
    'AugmentationProducer',
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
    # Legacy (D-05)
    'AugmentationMethod',
    'AugmentationTrainingStrategy',
    'DualAugmentation',
    'TrainableAugmentation',
    'TrainingViews',
    # Per-model concrete augmentations
    'AdversarialTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentation',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentation',
    'CropShiftAugmentationParameters',
    'RIPTrainingStrategy',
]
