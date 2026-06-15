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

**Concrete augmentations** (lazy-imported for backward compatibility):

- :class:`CropShiftProducer` — TS2Vec crop-and-shift producer.
- :class:`CosTRandomFunctionAugmentation` — CoST random-function augmentation.
- :class:`AutoTCLNeuralNetworkAugmentation` — AutoTCL neural-network augmentation.
- :class:`RIPTrainingStrategy` — AutoTCL RIP training strategy.
- :class:`AdversarialTrainingStrategy` — AutoTCL adversarial training strategy.
"""

from __future__ import annotations

from typing import Any

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
    'AdversarialTrainingStrategy',
    'AlignedPair',
    'Augmentation',
    'AugmentationProducer',
    'AugmentationTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'ComposeAugmentation',
    'CosTRandomFunctionAugmentation',
    'CropShiftProducer',
    'FullOverlapPair',
    'IndependentPair',
    'Jitter',
    'JitterParameters',
    'Permutation',
    'PermutationParameters',
    'RIPTrainingStrategy',
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


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import of concrete augmentations.

    Defers imports until first access, breaking the circular dependency
    chain when per-model __init__.py files trigger during package load.
    """
    if name == 'CropShiftProducer':
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
            CropShiftProducer,
        )

        return CropShiftProducer
    if name == 'CosTRandomFunctionAugmentation':
        from chronocratic.models.convolutional.dilated.cost.augmentation import (  # noqa: PLC0415
            CosTRandomFunctionAugmentation,
        )

        return CosTRandomFunctionAugmentation
    if name == 'AutoTCLNeuralNetworkAugmentation':
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (  # noqa: PLC0415
            AutoTCLNeuralNetworkAugmentation,
        )

        return AutoTCLNeuralNetworkAugmentation
    if name in {'RIPTrainingStrategy', 'AdversarialTrainingStrategy'}:
        from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (  # noqa: PLC0415
            AdversarialTrainingStrategy,
            RIPTrainingStrategy,
        )

        return RIPTrainingStrategy if name == 'RIPTrainingStrategy' else AdversarialTrainingStrategy

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
