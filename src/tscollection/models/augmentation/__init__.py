"""Augmentation package — abstract types and concrete re-exports.

This package owns the augmentation ABCs that the rest of the codebase
codes against:

- :class:`AugmentationMethod` / :class:`TrainableAugmentation` /
  :class:`AugmentationTrainingStrategy` / :class:`TrainingViews` from
  ``base.py``.
- :class:`DualAugmentation` from ``dual.py`` — the abstract
  two-view contract used by contrastive setups.

Concrete augmentations live alongside the models that use them but are
re-exported here for callers that prefer a single import path:

- TS2Vec: ``ts2vec/augmentation.py``
- CoST: ``cost/augmentation.py``
- AutoTCL: ``autotcl/augmentation/`` package
- TS-TCC: ``tstcc/augmentations.py``
"""

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
    CropShiftAugmentation,
    CropShiftAugmentationParameters,
)

from .base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)
from .dual import DualAugmentation

__all__ = [
    'AdversarialTrainingStrategy',
    'AugmentationMethod',
    'AugmentationTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentation',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentation',
    'CropShiftAugmentationParameters',
    'DualAugmentation',
    'RIPTrainingStrategy',
    'TrainableAugmentation',
    'TrainingViews',
]
