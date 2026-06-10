"""Augmentation package — abstract types only.

This package owns the augmentation ABCs that the rest of the codebase
codes against:

- :class:`AugmentationMethod` / :class:`TrainableAugmentation` /
  :class:`AugmentationTrainingStrategy` / :class:`TrainingViews` from
  ``base.py``.
- :class:`PairedAugmentation` from ``composition.py`` — the abstract
  two-view contract used by contrastive setups.

Concrete augmentations live alongside the models that use them:

- TS2Vec: ``ts2vec/augmentation.py``
- CoST: ``cost/augmentation.py``
- AutoTCL: ``autotcl/augmentation/`` package
- TS-TCC: ``ts_tcc/augmentations.py``

The lazy ``__getattr__`` below re-exports the per-model concrete classes
through this barrel for callers that prefer one import path, without
introducing circular dependencies during package load.
"""

from __future__ import annotations

from typing import Any

from .base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)
from .composition import PairedAugmentation

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
    'PairedAugmentation',
    'RIPTrainingStrategy',
    'TrainableAugmentation',
    'TrainingViews',
]


def __getattr__(name: str) -> Any:  # noqa: ANN401, PLR0911
    """Lazy import of concrete per-model augmentations.

    Defers imports until first access, breaking the circular dependency
    chain when per-model __init__.py files trigger during package load.
    """
    if name == 'CropShiftAugmentationParameters':
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
            CropShiftAugmentationParameters,
        )

        return CropShiftAugmentationParameters
    if name == 'CosTRandomFunctionAugmentationParameters':
        from tscollection.models.convolutional.dilated.cost.augmentation import (  # noqa: PLC0415
            CosTRandomFunctionAugmentationParameters,
        )

        return CosTRandomFunctionAugmentationParameters
    if name == 'AutoTCLNeuralNetworkAugmentationParameters':
        from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (  # noqa: PLC0415
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        return AutoTCLNeuralNetworkAugmentationParameters
    if name == 'CropShiftAugmentation':
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (  # noqa: PLC0415
            CropShiftAugmentation,
        )

        return CropShiftAugmentation
    if name == 'CosTRandomFunctionAugmentation':
        from tscollection.models.convolutional.dilated.cost.augmentation import (  # noqa: PLC0415
            CosTRandomFunctionAugmentation,
        )

        return CosTRandomFunctionAugmentation
    if name == 'AutoTCLNeuralNetworkAugmentation':
        from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (  # noqa: PLC0415
            AutoTCLNeuralNetworkAugmentation,
        )

        return AutoTCLNeuralNetworkAugmentation
    if name == 'RIPTrainingStrategy':
        from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (  # noqa: PLC0415
            RIPTrainingStrategy,
        )

        return RIPTrainingStrategy
    if name == 'AdversarialTrainingStrategy':
        from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (  # noqa: PLC0415
            AdversarialTrainingStrategy,
        )

        return AdversarialTrainingStrategy

    msg = f"module 'tscollection.models.augmentation' has no attribute '{name}'"
    raise AttributeError(msg)
