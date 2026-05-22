"""Augmentation package — barrel re-export.

Re-exports all symbols from ``base.py`` and per-model augmentation modules.
Uses lazy imports to avoid circular dependencies when per-model modules
(cost, ts2vec, autotcl) load through their __init__.py barrels.

New code should import concrete augmentations from per-model directories:
    - TS2Vec: ``ts2vec/augmentation.py``
    - CoST: ``cost/augmentation.py``
    - AutoTCL: ``autotcl/augmentation/`` package
"""

from __future__ import annotations

from typing import Any

# Always available — ABCs from base.py (no circular dependency)
from .base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)

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
    'RIPTrainingStrategy',
    'TrainableAugmentation',
    'TrainingViews',
]


def __getattr__(name: str) -> Any:  # noqa: ANN401, PLR0911
    """Lazy import of concrete augmentations and params.

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
