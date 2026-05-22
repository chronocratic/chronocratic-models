"""Augmentation strategies — backward-compatibility barrel.

The ABCs (``TrainingViews``, ``AugmentationMethod``,
``AugmentationTrainingStrategy``, ``TrainableAugmentation``) are now defined
in ``base.py``. Concrete implementations have been moved to per-model
augmentation modules. This file re-exports everything for backward
compatibility so existing import paths continue to work.

Uses lazy imports to avoid circular dependencies when per-model modules
(cost, ts2vec, autotcl) load through their __init__.py barrels.

New code should import from the per-model locations:
    - TS2Vec: ``ts2vec/augmentation.py``
    - CoST: ``cost/augmentation.py``
    - AutoTCL: ``autotcl/augmentation/`` package
"""

from __future__ import annotations

from typing import Any

from tscollection.models.augmentation.base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)

__all__ = [  # noqa: F822
    'AdversarialTrainingStrategy',
    'AugmentationMethod',
    'AugmentationTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'CosTRandomFunctionAugmentation',
    'CropShiftAugmentation',
    'RIPTrainingStrategy',
    'TrainableAugmentation',
    'TrainingViews',
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import of concrete augmentations and strategies.

    Defers imports until first access, breaking the circular dependency
    chain when per-model __init__.py files trigger during package load.
    """
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

    msg = f"module 'tscollection.models.augmentation.strategies' has no attribute '{name}'"
    raise AttributeError(msg)
