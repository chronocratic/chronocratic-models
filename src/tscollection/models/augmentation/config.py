"""Augmentation parameter configs — backward-compatibility barrel.

The parameter dataclasses have been moved to per-model augmentation
modules. This file re-exports them for backward compatibility via lazy
imports to avoid circular dependencies.

New code should import from the per-model locations:
    - TS2Vec: ``ts2vec/augmentation.py``
    - CoST: ``cost/augmentation.py``
    - AutoTCL: ``autotcl/augmentation/methods.py``
"""

from __future__ import annotations

from typing import Any

__all__ = [  # noqa: F822
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentationParameters',
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy import of parameter dataclasses.

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

    msg = f"module 'tscollection.models.augmentation.config' has no attribute '{name}'"
    raise AttributeError(msg)
