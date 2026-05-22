"""Central re-export of augmentation parameter dataclasses.

Provides backward-compatible import paths for augmentation parameter
dataclasses that now live in per-model augmentation files. Consumers
can use::

    from tscollection.models.configs.augmentation.methods import CropShiftAugmentationParameters

instead of the per-model paths.
"""

__all__ = [
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentationParameters',
]

from tscollection.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentationParameters,
)
from tscollection.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentationParameters,
)
from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
    CropShiftAugmentationParameters,
)
