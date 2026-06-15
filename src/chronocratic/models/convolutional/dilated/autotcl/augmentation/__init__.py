"""AutoTCL augmentation package.

Barrel re-exports for the per-model augmentation module.
"""

__all__ = [
    'AdversarialTrainingStrategy',
    'AutoTCLNeuralNetworkAugmentation',
    'AutoTCLNeuralNetworkAugmentationParameters',
    'RIPTrainingStrategy',
]

from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
    AdversarialTrainingStrategy,
    RIPTrainingStrategy,
)
