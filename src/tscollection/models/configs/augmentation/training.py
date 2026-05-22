"""Central re-export of augmentation training strategies and base ABC.

Provides backward-compatible import paths for training strategy classes
that now live in per-model augmentation files and the shared base module.
Consumers can use::

    from tscollection.models.configs.augmentation.training import RIPTrainingStrategy

instead of the per-model paths.
"""

__all__ = [
    'AdversarialTrainingStrategy',
    'AugmentationTrainingStrategy',
    'RIPTrainingStrategy',
]

from tscollection.models.augmentation.base import AugmentationTrainingStrategy
from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (
    AdversarialTrainingStrategy,
    RIPTrainingStrategy,
)
