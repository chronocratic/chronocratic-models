__all__ = [
    'AutoTCLAugmentationMode',
    'AutoTCLNeuralNetworkAugmentationTrainingMode',
    'CoSTAugmentationMode',
    'TS2VecAugmentationMode',
]

from enum import Enum


class AutoTCLNeuralNetworkAugmentationTrainingMode(Enum):
    RELEVANT_INFORMATION_PRINCIPLE = 'relevant_information_principle'
    ADVERSARIAL = 'adversarial'


class AutoTCLAugmentationMode(Enum):
    NEURAL_NETWORK = 'neural_network'


class TS2VecAugmentationMode(Enum):
    CROP_SHIFT = 'crop_shift'


class CoSTAugmentationMode(Enum):
    RANDOM_FUNCTIONS = 'random_functions'
