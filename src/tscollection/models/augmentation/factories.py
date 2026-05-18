__all__ = [
    'AutoTCLAugmentationMethodFactory',
    'CoSTAugmentationMethodFactory',
    'TS2VecAugmentationMethodFactory',
]

from src.tscollection.models.augmentation.enums import (
    AutoTCLAugmentationMode,
    CoSTAugmentationMode,
    TS2VecAugmentationMode,
)
from src.tscollection.models.augmentation.strategies import (
    AugmentationMethod,
    AutoTCLNeuralNetworkAugmentation,
    CoSTAugmentationMethod,
    CosTRandomFunctionAugmentation,
    CropShiftAugmentation,
)


class AutoTCLAugmentationMethodFactory:
    @staticmethod
    def get_augmentation_method(mode: AutoTCLAugmentationMode, params: dict) -> AugmentationMethod:
        """Return the augmentation strategy for an AutoTCL model.

        Args:
            mode: Selects the augmentation strategy — neural network.
            params: Hyperparameters forwarded to the chosen strategy.

        Returns:
            An ``AugmentationMethod`` instance configured for the given mode.

        Raises:
            ValueError: If ``mode`` is not a recognised ``AutoTCLAugmentationMode``.
        """
        if mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return AutoTCLNeuralNetworkAugmentation(params=params)
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)


class TS2VecAugmentationMethodFactory:
    @staticmethod
    def get_augmentation_method(mode: TS2VecAugmentationMode, params: dict) -> AugmentationMethod:
        """Return the augmentation strategy for a TS2Vec model.

        Args:
            mode: Selects the augmentation strategy — crop-and-shift.
            params: Hyperparameters forwarded to the chosen strategy. Ignored
                for ``CROP_SHIFT``, which requires no configuration.

        Returns:
            An ``AugmentationMethod`` instance configured for the given mode.

        Raises:
            ValueError: If ``mode`` is not a recognised ``TS2VecAugmentationMode``.
        """
        if mode == TS2VecAugmentationMode.CROP_SHIFT:
            return CropShiftAugmentation()
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)


class CoSTAugmentationMethodFactory:
    @staticmethod
    def get_augmentation_method(mode: CoSTAugmentationMode, params: dict) -> CoSTAugmentationMethod:
        """Return the augmentation strategy for a CoST model.

        Args:
            mode: Selects the augmentation strategy — random jitter/scale/shift
                functions.
            params: Hyperparameters forwarded to the chosen strategy.

        Returns:
            An ``AugmentationMethod`` instance configured for the given mode.

        Raises:
            ValueError: If ``mode`` is not a recognised ``CoSTAugmentationMode``.
        """
        if mode == CoSTAugmentationMode.RANDOM_FUNCTIONS:
            return CosTRandomFunctionAugmentation(params=params)
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)
