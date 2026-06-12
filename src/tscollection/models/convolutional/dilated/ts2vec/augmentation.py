"""TS2Vec augmentation: crop-and-shift producer.

Contains the ``CropShiftProducer`` class (reshaped from
``CropShiftAugmentation``) and its ``CropShiftAugmentationParameters``
dataclass. Returns :class:`AlignedPair` instead of
:class:`TrainingViews`, eliminating the metadata dict dependency
(SPEC §2.3 root cause #1, §4.7, §5).

Imports ``AugmentationProducer`` and ``AlignedPair`` directly from
``augmentation/base.py`` (NOT the barrel) to avoid circular dependencies.
"""

__all__ = ['CropShiftAugmentation', 'CropShiftAugmentationParameters', 'CropShiftProducer']

from dataclasses import dataclass

import numpy as np
import torch

from tscollection.models.augmentation.base import AlignedPair


@dataclass
class CropShiftAugmentationParameters:
    """Parameters for :class:`CropShiftProducer`.

    Controls the temporal granularity of the random crop-and-shift
    augmentation used by TS2Vec.

    Args:
        temporal_unit: Controls the minimum crop length as
            ``2 ** (temporal_unit + 1)``. Defaults to ``0``.
    """

    temporal_unit: int = 0


class CropShiftProducer:
    """Random crop-and-shift producer used by TS2Vec.

    Produces two overlapping random crops of the input tensor, applying
    independent per-sample temporal offsets. Returns an
    :class:`AlignedPair` with ``overlap_length`` set to the crop length,
    eliminating the need for a metadata dict.

    Satisfies ``AugmentationProducer[AlignedPair]`` structurally.

    Args:
        params: Optional configuration controlling the temporal unit.
            When ``None``, defaults to ``CropShiftAugmentationParameters()``.
    """

    def __init__(self, params: CropShiftAugmentationParameters | None = None) -> None:
        """Initialize the crop-and-shift producer.

        Args:
            params: Optional configuration controlling the temporal unit.
                When ``None``, defaults to ``CropShiftAugmentationParameters()``.
        """
        self._params = params if params is not None else CropShiftAugmentationParameters()

    def produce(self, x: torch.Tensor) -> AlignedPair:
        """Return two overlapping random crops of ``x`` with random per-sample shifts.

        A crop window is sampled uniformly, then extended in both directions.
        Each sample in the batch receives an independent random temporal offset,
        producing two overlapping subsequences that share a guaranteed common
        sub-interval of length ``crop_length``.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            AlignedPair with first, second tensors and overlap_length
            (replacing the old metadata['crop_length'] pattern).
        """
        # Lazy import to avoid circular dependency:
        # ts2vec/model.py imports from augmentation/strategies.py, so a module-level
        # import of ts2vec/utils.py would create a circular import chain.
        from tscollection.models.convolutional.dilated.ts2vec.utils import (  # noqa: PLC0415
            extract_subsequences_per_row,
        )

        temporal_unit = self._params.temporal_unit

        total_length = x.size(1)
        min_crop_length = 2 ** (temporal_unit + 1)

        if min_crop_length >= total_length:
            msg = (
                f'Crop minimum length ({min_crop_length}) exceeds input '
                f'time dimension ({total_length}). Reduce temporal_unit '
                f'or provide longer sequences.'
            )
            raise ValueError(msg)

        # Randomly determine the length of the crop
        crop_length = np.random.randint(  # noqa: NPY002
            low=min_crop_length, high=total_length + 1
        )

        # Randomly determine the starting and ending points for the crops
        crop_start = np.random.randint(  # noqa: NPY002
            total_length - crop_length + 1
        )
        crop_end = crop_start + crop_length
        crop_extension_start = np.random.randint(crop_start + 1)  # noqa: NPY002
        crop_extension_end = np.random.randint(  # noqa: NPY002
            low=crop_end, high=total_length + 1
        )

        # Random offset for each sample in the batch
        crop_offsets = np.random.randint(  # noqa: NPY002
            low=-crop_extension_start, high=total_length - crop_extension_end + 1, size=x.size(0)
        )

        # Generate augmented subsequences 1 by cropping and shifting
        augmented_subsequences_1 = extract_subsequences_per_row(
            array=x,
            indices=crop_offsets + crop_extension_start,
            num_elements=crop_end - crop_extension_start,
        )

        # Generate augmented subsequences 2 by cropping and shifting
        augmented_subsequences_2 = extract_subsequences_per_row(
            array=x, indices=crop_offsets + crop_start, num_elements=crop_extension_end - crop_start
        )

        return AlignedPair(
            first=augmented_subsequences_1,
            second=augmented_subsequences_2,
            overlap_length=crop_length,
        )


# D-05: Backward compat alias — keep until final delete commit
CropShiftAugmentation = CropShiftProducer
