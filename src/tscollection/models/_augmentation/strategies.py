__all__ = [
    'AugmentationMethod',
    'AutoTCLNeuralNetworkAugmentation',
    'CoSTAugmentationMethod',
    'CosTRandomFunctionAugmentation',
    'CropShiftAugmentation',
]

from abc import ABC, abstractmethod
import random

import numpy as np
import torch

from tscollection.models.encoders import AutoTCLAugmentationTimeSeriesEncoder


class AugmentationMethod(ABC):
    """Abstract base class for all time-series augmentation strategies."""

    @abstractmethod
    def _setup(self) -> None:
        """Initialise internal state after construction."""

    @abstractmethod
    def augment(
        self,
        data: torch.Tensor,
        **kwargs,  ## noqa: ANN003
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int]:
        """Return one or two augmented views of ``data``.

        Args:
            data: Input time-series tensor of shape ``(batch, time, channels)``.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            A single augmented tensor, or a tuple of two augmented tensors and
            an integer (e.g. crop length) for strategies that produce paired views.
        """

    @abstractmethod
    def get_model(self) -> object:
        """Return the underlying model or callable used for augmentation."""


class CoSTAugmentationMethod(AugmentationMethod, ABC):
    """Narrowed base for augmentation strategies compatible with CoST.

    Guarantees that ``augment`` returns a plain ``torch.Tensor``, which
    ``CoST._compute_total_loss`` requires.
    """

    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  ## noqa: ANN003
        """Return a single augmented view of ``data``."""

class AutoTCLNeuralNetworkAugmentation(AugmentationMethod):
    """Augmentation driven by a learned neural network (AutoTCL)."""

    def __init__(self, params: dict) -> None:
        """Initialise the neural-network augmentation.

        Args:
            params: Keyword arguments forwarded to
                ``AutoTCLAugmentationTimeSeriesEncoder``.
        """
        self.params = params
        self.model: AutoTCLAugmentationTimeSeriesEncoder
        self._setup()

    def _build_model(self) -> None:
        """Instantiate the underlying encoder model."""
        self.model = AutoTCLAugmentationTimeSeriesEncoder(**self.params)

    def _setup(self) -> None:
        self._build_model()

    def augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ANN003 ARG002
        """Return an augmented view produced by the encoder model.

        Args:
            data: Input time-series tensor.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            Augmented tensor from the encoder's ``augment`` method.
        """
        return self.model.augment(data)

    def get_model(self) -> AutoTCLAugmentationTimeSeriesEncoder:
        """Return the underlying ``AutoTCLAugmentationTimeSeriesEncoder``."""
        return self.model


class CropShiftAugmentation(AugmentationMethod):
    """Augmentation method used by TS2Vec."""

    def _setup(self) -> None:
        pass

    def augment(self, data: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, int]:  # noqa: ANN003
        """Return two overlapping random crops of ``data`` with random per-sample shifts.

        A crop window is sampled uniformly, then extended in both directions.
        Each sample in the batch receives an independent random temporal offset,
        producing two overlapping subsequences that share a guaranteed common
        sub-interval of length ``crop_length``.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.
            **kwargs:
                temporal_unit (int): Controls the minimum crop length as
                    ``2 ** (temporal_unit + 1)``. Defaults to ``0``.

        Returns:
            A 3-tuple ``(view1, view2, crop_length)`` where ``view1`` and
            ``view2`` are the two augmented subsequence tensors and
            ``crop_length`` is the length of their shared interval.
        """
        from tscollection.models.ts2vec.utils import extract_subsequences_per_row

        temporal_unit = kwargs.get('temporal_unit', 0)
        x = data

        total_length = x.size(1)

        # Randomly determine the length of the crop
        crop_length = np.random.randint(low=2 ** (temporal_unit + 1), high=total_length + 1)  # noqa: NPY002

        # Randomly determine the starting and ending points for the crops
        crop_start = np.random.randint(total_length - crop_length + 1)  # noqa: NPY002
        crop_end = crop_start + crop_length
        crop_extension_start = np.random.randint(crop_start + 1)  # noqa: NPY002
        crop_extension_end = np.random.randint(low=crop_end, high=total_length + 1)  # noqa: NPY002

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

        return augmented_subsequences_1, augmented_subsequences_2, crop_length

    def get_model(self) -> None:
        """No underlying model; augmentation is purely algorithmic."""


class CosTRandomFunctionAugmentation(CoSTAugmentationMethod):
    """Augmentation method used by CoST."""

    def __init__(self, params: dict) -> None:
        """Initialise the random-function augmentation.

        Args:
            params: Must contain ``sigma`` (noise scale). Optionally ``p``
                (probability of applying each transform, default ``0.5``).
        """
        self.params = params
        self._setup()

    def _setup(self) -> None:
        self._sigma = self.params['sigma']
        self._p = self.params.get('p', 0.5)

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with std ``sigma`` with probability ``p``."""
        if random.random() > self._p:  # noqa: S311
            return x
        return x + (torch.randn(x.shape, device=x.device) * self._sigma)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by a Gaussian factor around 1 with probability ``p``."""
        if random.random() > self._p:  # noqa: S311
            return x
        return x * (torch.randn(x.size(-1), device=x.device) * self._sigma + 1)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Add a per-channel Gaussian offset with probability ``p``."""
        if random.random() > self._p:  # noqa: S311
            return x
        return x + (torch.randn(x.size(-1), device=x.device) * self._sigma)

    def augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ANN003 ARG002
        """Return ``data`` after stochastically applying scale, shift, and jitter.

        Each of the three transforms is applied independently with probability
        ``p``. The composition order is scale → shift → jitter.

        Args:
            data: Input time-series tensor.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            Augmented tensor of the same shape as ``data``.
        """
        return self._jitter(self._shift(self._scale(data)))

    def get_model(self) -> None:
        """No underlying model; augmentation is purely algorithmic."""
