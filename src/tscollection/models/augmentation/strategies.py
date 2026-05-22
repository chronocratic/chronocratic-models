__all__ = [
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import AdamW

from tscollection.models.augmentation.config import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
    CropShiftAugmentationParameters,
)
from tscollection.models.cnn.dilated.encoders.encoders import AutoTCLAugmentationTimeSeriesEncoder
from tscollection.models.losses import (
    info_nce_loss,
    maximum_mean_discrepancy_with_gaussian_kernel_loss,
)

# --------------------------------------------------------------------------- #
# TrainingViews
# --------------------------------------------------------------------------- #


@dataclass
class TrainingViews:
    """Container for augmentation output views and metadata.

    The number and shape of views is defined by the model-augmentation contract,
    not enforced at runtime. Models should document their expected view structure.

    Examples:
        TS2Vec: views has 2 tensors, metadata has 'crop_length' (int).
        CoST: views has 1 tensor, metadata is empty.
        AutoTCL: views has 1 tensor, metadata may have 'augmentation_factor'.
    """

    views: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# AugmentationMethod ABC
# --------------------------------------------------------------------------- #


class AugmentationMethod(ABC):
    """Abstract base class for all time-series augmentation strategies.

    Subclass this to create a new augmentation. Implement ``augment()`` to
    define the transform. The model calls this polymorphically -- no enum
    dispatch needed.

    Pure transform. No ``train_step``, no ``configure_optimizer``, no ``_setup``.
    """

    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401
        """Return augmented views of ``data``.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            TrainingViews containing augmented tensor(s) and metadata.
        """
        ...


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy ABC
# --------------------------------------------------------------------------- #


class AugmentationTrainingStrategy(ABC):
    """Defines how a trainable augmentation network is optimized.

    Subclass to create a new training strategy. Implement ``compute_loss()``
    to define the loss function. Override ``should_train()`` for epoch-gated
    schedules.
    """

    @abstractmethod
    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the augmentation network loss.

        Args:
            x_embeddings: Encodings of the original data.
            aug_x_embeddings: Encodings of the augmented data.
            augmentation_factor: Learned augmentation weights/factors.

        Returns:
            Scalar loss tensor requiring gradients.
        """
        ...

    def should_train(self, epoch: int, batch_idx: int) -> bool:  # noqa: ARG002
        """Determine if aug-network training should run this step.

        Default: train every step. Override for epoch-gated schedules.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return True


# --------------------------------------------------------------------------- #
# RIPTrainingStrategy
# --------------------------------------------------------------------------- #


class RIPTrainingStrategy(AugmentationTrainingStrategy):
    """Relevant Information Principle training strategy.

    Minimizes the Maximum Mean Discrepancy between original and augmented
    embeddings, with consistency and regularization penalties on the
    augmentation factor.

    Extracted from AutoTCL's
    ``_augmentation_loss_network_augmentation_relevant_information_principle``.
    """

    def __init__(
        self,
        consistency_weight: float = 0.001,
        regularization_weight: float = 0.001,
        regularization_threshold: float = 0.4,
    ) -> None:
        """Initialize the RIP training strategy.

        Args:
            consistency_weight: Weight for the regular consistency term.
            regularization_weight: Weight for the regularization loss.
            regularization_threshold: Threshold for the regularization term.
        """
        self._consistency_weight = consistency_weight
        self._regularization_weight = regularization_weight
        self._regularization_threshold = regularization_threshold

    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the RIP augmentation loss.

        Args:
            x_embeddings: Encodings of the original data.
            aug_x_embeddings: Encodings of the augmented data.
            augmentation_factor: Learned augmentation weights/factors.

        Returns:
            Scalar loss tensor combining MMD, regularization, and consistency.
        """
        # Lazy import to avoid circular dependency:
        # strategies.py imports from encoders.py, which may import from augmentation/__init__.py,
        # which imports from strategies.py — utils.py also imports from the barrel.
        from tscollection.models.cnn.dilated.autotcl.utils import (  # noqa: PLC0415
            calculate_regular_consistency,
        )

        vx_distance = maximum_mean_discrepancy_with_gaussian_kernel_loss(
            x_embeddings, aug_x_embeddings
        )
        regular_consistency = calculate_regular_consistency(weights=augmentation_factor)
        regularization_loss = F.relu(
            torch.sum(augmentation_factor, dim=-1).mean() - self._regularization_threshold
        )

        return (
            vx_distance
            + self._regularization_weight * regularization_loss
            + self._consistency_weight * regular_consistency
        )


# --------------------------------------------------------------------------- #
# AdversarialTrainingStrategy
# --------------------------------------------------------------------------- #


class AdversarialTrainingStrategy(AugmentationTrainingStrategy):
    """Adversarial training strategy.

    Maximizes the InfoNCE loss between original and augmented embeddings,
    encouraging the augmentation network to produce hard-to-distinguish
    augmented views.

    Extracted from AutoTCL's
    ``_augmentation_loss_neural_network_augmentation_adversarial``.
    """

    def compute_loss(
        self,
        x_embeddings: torch.Tensor,
        aug_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute the adversarial augmentation loss.

        Args:
            x_embeddings: Encodings of the original data.
            aug_x_embeddings: Encodings of the augmented data.
            augmentation_factor: Unused by adversarial strategy.

        Returns:
            Scalar loss tensor (negative InfoNCE).
        """
        return -1.0 * info_nce_loss(x_embeddings, aug_x_embeddings, temperature=1.0)


# --------------------------------------------------------------------------- #
# TrainableAugmentation
# --------------------------------------------------------------------------- #


class TrainableAugmentation(AugmentationMethod, nn.Module, ABC):
    """Augmentation with learnable parameters.

    Composes an ``AugmentationTrainingStrategy`` for loss computation.
    AutoTCL-specific; not a general pattern for TS2Vec/CoST.
    """

    def __init__(self, training_strategy: AugmentationTrainingStrategy) -> None:
        """Initialize a trainable augmentation.

        Args:
            training_strategy: Strategy for computing the augmentation loss.
        """
        super().__init__()
        self._training_strategy = training_strategy

    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401
        """Return an augmented view produced by the encoder model.

        Args:
            data: Input time-series tensor.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            TrainingViews containing the augmented tensor(s) and metadata.
        """
        ...

    def configure_optimizer(self, lr: float) -> AdamW:
        """Return optimizer over this module's parameters.

        Args:
            lr: Learning rate for the augmentation network optimizer.

        Returns:
            AdamW optimizer for this module's parameters.
        """
        return AdamW(self.parameters(), lr=lr)

    def train_step(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor | None:
        """Run one augmentation-network training step.

        Forward pass through aug network to get augmentation_factor and
        augmented_data, encode both inputs through the passed encoder,
        then delegate to strategy.compute_loss().

        Args:
            x: Original input data.
            encoder: The main encoder module to compute embeddings.
            batch_idx: Current batch index (passed to should_train).

        Returns:
            Loss tensor if strategy.should_train(), otherwise None.
        """
        features = self.forward(x)
        augmentation_factor = features['augmentation_factor']
        augmented_x = features['augmented_data']
        x_embeddings = encoder(x)
        aug_x_embeddings = encoder(augmented_x)

        return self._training_strategy.compute_loss(
            x_embeddings=x_embeddings,
            aug_x_embeddings=aug_x_embeddings,
            augmentation_factor=augmentation_factor,
        )


# --------------------------------------------------------------------------- #
# Concrete augmentations
# --------------------------------------------------------------------------- #


class AutoTCLNeuralNetworkAugmentation(TrainableAugmentation):
    """Augmentation driven by a learned neural network (AutoTCL).

    Inherits from ``TrainableAugmentation`` to provide optimizer configuration
    and training-step delegation via a composed ``AugmentationTrainingStrategy``.
    """

    def __init__(
        self,
        params: AutoTCLNeuralNetworkAugmentationParameters | dict[str, Any],
        training_strategy: AugmentationTrainingStrategy | None = None,
    ) -> None:
        """Initialize the neural-network augmentation.

        Args:
            params: Configuration for the underlying encoder. Accepts either
                an ``AutoTCLNeuralNetworkAugmentationParameters`` dataclass or
                a dict with encoder kwargs for backward compatibility.
            training_strategy: Strategy for computing the augmentation loss.
                When ``None``, defaults to ``RIPTrainingStrategy()`` for
                backward compatibility with factory-based instantiation.
        """
        if isinstance(params, dict):
            # Backward-compat shim for dict-based params (factories)
            params = AutoTCLNeuralNetworkAugmentationParameters(**params)  # type: ignore  # noqa: PGH003
        strategy = training_strategy if training_strategy is not None else RIPTrainingStrategy()
        super().__init__(training_strategy=strategy)
        self.params = params
        self._build_model()

    def _build_model(self) -> None:
        """Instantiate the underlying encoder model."""
        self.model = AutoTCLAugmentationTimeSeriesEncoder(**vars(self.params))

    def forward(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the encoder forward pass.

        Args:
            data: Input time-series tensor of shape ``(batch, time, channels)``.

        Returns:
            Dict with ``augmented_data`` and ``augmentation_factor`` tensors.
        """
        return self.model(data)

    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401, ARG002
        """Return an augmented view produced by the encoder model.

        Args:
            data: Input time-series tensor.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            TrainingViews containing the augmented tensor.
        """
        return TrainingViews(views=(self.model.augment(data),), metadata={})

    def get_model(self) -> AutoTCLAugmentationTimeSeriesEncoder:
        """Return the underlying ``AutoTCLAugmentationTimeSeriesEncoder``."""
        return self.model


class CropShiftAugmentation(AugmentationMethod):
    """Random crop-and-shift augmentation used by TS2Vec.

    Produces two overlapping random crops of the input tensor, applying
    independent per-sample temporal offsets.
    """

    def __init__(self, params: CropShiftAugmentationParameters | None = None) -> None:
        """Initialize the crop-and-shift augmentation.

        Args:
            params: Optional configuration controlling the temporal unit.
                When ``None``, defaults to ``CropShiftAugmentationParameters()``.
        """
        self._params = params if params is not None else CropShiftAugmentationParameters()

    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401
        """Return two overlapping random crops of ``data`` with random per-sample shifts.

        A crop window is sampled uniformly, then extended in both directions.
        Each sample in the batch receives an independent random temporal offset,
        producing two overlapping subsequences that share a guaranteed common
        sub-interval of length ``crop_length``.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.
            **kwargs:
                temporal_unit (int): Overrides the configured temporal unit.
                    Controls the minimum crop length as
                    ``2 ** (temporal_unit + 1)``. Defaults to value from
                    ``params`` (or ``0`` when no params provided).

        Returns:
            TrainingViews with two augmented tensors and crop_length metadata.
        """
        # Lazy import to avoid circular dependency:
        # ts2vec/model.py imports from augmentation/strategies.py, so a module-level
        # import of ts2vec/utils.py would create a circular import chain.
        from tscollection.models.cnn.dilated.ts2vec.utils import (  # noqa: PLC0415
            extract_subsequences_per_row,
        )

        temporal_unit = kwargs.get('temporal_unit', self._params.temporal_unit)
        x = data

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
        crop_start = np.random.randint(total_length - crop_length + 1)  # noqa: NPY002
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

        return TrainingViews(
            views=(augmented_subsequences_1, augmented_subsequences_2),
            metadata={'crop_length': crop_length},
        )


class CosTRandomFunctionAugmentation(AugmentationMethod):
    """Stochastic jitter/scale/shift augmentation used by CoST."""

    def __init__(self, params: CosTRandomFunctionAugmentationParameters | dict[str, Any]) -> None:
        """Initialize the random-function augmentation.

        Args:
            params: Configuration controlling noise scale and apply
                probability. Accepts either a
                ``CosTRandomFunctionAugmentationParameters`` dataclass or a
                dict with ``sigma`` (required) and ``p`` (optional, default
                ``0.5``) keys for backward compatibility.
        """
        if isinstance(params, CosTRandomFunctionAugmentationParameters):
            self._params = params
            self._sigma = params.sigma
            self._p = params.p
        else:
            # Backward-compat shim for dict-based params (factories)
            self._params = CosTRandomFunctionAugmentationParameters(
                sigma=params['sigma'], p=params.get('p', 0.5)
            )
            self._sigma = self._params.sigma
            self._p = self._params.p

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with std ``sigma`` with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x + (torch.randn(x.shape, device=x.device) * self._sigma)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by a Gaussian factor around 1 with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x * (torch.randn(x.size(-1), device=x.device) * self._sigma + 1)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Add a per-channel Gaussian offset with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x + (torch.randn(x.size(-1), device=x.device) * self._sigma)

    def augment(self, data: torch.Tensor, **kwargs: Any) -> TrainingViews:  # noqa: ANN401, ARG002
        """Return ``data`` after stochastically applying scale, shift, and jitter.

        Each of the three transforms is applied independently with probability
        ``p``. The composition order is scale → shift → jitter.

        Args:
            data: Input time-series tensor.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            TrainingViews containing the augmented tensor.
        """
        result = self._jitter(self._shift(self._scale(data)))
        return TrainingViews(views=(result,), metadata={})
