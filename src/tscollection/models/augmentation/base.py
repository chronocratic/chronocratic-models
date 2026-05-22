"""Abstract base classes for augmentation strategies.

This module defines the shared augmentation hierarchy extracted from the
monolithic ``strategies.py``. It contains only abstract base classes and the
``TrainingViews`` dataclass (~200 lines). Concrete implementations live in
per-model augmentation files.

Exported symbols:
    - ``AugmentationMethod``: Abstract transform interface.
    - ``AugmentationTrainingStrategy``: Abstract training-loss interface.
    - ``TrainableAugmentation``: Abstract trainable augmentation (nn.Module).
    - ``TrainingViews``: Dataclass for augmentation output views + metadata.
"""

from __future__ import annotations

__all__ = [
    'AugmentationMethod',
    'AugmentationTrainingStrategy',
    'TrainableAugmentation',
    'TrainingViews',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

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

    def __init__(self, training_ratio_step: int = 1) -> None:
        """Initialize the training strategy.

        Args:
            training_ratio_step: Train the aug network every N epochs.
                Default ``1`` means every epoch. Matches the original
                ``augmentation_network_training_ratio_step`` from
                ``augmentation_mode_params``.
        """
        self._training_ratio_step = training_ratio_step

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

        Default: train when ``epoch % training_ratio_step == 0``.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return epoch % self._training_ratio_step == 0


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

    def should_train_augmentation(self, epoch: int, batch_idx: int) -> bool:
        """Check whether the aug-network should train this step.

        Delegates to the composed training strategy to avoid exposing
        the private ``_training_strategy`` attribute.

        Args:
            epoch: Current training epoch.
            batch_idx: Current batch index within the epoch.

        Returns:
            ``True`` if the augmentation network should be trained this step.
        """
        return self._training_strategy.should_train(epoch, batch_idx)

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
