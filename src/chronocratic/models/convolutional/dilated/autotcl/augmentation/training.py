"""AutoTCL augmentation training strategies.

Contains ``RIPTrainingStrategy`` and ``AdversarialTrainingStrategy``, moved
from the shared ``augmentation/strategies.py`` for per-model self-containment.

Imports ``AugmentationTrainingStrategy`` directly from ``augmentation/base.py``
(NOT the barrel) to avoid circular dependencies.
"""

__all__ = ["AdversarialTrainingStrategy", "RIPTrainingStrategy"]

import torch
from torch.nn import functional as F  # noqa: N812

from chronocratic.models.augmentation.base import AugmentationTrainingStrategy
from chronocratic.models.convolutional.dilated.autotcl.losses import (
    info_nce_loss,
    maximum_mean_discrepancy_with_gaussian_kernel_loss,
)


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
        training_ratio_step: int = 1,
    ) -> None:
        """Initialize the RIP training strategy.

        Args:
            consistency_weight: Weight for the regular consistency term.
            regularization_weight: Weight for the regularization loss.
            regularization_threshold: Threshold for the regularization term.
            training_ratio_step: Train the aug network every N epochs.
                Default ``1`` means every epoch.
        """
        super().__init__(training_ratio_step=training_ratio_step)
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
        # which imports from strategies.py -- utils.py also imports from the barrel.
        from chronocratic.models.convolutional.dilated.autotcl.utils import (  # noqa: PLC0415
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


class AdversarialTrainingStrategy(AugmentationTrainingStrategy):
    """Adversarial training strategy.

    Maximizes the InfoNCE loss between original and augmented embeddings,
    encouraging the augmentation network to produce hard-to-distinguish
    augmented views.

    Extracted from AutoTCL's
    ``_augmentation_loss_neural_network_augmentation_adversarial``.
    """

    def __init__(self, training_ratio_step: int = 1) -> None:
        """Initialize the adversarial training strategy.

        Args:
            training_ratio_step: Train the aug network every N epochs.
                Default ``1`` means every epoch.
        """
        super().__init__(training_ratio_step=training_ratio_step)

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
