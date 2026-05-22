"""AutoTCL augmentation methods.

Contains ``AutoTCLNeuralNetworkAugmentation`` and its
``AutoTCLNeuralNetworkAugmentationParameters`` dataclass, moved from the
shared ``augmentation/strategies.py`` and ``augmentation/config.py`` for
per-model self-containment.

Imports ``TrainableAugmentation``, ``TrainingViews``, and
``AugmentationTrainingStrategy`` directly from ``augmentation/base.py`` (NOT
the barrel) to avoid circular dependencies.
"""

__all__ = ['AutoTCLNeuralNetworkAugmentation', 'AutoTCLNeuralNetworkAugmentationParameters']

from dataclasses import dataclass, field
from typing import Any

import torch

from tscollection.models.augmentation.base import (
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)
from tscollection.models.convolutional.dilated.autotcl.augmentation.training import (
    RIPTrainingStrategy,
)
from tscollection.models.convolutional.dilated.encoders.encoders import (
    AutoTCLAugmentationTimeSeriesEncoder,
)
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass
class AutoTCLNeuralNetworkAugmentationParameters:
    """Parameters for :class:`AutoTCLNeuralNetworkAugmentation`.

    Mirrors the constructor signature of
    :class:`AutoTCLAugmentationTimeSeriesEncoder`, allowing ``vars()``
    unpacking to instantiate the encoder directly.

    Args:
        input_dims: Number of input features (channels).
        output_dims: Number of output features produced by the encoder.
        kernel_sizes: DWT decomposition levels as kernel sizes. Empty
            list means the encoder selects levels automatically.
        hidden_dims: Number of hidden units in each encoder layer.
        feature_extractor_depth: Number of encoder layers.
        dropout_rate: Dropout probability after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel.
        mask_mode: Strategy for masking input tokens.
        num_augmentation_channels: Number of augmentation output channels.
        gumbel_bias: Bias term for the Gumbel-sigmoid mask sampling.
        zeta: Scaling factor for the Gumbel temperature.
        gamma_zeta: Weight for the zeta regularization term.
        hard_mask: Whether to use a hard (binary) mask at inference time.
    """

    input_dims: int
    output_dims: int
    kernel_sizes: list[int] = field(default_factory=list)
    hidden_dims: int = 64
    feature_extractor_depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    num_augmentation_channels: int = 1
    gumbel_bias: float = 0.001
    zeta: float = 1.0
    gamma_zeta: float = 0.05
    hard_mask: bool = True


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

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
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
