"""Configuration for the MCL (MixUp Contrastive Learning) model.

Provides MCLModelParameters with MCL-specific settings for the FCN
encoder and MixUp contrastive criterion.
"""

__all__ = ["MCLModelParameters"]

from dataclasses import dataclass

from chronocratic.models.enums.layers import NormalizationLayerType


@dataclass(kw_only=True)
class MCLModelParameters:
    """Configuration for the MCL model.

    Args:
        input_dim: Number of input features (channels) in the time series.
        representation_dim: Width of the vector encode() returns.
            The representation size of the encoder output.
        alpha: Beta-distribution parameter controlling the MixUp
            interpolation coefficient.
        learning_rate: Base learning rate for the Adam optimizer.
        encoder_channels: Tuple of channel counts for each Conv1d block
            in the FCN encoder.
        encoder_kernels: Tuple of kernel sizes for each Conv1d block
            in the FCN encoder.
        encoder_dilations: Tuple of dilation rates for each Conv1d block
            in the FCN encoder.
        projection_dim: Hidden dimension of the projection head used
            for contrastive learning.
        sync_dist: Whether to synchronize metrics across distributed
            processes during logging.
        normalization_layer_type: Normalization strategy for encoder and
            projection head. ``CHANNEL`` (default) uses GroupNorm for
            batch_size=1 safety. ``BATCH`` uses BatchNorm1d.
    """

    input_dim: int
    representation_dim: int = 128  # migration(a13): renamed from `output_dims` - width of the vector encode() returns. See CHANGELOG v0.1.0a13.
    alpha: float = 1.0
    learning_rate: float = 1e-3
    encoder_channels: tuple[int, ...] = (128, 256, 128)
    encoder_kernels: tuple[int, ...] = (7, 5, 3)
    encoder_dilations: tuple[int, ...] = (2, 4, 8)
    projection_dim: int = 128
    sync_dist: bool = False
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL
