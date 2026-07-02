"""Configuration for the MCL (MixUp Contrastive Learning) model.

Provides MCLModelParameters with MCL-specific settings for the FCN
encoder and MixUp contrastive criterion.
"""

__all__ = ["MCLModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class MCLModelParameters:
    """Configuration for the MCL model.

    Args:
        input_dims: Number of input features (channels) in the time series.
        output_dims: Number of output features produced by the encoder.
        alpha: Beta-distribution parameter controlling the MixUp
            interpolation coefficient.
        learning_rate: Base learning rate for the Adam optimizer.
        encoder_channels: Tuple of channel counts for each Conv1d block
            in the FCN encoder.
        encoder_kernels: Tuple of kernel sizes for each Conv1d block
            in the FCN encoder.
        encoder_dilations: Tuple of dilation rates for each Conv1d block
            in the FCN encoder.
        projection_dims: Hidden dimension of the projection head used
            for contrastive learning.
        sync_dist: Whether to synchronize metrics across distributed
            processes during logging.
        norm: Normalization strategy for encoder and projection head.
            ``"layer"`` (default) uses GroupNorm for batch_size=1 safety.
            ``"batch"`` uses BatchNorm1d (original behavior).
    """

    input_dims: int
    output_dims: int = 128
    alpha: float = 1.0
    learning_rate: float = 1e-3
    encoder_channels: tuple[int, ...] = (128, 256, 128)
    encoder_kernels: tuple[int, ...] = (7, 5, 3)
    encoder_dilations: tuple[int, ...] = (2, 4, 8)
    projection_dims: int = 128
    sync_dist: bool = False
    norm: str = "layer"
