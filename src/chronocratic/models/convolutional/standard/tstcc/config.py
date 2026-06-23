"""Configuration for the TS-TCC model.

Provides TSTCCModelParameters with all settings for the TCC encoder,
temporal contrast head, and NT-Xent contextual loss for self-supervised
pretraining.
"""

__all__ = ["TSTCCModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class TSTCCModelParameters:
    """Configuration for the TS-TCC model.

    Args:
        input_dims: Number of input features (dimensions) in the time
            series.
        conv_kernel_size: Convolutional kernel size used in the TCC encoder.
        stride: Convolutional stride used in the TCC encoder.
        output_dims: Number of channels produced by the final encoder
            block (also used as the temporal-contrast input dim).
        encoder_channels: Tuple of channel counts for the first two
            encoder convolution blocks.
        encoder_inner_kernels: Tuple of kernel sizes for the inner
            convolution blocks (block 2 and block 3).
        dropout_rate: Dropout probability applied inside the TCC encoder.
        temporal_contrast_hidden_dim: Hidden dimensionality of the temporal
            contrast module.
        temporal_contrast_timesteps: Number of future timesteps the temporal
            contrast module predicts.
        temperature: Temperature scaling for the NT-Xent contextual loss.
        use_cosine_similarity: Whether the NT-Xent loss uses cosine
            similarity (otherwise dot-product).
        learning_rate: Base learning rate for the two Adam optimizers
            (encoder and temporal-contrast).
        temporal_loss_weight: Weight of the temporal-contrast loss term in
            the self-supervised objective.
        contextual_loss_weight: Weight of the contextual NT-Xent loss term
            in the self-supervised objective.
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
    """

    input_dims: int
    conv_kernel_size: int
    stride: int
    output_dims: int = 128
    encoder_channels: tuple[int, ...] = (32, 64)
    encoder_inner_kernels: tuple[int, ...] = (8, 8)
    dropout_rate: float = 0.35
    temporal_contrast_hidden_dim: int = 100
    temporal_contrast_timesteps: int = 6
    temperature: float = 0.2
    use_cosine_similarity: bool = True
    learning_rate: float = 3e-4
    temporal_loss_weight: float = 1.0
    contextual_loss_weight: float = 0.7
    sync_dist: bool = False
