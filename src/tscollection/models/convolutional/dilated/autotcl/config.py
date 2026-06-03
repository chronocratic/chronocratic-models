"""Configuration for the AutoTCL model.

Provides AutoTCLModelParameters with AutoTCL-specific settings:
DWT kernel sizes for multi-scale feature extraction, mask mode,
learning rate, training length cap, and distributed-sync flag.
"""

__all__ = ['AutoTCLModelParameters']

from dataclasses import dataclass, field

from tscollection.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass(kw_only=True)
class AutoTCLModelParameters:
    """Configuration for the AutoTCL model.

    Args:
        input_dims: Number of input features (channels) in the time series.
        kernel_sizes: DWT decomposition levels as kernel sizes.
        hidden_dims: Number of hidden units in each encoder layer.
        output_dims: Number of output features produced by the encoder.
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel in each layer.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        max_train_length: Maximum sequence length; longer samples are
            truncated. ``None`` means no limit.
        meta_learning_rate: Learning rate for the augmentation network
            optimizer.
        local_loss_weight: Weight for the local InfoNCE loss term in the
            encoder contrastive loss.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    input_dims: int
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    meta_learning_rate: float = 1e-2
    local_loss_weight: float = 0.1
    sync_dist: bool = False
