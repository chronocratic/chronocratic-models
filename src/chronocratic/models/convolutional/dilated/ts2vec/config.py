"""Configuration for the TS2Vec model.

Provides TS2VecModelParameters with all TS2Vec-specific runtime
settings: mask mode, learning rate, training length cap, temporal
unit, and distributed-sync flag.
"""

__all__ = ['TS2VecModelParameters']

from dataclasses import dataclass

from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass(kw_only=True)
class TS2VecModelParameters:
    """Configuration for the TS2Vec model.

    Args:
        input_dims: Number of input features (channels) in the time series.
        hidden_dims: Number of hidden units in each encoder layer.
        output_dims: Number of output features produced by the encoder.
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel in each layer.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        max_train_length: Maximum sequence length; longer samples are
            truncated. ``None`` means no limit.
        temporal_unit: Token-level temporal unit index.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    input_dims: int
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    temporal_unit: int = 0
    sync_dist: bool = False
