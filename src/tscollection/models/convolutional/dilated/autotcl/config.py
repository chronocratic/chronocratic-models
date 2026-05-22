"""Configuration for the AutoTCL model.

Extends DilatedCNNModelParameters with AutoTCL-specific settings:
DWT kernel sizes for multi-scale feature extraction, mask mode,
learning rate, training length cap, and distributed-sync flag.
"""

__all__ = ['AutoTCLModelParameters']

from dataclasses import dataclass, field

from tscollection.models.convolutional.dilated.config import DilatedCNNModelParameters
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass
class AutoTCLModelParameters(DilatedCNNModelParameters):
    """Configuration for the AutoTCL model.

    Extends :class:`DilatedCNNModelParameters` with AutoTCL-specific
    settings: DWT kernel sizes for multi-scale feature extraction,
    mask mode, learning rate, training length cap, and distributed-sync
    flag.

    Args:
        kernel_sizes: DWT decomposition levels as kernel sizes. Empty list
            means the encoder selects levels automatically.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        max_train_length: Maximum sequence length; longer samples are
            truncated. ``None`` means no limit.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    kernel_sizes: list[int] = field(default_factory=list)
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    sync_dist: bool = False
