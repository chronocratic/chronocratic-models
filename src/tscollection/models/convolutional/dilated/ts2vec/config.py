"""Configuration for the TS2Vec model.

Extends DilatedCNNModelParameters with TS2Vec-specific runtime
settings: mask mode, learning rate, training length cap, temporal
unit, and distributed-sync flag.
"""

__all__ = ['TS2VecModelParameters']

from dataclasses import dataclass

from tscollection.models.convolutional.dilated.config import DilatedCNNModelParameters
from tscollection.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass
class TS2VecModelParameters(DilatedCNNModelParameters):
    """Configuration for the TS2Vec model.

    Extends :class:`DilatedCNNModelParameters` with TS2Vec-specific
    runtime settings: mask mode, learning rate, training length cap,
    temporal unit, and distributed-sync flag.

    Args:
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        max_train_length: Maximum sequence length; longer samples are
            truncated. ``None`` means no limit.
        temporal_unit: Token-level temporal unit index.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    temporal_unit: int = 0
    sync_dist: bool = False
