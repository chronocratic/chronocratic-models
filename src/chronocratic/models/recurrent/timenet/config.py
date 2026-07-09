"""Configuration for the TimeNet model.

Provides TimeNetModelParameters with settings for the GRU encoder /
decoder pair used by the autoencoder pretraining objective.
"""

__all__ = ["TimeNetModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class TimeNetModelParameters:
    """Configuration for the TimeNet model.

    Args:
        hidden_dim: Number of hidden units in each GRU layer of the
            encoder and decoder.
        depth: Number of stacked GRU layers in each of the encoder
            and decoder.
        input_dim: Number of input features (channels) in the time series.
        dropout_rate: Dropout probability inserted between successive GRU
            layers. ``0`` disables dropout.
        learning_rate: Base learning rate for the Adam optimizer.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    input_dim: int
    hidden_dim: int = 64
    depth: int = 3
    dropout_rate: float = 0.4
    learning_rate: float = 5e-3
    sync_dist: bool = False
