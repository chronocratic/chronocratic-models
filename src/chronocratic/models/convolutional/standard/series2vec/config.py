"""Configuration for the Series2Vec model.

Provides Series2VecModelParameters with all settings for the dual
time/frequency Series2Vec encoder, soft-DTW target computation, and
optimizer choice.
"""

__all__ = ["Series2VecModelParameters"]

from dataclasses import dataclass
from typing import Literal

OptimizerName = Literal["Adam", "RAdam", "AdamW"]


@dataclass(kw_only=True)
class Series2VecModelParameters:
    """Configuration for the Series2Vec model.

    Args:
        input_dims: Number of input features (channels) in the time
            series.
        embedding_dims: Token embedding dimensionality.
        num_heads: Number of attention heads in the transformer encoder.
        feedforward_dims: Hidden dimensionality of the transformer
            feed-forward block.
        representation_dims: Output dimensionality of the projection
            head used for pretraining.
        dropout_rate: Dropout probability applied throughout the
            network.
        encoder_kernel_size: Kernel size of the convolutional tokenizer.
        learning_rate: Base learning rate for the optimizer.
        soft_dtw_gamma: Smoothing parameter for the soft-DTW distance
            used as the temporal target.
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
        optimizer_name: Optimizer to use; one of ``'Adam'``, ``'RAdam'``,
            or ``'AdamW'``.
        weight_decay: L2 weight-decay coefficient passed to the
            optimizer.
    """

    input_dims: int
    embedding_dims: int
    num_heads: int
    feedforward_dims: int
    representation_dims: int
    dropout_rate: float
    encoder_kernel_size: int = 8
    learning_rate: float = 1e-3
    soft_dtw_gamma: float = 0.1
    sync_dist: bool = False
    optimizer_name: OptimizerName = "RAdam"
    weight_decay: float = 0.0
