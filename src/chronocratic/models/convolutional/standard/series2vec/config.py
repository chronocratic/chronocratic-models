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
        embedding_dims: Token embedding dimensionality. Defaults to 16.
        num_heads: Number of attention heads in the transformer encoder.
            Defaults to 8.
        feedforward_dims: Hidden dimensionality of the transformer
            feed-forward block. Defaults to 256.
        representation_dims: Output dimensionality of the projection
            head used for pretraining. Defaults to 320.
        dropout_rate: Dropout probability applied throughout the
            network. Defaults to 0.01.
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
        norm: Normalization strategy for encoder and projection head.
            ``"layer"`` (default) uses GroupNorm for batch_size=1 safety.
            ``"batch"`` uses BatchNorm (original behavior).
        singleton_split_count: Number of contiguous windows to split a
            singleton batch into for pairwise loss computation. Defaults
            to ``3`` to ensure sufficient gradients at batch_size=1.
    """

    input_dims: int
    embedding_dims: int = 16
    num_heads: int = 8
    feedforward_dims: int = 256
    representation_dims: int = 320
    dropout_rate: float = 0.01
    encoder_kernel_size: int = 8
    learning_rate: float = 1e-3
    soft_dtw_gamma: float = 0.1
    sync_dist: bool = False
    optimizer_name: OptimizerName = "RAdam"
    weight_decay: float = 0.0
    norm: str = "layer"
    singleton_split_count: int = 3
