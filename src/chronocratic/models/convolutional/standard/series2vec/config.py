"""Configuration for the Series2Vec model.

Provides Series2VecModelParameters with all settings for the dual
time/frequency Series2Vec encoder, soft-DTW target computation, and
optimizer choice.
"""

__all__ = ["Series2VecModelParameters"]

from dataclasses import dataclass
from typing import Literal

from chronocratic.models.enums.layers import NormalizationLayerType

OptimizerName = Literal["Adam", "RAdam", "AdamW"]


@dataclass(kw_only=True)
class Series2VecModelParameters:
    """Configuration for the Series2Vec model.

    Args:
        input_dim: Number of input features (channels) in the time
            series.
        embedding_dim: Token embedding dimensionality. Defaults to 16.
        num_heads: Number of attention heads in the transformer encoder.
            Defaults to 8.
        feedforward_dim: Hidden dimensionality of the transformer
            feed-forward block. Defaults to 256.
        representation_dim: Output dimensionality of the projection
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
        normalization_layer_type: Normalization strategy for encoder and
            projection head. ``CHANNEL`` (default) uses GroupNorm for
            batch_size=1 safety. ``BATCH`` uses BatchNorm.
        singleton_split_count: Number of contiguous windows to split a
            singleton batch into for pairwise loss computation. Defaults
            to ``3`` to ensure sufficient gradients at batch_size=1.
    """

    input_dim: int
    embedding_dim: int = 16
    num_heads: int = 8
    feedforward_dim: int = 256
    # migration(a13): renamed from representation_dims — width of the vector encode() returns. See CHANGELOG v0.1.0a13.
    representation_dim: int = 320
    dropout_rate: float = 0.01
    encoder_kernel_size: int = 8
    learning_rate: float = 1e-3
    soft_dtw_gamma: float = 0.1
    sync_dist: bool = False
    optimizer_name: OptimizerName = "RAdam"
    weight_decay: float = 0.0
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL
    singleton_split_count: int = 3
