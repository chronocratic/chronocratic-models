"""Configuration for the TST (Time Series Transformer) model.

Provides TSTModelParameters with all settings for the transformer
backbone used during masked-reconstruction pretraining.
"""

__all__ = ["TSTModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class TSTModelParameters:
    """Configuration for the TST model.

    Args:
        input_dims: Number of input features (channels) in the time
            series.
        sequence_length: Maximum sequence length supported by the positional
            encoding.
        hidden_dims: Transformer model (token) dimensionality.
        num_heads: Number of attention heads.
        depth: Number of stacked transformer encoder layers.
        feedforward_dims: Hidden dimensionality of the transformer
            feed-forward block.
        dropout_rate: Dropout probability used throughout the transformer.
        pos_encoding: Positional-encoding type (e.g. ``'fixed'`` or
            ``'learnable'``) passed to the encoder.
        activation: Activation function name passed to the transformer
            feed-forward block.
        norm: Normalization layer name (``'BatchNorm'`` or
            ``'LayerNorm'``) used inside the encoder.
        freeze: When ``True``, freezes the backbone weights and only
            trains the output layer.
        learning_rate: Base learning rate for the Adam optimizer.
        lr_step: Milestones (in epochs) for the MultiStepLR scheduler.
            ``None`` means no decay (defaults to a single far-future
            milestone internally).
        lr_factor: Multiplicative decay factor applied at each
            ``lr_step`` milestone.
        weight_decay: L2 regularization coefficient. Applied to the output
            layer only when ``global_reg=False``, or to all parameters
            (via optimizer weight decay) when ``global_reg=True``.
        global_reg: Whether ``weight_decay`` is applied globally as
            weight decay (``True``) or only to the output layer
            (``False``).
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
    """

    input_dims: int
    sequence_length: int
    hidden_dims: int = 64
    num_heads: int = 8
    depth: int = 3
    feedforward_dims: int = 256
    dropout_rate: float = 0.1
    pos_encoding: str = "fixed"
    activation: str = "gelu"
    norm: str = "BatchNorm"
    freeze: bool = False
    learning_rate: float = 1e-3
    lr_step: tuple[int, ...] | None = None
    lr_factor: float = 0.1
    weight_decay: float = 0.0
    global_reg: bool = False
    sync_dist: bool = False
