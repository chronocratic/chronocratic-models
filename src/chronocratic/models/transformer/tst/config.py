"""Configuration for the TST (Time Series Transformer) model.

Provides TSTModelParameters with all settings for the transformer
backbone used during masked-reconstruction pretraining.
"""

__all__ = ['TSTModelParameters']

from dataclasses import dataclass


@dataclass(kw_only=True)
class TSTModelParameters:
    """Configuration for the TST model.

    Args:
        feat_dim: Number of input features (channels) in the time
            series.
        max_seq_len: Maximum sequence length supported by the positional
            encoding.
        d_model: Transformer model (token) dimensionality.
        n_heads: Number of attention heads.
        num_layers: Number of stacked transformer encoder layers.
        dim_feedforward: Hidden dimensionality of the transformer
            feed-forward block.
        dropout: Dropout probability used throughout the transformer.
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
        l2_reg: L2 regularization coefficient. Applied to the output
            layer only when ``global_reg=False``, or to all parameters
            (via optimizer weight decay) when ``global_reg=True``.
        global_reg: Whether ``l2_reg`` is applied globally as
            weight decay (``True``) or only to the output layer
            (``False``).
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
    """

    feat_dim: int
    max_seq_len: int
    d_model: int = 64
    n_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    pos_encoding: str = 'fixed'
    activation: str = 'gelu'
    norm: str = 'BatchNorm'
    freeze: bool = False
    learning_rate: float = 1e-3
    lr_step: list[int] | None = None
    lr_factor: float = 0.1
    l2_reg: float = 0.0
    global_reg: bool = False
    sync_dist: bool = False
