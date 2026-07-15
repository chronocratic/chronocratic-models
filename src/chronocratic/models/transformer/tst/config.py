"""Configuration for the TST (Time Series Transformer) model.

Provides TSTModelParameters with all settings for the transformer
backbone used during masked-reconstruction pretraining.
"""

__all__ = ["TSTModelParameters"]

from dataclasses import dataclass

from chronocratic.models.enums.layers import NormalizationLayerType


@dataclass(kw_only=True)
class TSTModelParameters:
    """Configuration for the TST model.

    Args:
        input_dim: Number of input features (channels) in the time
            series.
        sequence_length: Maximum sequence length supported by the positional
            encoding.
        hidden_dim: Transformer model (token) dimensionality.
        num_heads: Number of attention heads.
        depth: Number of stacked transformer encoder layers.
        feedforward_dim: Hidden dimensionality of the transformer
            feed-forward block.
        dropout_rate: Dropout probability used throughout the transformer.
        masking_ratio: Fraction of input elements zeroed during
            masked-reconstruction pretraining. Each element is masked
            independently (Bernoulli). ``0.15`` matches the upstream default.
        pos_encoding: Positional-encoding type (e.g. ``'fixed'`` or
            ``'learnable'``) passed to the encoder.
        activation: Activation function name passed to the transformer
            feed-forward block.
        normalization_layer_type: Normalization layer used inside the
            encoder. ``BATCH`` (default) uses custom BatchNorm transformer
            layers. ``CHANNEL`` uses PyTorch's LayerNorm-based
            TransformerEncoderLayer.
        freeze: When ``True``, freezes the backbone weights and only
            trains the output layer.
        learning_rate: Base learning rate for the Adam optimizer.
        lr_step: Milestones (in epochs) for the MultiStepLR scheduler.
            ``None`` means no decay (defaults to a single far-future
            milestone internally).
        lr_factor: Multiplicative decay factor applied at each
            ``lr_step`` milestone.
        weight_decay: L2 regularization coefficient. Inactive at the
            default ``0.0``. When positive, applied to all parameters via
            optimizer weight decay if ``global_reg=True``, or added to the
            training loss as an L2 penalty on the output layer alone if
            ``global_reg=False``.
        global_reg: Selects where a positive ``weight_decay`` is applied:
            globally via the optimizer (``True``) or to the output layer
            only (``False``). No effect when ``weight_decay=0.0``.
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
    """

    input_dim: int
    sequence_length: int
    hidden_dim: int = 64
    num_heads: int = 8
    depth: int = 3
    feedforward_dim: int = 256
    dropout_rate: float = 0.1
    masking_ratio: float = 0.15
    pos_encoding: str = "fixed"
    activation: str = "gelu"
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.BATCH
    freeze: bool = False
    learning_rate: float = 1e-3
    lr_step: tuple[int, ...] | None = None
    lr_factor: float = 0.1
    weight_decay: float = 0.0
    global_reg: bool = False
    sync_dist: bool = False
