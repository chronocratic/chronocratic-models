"""Typed configuration dataclasses for time-series model parameters.

Provides dataclasses that map 1:1 to model __init__ signatures,
excluding augmentation fields (deferred to Phase 3 per D-01)
and runner artifacts (stripped per D-02).

Hierarchy:
    ModelParameters (ABC)
    └── DilatedCNNModelParameters
        ├── TS2VecModelParameters
        └── AutoTCLModelParameters
    └── CoSTModelParameters (inherits directly from ModelParameters per D-03)
"""

__all__ = [
    'AutoTCLModelParameters',
    'CoSTModelParameters',
    'DilatedCNNModelParameters',
    'ModelParameters',
    'TS2VecModelParameters',
]

import abc
from dataclasses import dataclass, field

from tscollection.models.encoders.masking import MaskMode


@dataclass
class ModelParameters(abc.ABC):
    """Abstract base class for all model parameter configurations.

    Provides no fields itself; serves as a common type for IDE
    autocompletion and static type checking across all model configs.

    Raises:
        TypeError: If instantiated directly. Subclasses must be used.
    """

    def __new__(cls: type, **kwargs: object) -> "ModelParameters":
        if cls is ModelParameters:
            msg = 'ModelParameters is abstract and cannot be instantiated directly'
            raise TypeError(msg)
        return super().__new__(cls)


@dataclass
class DilatedCNNModelParameters(ModelParameters):
    """Shared parameters for dilated CNN-based models.

    Captures the common encoding pipeline used by TS2Vec and AutoTCL:
    a stack of dilated Conv1d layers with skip connections.

    Args:
        input_dims: Number of input features (channels) in the time series.
        hidden_dims: Number of hidden units in each encoder layer.
        output_dims: Number of output features produced by the encoder.
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        conv_kernel_size: Size of the convolutional kernel in each layer.
    """

    input_dims: int
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3


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


@dataclass
class CoSTModelParameters(ModelParameters):
    """Configuration for the CoST model.

    Inherits directly from :class:`ModelParameters` (not
    ``DilatedCNNModelParameters``) because CoST uses a seasonal-trend
    decomposition encoder with a distinct parameter surface.

    Args:
        input_dims: Number of input features (channels) in the time series.
        sequence_length: Length of each input time series sample.
        kernel_sizes: DWT decomposition levels as kernel sizes. Empty list
            means the encoder selects levels automatically.
        max_train_length: Maximum sequence length for training samples.
        hidden_dims: Number of hidden units in each encoder layer.
        output_dims: Number of output features produced by the encoder.
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        seasonal_loss_weight: Weight for the seasonal contrastive loss term.
        queue_size: Size of the memory queue for contrastive learning.
        momentum: Momentum coefficient for the key encoder update.
        temperature: Temperature scaling for the contrastive loss.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    input_dims: int
    sequence_length: int
    kernel_sizes: list[int] = field(default_factory=list)
    max_train_length: int = 201
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    seasonal_loss_weight: float = 0.1
    queue_size: int = 65536
    momentum: float = 0.999
    temperature: float = 0.07
    sync_dist: bool = False


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
