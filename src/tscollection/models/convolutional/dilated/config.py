"""Configuration for dilated CNN-based time-series models.

Provides DilatedCNNModelParameters as the shared base class for
models that use a stack of dilated Conv1d layers with skip
connections (TS2Vec, AutoTCL).
"""

__all__ = ['DilatedCNNModelParameters']

from dataclasses import dataclass

from tscollection.models.config import ModelParameters


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
