"""Configuration for the CoST model.

Provides CoSTModelParameters with CoST-specific settings including
seasonal-trend decomposition encoder parameters and contrastive
learning configuration.
"""

__all__ = ["CoSTModelParameters"]

from dataclasses import dataclass

from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode


@dataclass(kw_only=True)
class CoSTModelParameters:
    """Configuration for the CoST model.

    Args:
        input_dim: Number of input features (channels) in the time series.
        sequence_length: Length of each input time series sample.
        kernel_sizes: DWT decomposition levels as kernel sizes.
        max_train_length: Maximum sequence length for training samples.
        hidden_dim: Number of hidden units in each encoder layer.
        representation_dim: Number of output features produced by the encoder
            (the size of the encode() representation).
        depth: Number of encoder layers.
        dropout_rate: Dropout probability applied after each encoder layer.
        conv_kernel_size: Convolutional kernel size in the dilated encoder.
            Defaults to ``3`` matching the source CoST implementation.
        mask_mode: Strategy for masking input tokens during training.
        learning_rate: Base learning rate for the optimizer.
        seasonal_loss_weight: Weight for the seasonal contrastive loss term.
        queue_size: Size of the memory queue for contrastive learning.
            The source repo's ``CoSTModel`` class defines ``K=65536`` as its
            constructor default, but the training wrapper always passes
            ``K=256`` explicitly. We use ``256`` to match the authors' actual
            usage rather than the unused class default.
        momentum: Momentum coefficient for the key encoder update.
        temperature: Temperature scaling for the contrastive loss.
        sync_dist: Whether to synchronize metrics across distributed
            processes.
    """

    input_dim: int
    sequence_length: int
    kernel_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    max_train_length: int = 201
    hidden_dim: int = 64
    representation_dim: int = 320  # renamed from `output_dims` in the original codebase
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    seasonal_loss_weight: float = 0.0005
    queue_size: int = 256
    momentum: float = 0.999
    temperature: float = 0.07
    sync_dist: bool = False
