"""Configuration for the TimeVAE model.

Provides TimeVAEModelParameters with all settings for the
variational autoencoder, including the optional trend, seasonal,
and residual decoder branches.
"""

__all__ = ["TimeVAEModelParameters"]

from dataclasses import dataclass


@dataclass(kw_only=True)
class TimeVAEModelParameters:
    """Configuration for the TimeVAE model.

    Args:
        sequence_length: Length of each input time series sample.
        input_dim: Number of input features (channels).
        latent_dim: Dimensionality of the latent space.
        reconstruction_weight: Weight applied to the reconstruction term
            of the VAE loss (the KL term is unweighted).
        learning_rate: Base learning rate for the optimizer.
        hidden_layer_sizes: Output channel sizes of the successive
            Conv1d / ConvTranspose1d blocks in the encoder and
            residual decoder.
        conv_kernel_size: Kernel size for the encoder Conv1d layers.
            Defaults to ``3`` matching the source TimeVAE implementation.
        conv_stride: Stride for the encoder Conv1d layers.
            Defaults to ``2`` matching the source TimeVAE implementation.
        trend_poly: Degree of the polynomial trend basis used by the
            trend decoder branch. ``0`` disables the trend branch.
        custom_seasonality: Optional tuple of ``(num_seasons, len_per_season)``
            tuples describing additive seasonal components. ``None``
            disables the seasonal branch.
        use_residual_conn: Whether to include the residual ConvTranspose
            branch in the decoder.
        max_train_length: Maximum sequence length used during training; longer
            batches are randomly cropped to this length. ``None`` means no
            cap, which will fail on inputs longer than ``sequence_length``.
    """

    sequence_length: int
    input_dim: int
    latent_dim: int = 8
    reconstruction_weight: float = 3.0
    learning_rate: float = 1e-3
    hidden_layer_sizes: tuple[int, ...] = (50, 100, 200)
    conv_kernel_size: int = 3
    conv_stride: int = 2
    trend_poly: int = 0
    custom_seasonality: tuple[tuple[int, int], ...] | None = None
    use_residual_conn: bool = True
    max_train_length: int | None = None
