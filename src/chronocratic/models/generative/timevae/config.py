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
        sequence_length: Length of each input time-series sample.
        input_dims: Number of input features (channels).
        latent_dim: Dimensionality of the latent space.
        reconstruction_weight: Weight applied to the reconstruction term
            of the VAE loss (the KL term is unweighted).
        learning_rate: Base learning rate for the optimizer.
        hidden_layer_sizes: Output channel sizes of the successive
            Conv1d / ConvTranspose1d blocks in the encoder and
            residual decoder.
        trend_poly: Degree of the polynomial trend basis used by the
            trend decoder branch. ``0`` disables the trend branch.
        custom_seasonality: Optional tuple of ``(num_seasons, len_per_season)``
            tuples describing additive seasonal components. ``None``
            disables the seasonal branch.
        use_residual_conn: Whether to include the residual ConvTranspose
            branch in the decoder.
    """

    sequence_length: int
    input_dims: int
    latent_dim: int = 8
    reconstruction_weight: float = 3.0
    learning_rate: float = 1e-3
    hidden_layer_sizes: tuple[int, ...] = (50, 100, 200)
    trend_poly: int = 0
    custom_seasonality: tuple[tuple[int, int], ...] | None = None
    use_residual_conn: bool = True
