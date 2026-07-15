import warnings

import torch
from torch import nn

__all__ = ["TimeVAE", "TimeVAEDecoder", "TimeVAEEncoder", "_timevae_encoder_output_length"]

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.generative.timevae.vae_base import BaseVariationalAutoencoder, Sampling
from chronocratic.models.layers.general import (
    LevelModel,
    ResidualConnection,
    Seasonality,
    SeasonalLayer,
    TrendLayer,
)
from chronocratic.models.utils.helpers import _warn_sequence_fallback

# Minimum encoder output length for valid latent encoding.
_MIN_VALID_ENCODER_OUTPUT = 2


def _timevae_encoder_output_length(seq_len: int, num_layers: int, stride: int) -> int:
    """Compute the encoder spatial output after N Conv1d(k=3, s=stride, pad=1) layers.

    Conv1d output formula: L_out = (L_in - 1) // stride + 1
    Applied N times (one per encoder conv layer).

    Args:
        seq_len: Input sequence length.
        num_layers: Number of Conv1d layers in the encoder.
        stride: Stride of each Conv1d layer.

    Returns:
        Output spatial dimension after all encoder conv layers.
    """
    out_len = seq_len
    for _ in range(num_layers):
        out_len = (out_len - 1) // stride + 1
    return out_len


class TimeVAEEncoder(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        latent_dim: int,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers: nn.ModuleList = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(
                input_dim,
                hidden_layer_sizes[0],
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=1,
            )
        )
        self.layers.append(nn.ReLU())

        for i, num_filters in enumerate(hidden_layer_sizes[1:]):
            self.layers.append(
                nn.Conv1d(
                    hidden_layer_sizes[i],
                    num_filters,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=1,
                )
            )
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten())

        self.encoder_last_dense_dim = self._get_last_dense_dim(sequence_length, input_dim)

        self.encoder = nn.Sequential(*self.layers)
        del self.layers
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_log_var = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.sampling = Sampling()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an input batch into latent mean, log-variance, and sample."""
        x = x.transpose(1, 2)
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def _get_last_dense_dim(self, sequence_length: int, input_dim: int) -> int:
        with torch.no_grad():
            # device-ok: CPU test input for shape probing
            x = torch.randn(1, input_dim, sequence_length)
            for conv in self.layers:
                x = conv(x)
            return x.numel()


class TimeVAEDecoder(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        latent_dim: int,
        trend_poly: int = 0,
        custom_seasonality: tuple[Seasonality, ...] | None = None,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        use_residual_conn: bool = True,
        encoder_last_dense_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.custom_seasonality = custom_seasonality
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        if self.trend_poly > 0:
            self.trend_layer = TrendLayer(
                sequence_length=self.sequence_length,
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                trend_poly=self.trend_poly,
            )
        if self.custom_seasonality is not None and len(self.custom_seasonality) > 0:
            self.seasonal_layer = SeasonalLayer(
                sequence_length=self.sequence_length,
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                custom_seasonality=self.custom_seasonality,
            )
        self.use_residual_conn = use_residual_conn
        self.encoder_last_dense_dim = encoder_last_dense_dim
        self.level_model = LevelModel(
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
        )

        if use_residual_conn:
            if encoder_last_dense_dim is None:
                msg = "encoder_last_dense_dim is required when use_residual_conn is True."
                raise ValueError(msg)
            self.residual_conn = ResidualConnection(
                sequence_length=self.sequence_length,
                input_dim=self.input_dim,
                hidden_layer_sizes=hidden_layer_sizes,
                latent_dim=latent_dim,
                encoder_last_dense_dim=encoder_last_dense_dim,
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples into reconstructed time series batches."""
        outputs = self.level_model(z)
        if self.trend_poly > 0:
            outputs += self.trend_layer(z)

        # custom seasons
        if self.custom_seasonality is not None and len(self.custom_seasonality) > 0:
            outputs += self.seasonal_layer(z)

        if self.use_residual_conn:
            residuals = self.residual_conn(z)
            outputs += residuals

        return outputs


class TimeVAE(BaseVariationalAutoencoder, BasicEncodingMixin):
    """TimeVAE model.

    Variational autoencoder with interpretable trend and seasonality
    decomposition in the decoder. The encoder uses 1-D convolutions
    followed by a latent Gaussian representation.

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
        conv_stride: Stride for the encoder Conv1d layers.
        trend_poly: Degree of the polynomial trend basis used by the
            trend decoder branch. ``0`` disables the trend branch.
        custom_seasonality: Optional tuple of ``(num_seasons, len_per_season)``
            tuples describing additive seasonal components. ``None``
            disables the seasonal branch.
        use_residual_conn: Whether to include the residual ConvTranspose
            branch in the decoder.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/abudesai/timeVAE under MIT License.
    """

    model_name = "TimeVAE"
    _encoder: TimeVAEEncoder
    _decoder: TimeVAEDecoder

    @property
    def encoder(self) -> TimeVAEEncoder:
        """Return the TimeVAE encoder."""
        return self._encoder

    @property
    def decoder(self) -> TimeVAEDecoder:
        """Return the TimeVAE decoder."""
        return self._decoder

    @property
    def representation_dim(self) -> int:
        """Feature dim of the ``encode()`` output.

        For TimeVAE this equals ``latent_dim``, the dimensionality of
        the stochastic bottleneck.
        """
        return self.latent_dim

    def __init__(
        self,
        *,
        sequence_length: int,
        input_dim: int,
        latent_dim: int = 8,
        reconstruction_weight: float = 3.0,
        learning_rate: float = 1e-3,
        hidden_layer_sizes: tuple[int, ...] | None = None,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        trend_poly: int = 0,
        custom_seasonality: tuple[tuple[int, int], ...] | None = None,
        use_residual_conn: bool = True,
    ) -> None:
        super().__init__(
            sequence_length=sequence_length,
            input_dim=input_dim,
            latent_dim=latent_dim,
            reconstruction_weight=reconstruction_weight,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = (50, 100, 200)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.trend_poly = trend_poly
        self.custom_seasonality = custom_seasonality
        self.use_residual_conn = use_residual_conn

        # Auto-clamp conv_stride when encoder output spatial dim < 2 (D-10)
        num_encoder_layers = len(self.hidden_layer_sizes)
        encoder_out = _timevae_encoder_output_length(
            self.sequence_length, num_encoder_layers, self.conv_stride
        )
        if encoder_out < _MIN_VALID_ENCODER_OUTPUT:
            # Try reducing stride to 1
            encoder_out_stride1 = _timevae_encoder_output_length(
                self.sequence_length, num_encoder_layers, 1
            )
            if encoder_out_stride1 < _MIN_VALID_ENCODER_OUTPUT:
                msg = (
                    f"TimeVAE: sequence_length={self.sequence_length} with "
                    f"{num_encoder_layers} encoder layers produces encoder output "
                    f"{encoder_out_stride1} < 2 even with stride=1. "
                    f"Cannot prevent degenerate latent encoding."
                )
                raise ValueError(msg)
            original_stride = self.conv_stride
            self.conv_stride = 1
            warnings.warn(
                f"TimeVAE: encoder output ({encoder_out}) < 2 with conv_stride={original_stride}. "
                f"Auto-clamped conv_stride to 1 (output={encoder_out_stride1}).",
                UserWarning,
                stacklevel=2,
            )

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _build_encoder(self) -> TimeVAEEncoder:
        return TimeVAEEncoder(
            sequence_length=self.sequence_length,
            input_dim=self.input_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            latent_dim=self.latent_dim,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
        )

    def _get_encoder(self) -> nn.Module:
        """Expose the VAE encoder for ``BasicEncodingMixin.encode``."""
        return self._encoder

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Return the latent mean ``z_mean`` from the encoder output tuple.

        Args:
            encoder: The TimeVAEEncoder module.
            batch_x: Batch tensor of shape ``(B, seq_len, input_dim)``.
            output: Requested output shape. Defaults to VECTOR (2-D).

        Returns:
            Representations of shape ``(B, D)`` for VECTOR or
            ``(B, 1, D)`` for SEQUENCE (B=batch, D=latent_dim).
        """
        z_mean = encoder(batch_x)[0]  # (B, D) - D=latent_dim
        if output == EncodingOutputShape.VECTOR:
            return z_mean  # (B, D) — VECTOR
        if output == EncodingOutputShape.SEQUENCE:
            _warn_sequence_fallback(type(self))
            return z_mean.unsqueeze(1)  # (B, 1, D) — SEQUENCE (fake temporal axis)
        msg = f"TimeVAE does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    def _build_decoder(self) -> TimeVAEDecoder:
        return TimeVAEDecoder(
            sequence_length=self.sequence_length,
            input_dim=self.input_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            latent_dim=self.latent_dim,
            trend_poly=self.trend_poly,
            custom_seasonality=self.custom_seasonality,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
            use_residual_conn=self.use_residual_conn,
            encoder_last_dense_dim=self._encoder.encoder_last_dense_dim,
        )
