from collections.abc import Sequence
import math

import torch
from torch import fft, nn
import torch.nn.functional as F  # noqa: N812

__all__ = ["BandedFourierLayer", "LevelModel", "ResidualConnection", "SeasonalLayer", "TrendLayer"]

Seasonality = tuple[int, int]


class BandedFourierLayer(nn.Module):
    """Banded Fourier Layer for applying banded Fourier transform to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        band (int): Index of the band to process.
        num_bands (int): Total number of bands.
        length (int): Length of the input sequence. Default is 201.
    """

    def __init__(
        self, in_channels: int, out_channels: int, band: int, num_bands: int, length: int = 201
    ) -> None:
        super().__init__()

        self.length = length
        self.total_frequencies = (self.length // 2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.band = band
        self.num_bands = num_bands

        self.num_frequencies = self.total_frequencies // self.num_bands
        if self.band == self.num_bands - 1:
            self.num_frequencies += self.total_frequencies % self.num_bands

        self.start = self.band * (self.total_frequencies // self.num_bands)
        self.end = self.start + self.num_frequencies

        self.weight = nn.Parameter(
            torch.empty((self.num_frequencies, in_channels, out_channels), dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(
            torch.empty((self.num_frequencies, out_channels), dtype=torch.cfloat)
        )

        self.reset_parameters()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Banded Fourier Layer.

        Args:
            input_tensor: Input tensor of shape (batch_size, time_steps, in_channels).

        Returns:
            Output tensor after applying the banded Fourier transform.
        """
        batch_size, time_steps, _ = input_tensor.shape
        input_fft = fft.rfft(input_tensor, dim=1)
        output_fft = torch.zeros(
            batch_size,
            time_steps // 2 + 1,
            self.out_channels,
            device=input_tensor.device,
            dtype=torch.cfloat,
        )
        output_fft[:, self.start : self.end] = self._apply_fourier_transform(input_fft)
        return fft.irfft(output_fft, n=input_tensor.size(1), dim=1)

    def _apply_fourier_transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the Fourier transform to the input tensor for the specific band.

        Args:
            input_tensor: Input tensor in the Fourier domain.

        Returns:
            Transformed output tensor in the Fourier domain.
        """
        output = torch.einsum("bti,tio->bto", input_tensor[:, self.start : self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        """Initialize the layer parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # noqa: SLF001
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class TrendLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dims: int,
        latent_dim: int,
        trend_poly: int,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.input_dims * self.trend_poly)
        self.trend_dense2 = nn.Linear(
            self.input_dims * self.trend_poly, self.input_dims * self.trend_poly
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return polynomial trend values for each latent vector."""
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.input_dims, self.trend_poly)

        lin_space = torch.arange(0, float(self.sequence_length), 1, device=z.device) / self.sequence_length
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)

        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.permute(0, 2, 1)
        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dims: int,
        latent_dim: int,
        custom_seas: Sequence[Seasonality],
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dims = input_dims
        self.custom_seas = custom_seas

        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(latent_dim, input_dims * num_seasons)
                for num_seasons, len_per_season in custom_seas
            ]
        )

    def _get_season_indexes_over_seq(self, num_seasons: int, len_per_season: int) -> torch.Tensor:
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.view(-1)
        season_indexes = season_indexes.repeat(self.sequence_length // len_per_season + 1)[: self.sequence_length]
        return season_indexes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return additive seasonal values for each latent vector."""
        batch_size = z.shape[0]
        ones_tensor = torch.ones(
            (batch_size, self.input_dims, self.sequence_length), dtype=torch.int32, device=z.device
        )

        seasonal_components: list[torch.Tensor] = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)
            season_params = season_params.view(-1, self.input_dims, num_seasons)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            ).to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(1, 1, -1)
            season_vals = torch.gather(season_params, 2, dim2_idxes)

            seasonal_components.append(season_vals)

        all_seas_vals = torch.stack(seasonal_components, dim=-1)
        seasonal_values = torch.sum(all_seas_vals, dim=-1)
        seasonal_values = seasonal_values.permute(0, 2, 1)

        return seasonal_values

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Return the output shape for Keras-compatible callers."""
        return (input_shape[0], self.sequence_length, self.input_dims)


class LevelModel(nn.Module):
    def __init__(self, latent_dim: int, input_dims: int, sequence_length: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dims = input_dims
        self.sequence_length = sequence_length
        self.level_dense1 = nn.Linear(self.latent_dim, self.input_dims)
        self.level_dense2 = nn.Linear(self.input_dims, self.input_dims)
        self.relu = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the level component for each latent vector."""
        level_params = self.relu(self.level_dense1(z))
        level_params = self.level_dense2(level_params)
        level_params = level_params.view(-1, 1, self.input_dims)

        ones_tensor = torch.ones((1, self.sequence_length, 1), dtype=torch.float32, device=z.device)
        level_vals = level_params * ones_tensor
        return level_vals


class ResidualConnection(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dims: int,
        hidden_layer_sizes: Sequence[int],
        latent_dim: int,
        encoder_last_dense_dim: int,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dims = input_dims
        self.hidden_layer_sizes = hidden_layer_sizes

        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers: nn.ModuleList = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]

        for num_filters in reversed(hidden_layer_sizes[:-1]):
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            in_channels = num_filters

        self.deconv_layers.append(
            nn.ConvTranspose1d(
                in_channels, input_dims, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )

        length_in = encoder_last_dense_dim // hidden_layer_sizes[-1]
        for _ in range(len(hidden_layer_sizes)):
            length_in = (length_in - 1) * 2 - 2 * 1 + 3 + 1
        length_final = length_in

        self.final_dense = nn.Linear(input_dims * length_final, sequence_length * input_dims)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the residual decoder branch for each latent vector."""
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)

        for deconv in list(self.deconv_layers)[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))

        x = x.flatten(1)
        x = self.final_dense(x)
        residuals = x.view(-1, self.sequence_length, self.input_dims)
        return residuals
