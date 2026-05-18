__all__ = ['BandedFourierLayer']

import math

import torch
import torch.fft as fft
import torch.nn as nn


class BandedFourierLayer(nn.Module):
    """
    Banded Fourier Layer for applying banded Fourier transform to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        band (int): Index of the band to process.
        num_bands (int): Total number of bands.
        length (int): Length of the input sequence. Default is 201.
    """

    def __init__(self, in_channels: int, out_channels: int, band: int, num_bands: int, length: int = 201) -> None:
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

        self.weight = nn.Parameter(torch.empty((self.num_frequencies, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_frequencies, out_channels), dtype=torch.cfloat))

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Banded Fourier Layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, time_steps, in_channels).

        Returns:
            torch.Tensor: Output tensor after applying the banded Fourier transform.
        """
        batch_size, time_steps, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(batch_size, time_steps // 2 + 1,
                                 self.out_channels,
                                 device=input.device,
                                 dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._apply_fourier_transform(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _apply_fourier_transform(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the Fourier transform to the input tensor for the specific band.

        Args:
            input (torch.Tensor): Input tensor in the Fourier domain.

        Returns:
            torch.Tensor: Transformed output tensor in the Fourier domain.
        """
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        """
        Initialize the layer parameters.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
