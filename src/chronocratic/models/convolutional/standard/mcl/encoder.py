__all__ = ["FCNEncoder"]

import torch
from torch import nn


class FCNEncoder(nn.Module):
    """Configurable dilated Conv1D encoder for MCL.

    Builds ``len(encoder_channels)`` dilated Conv1D blocks dynamically,
    followed by adaptive average pooling and a final linear projection to
    ``output_dims``.

    Args:
        input_channels: Number of input feature channels.
        output_dims: Dimension of the flat encoder output.
        encoder_channels: Tuple of channel counts for each Conv1d block.
        encoder_kernels: Tuple of kernel sizes for each Conv1d block.
        encoder_dilations: Tuple of dilation rates for each Conv1d block.
    """

    def __init__(
        self,
        input_channels: int,
        output_dims: int = 320,
        encoder_channels: tuple[int, ...] = (128, 256, 128),
        encoder_kernels: tuple[int, ...] = (7, 5, 3),
        encoder_dilations: tuple[int, ...] = (2, 4, 8),
    ) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.encoder_kernels = encoder_kernels
        self.encoder_dilations = encoder_dilations

        layers: list[nn.Module] = []
        in_ch = input_channels
        for ch, k, d in zip(encoder_channels, encoder_kernels, encoder_dilations, strict=True):
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2 * d, dilation=d))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            in_ch = ch

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch, output_dims))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of time series into flat FCN representations."""
        return self.layers(x)
