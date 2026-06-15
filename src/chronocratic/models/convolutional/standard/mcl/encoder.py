__all__ = ["FCNEncoder"]

import torch
from torch import nn


class FCNEncoder(nn.Module):
    """Three-block dilated Conv1D encoder for MCL.

    Produces a flat representation of size ``output_dims`` via three
    dilated Conv1D blocks, adaptive average pooling, and a final linear
    projection.
    """

    def __init__(self, input_channels: int, output_dims: int = 320) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of time series into flat FCN representations."""
        return self.layers(x)
