__all__ = ["TCCEncoder"]

import torch
from torch import nn

# 3-block architecture requires exactly 2 channel/kernel values
_EXPECTED_CHANNEL_COUNT = 2


class TCCEncoder(nn.Module):
    """Three-block Conv1D encoder backbone for TS-TCC.

    Returns the convolutional feature map ``(B, output_dims, L')`` used for
    contrastive learning and downstream representation extraction.
    """

    def __init__(
        self,
        input_dims: int,
        conv_kernel_size: int,
        stride: int,
        output_dims: int = 128,
        dropout_rate: float = 0.35,
        encoder_channels: tuple[int, ...] = (32, 64),
        encoder_inner_kernels: tuple[int, ...] = (8, 8),
    ) -> None:
        super().__init__()
        self.output_dims = output_dims

        if len(encoder_channels) != _EXPECTED_CHANNEL_COUNT:
            msg = (
                f"encoder_channels must have exactly {_EXPECTED_CHANNEL_COUNT} elements, "
                f"got {len(encoder_channels)}"
            )
            raise ValueError(msg)
        if len(encoder_inner_kernels) != _EXPECTED_CHANNEL_COUNT:
            msg = (
                f"encoder_inner_kernels must have exactly {_EXPECTED_CHANNEL_COUNT} elements, "
                f"got {len(encoder_inner_kernels)}"
            )
            raise ValueError(msg)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_dims,
                encoder_channels[0],
                kernel_size=conv_kernel_size,
                stride=stride,
                bias=False,
                padding=conv_kernel_size // 2,
            ),
            nn.BatchNorm1d(encoder_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout_rate),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                encoder_channels[0],
                encoder_channels[1],
                kernel_size=encoder_inner_kernels[0],
                stride=1,
                bias=False,
                padding=encoder_inner_kernels[0] // 2,
            ),
            nn.BatchNorm1d(encoder_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                encoder_channels[1],
                output_dims,
                kernel_size=encoder_inner_kernels[1],
                stride=1,
                bias=False,
                padding=encoder_inner_kernels[1] // 2,
            ),
            nn.BatchNorm1d(output_dims),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch and return the convolutional feature map.

        Args:
            x: ``(batch, seq_len, input_dims)`` — input data in (B,T,C) layout

        Returns:
            features: ``(batch, output_dims, reduced_seq_len)``
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.conv_block3(x)
