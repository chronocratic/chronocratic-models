__all__ = ["TCCEncoder"]

import torch
from torch import nn

from chronocratic.models.enums.layers import NormalizationLayerType

# 3-block architecture requires exactly 2 channel/kernel values
_EXPECTED_CHANNEL_COUNT = 2


class TCCEncoder(nn.Module):
    """Three-block Conv1D encoder backbone for TS-TCC.

    Returns the convolutional feature map ``(B, representation_dim, L')`` used
    for contrastive learning and downstream representation extraction.

    Args:
        input_dim: Number of input features (channels).
        conv_kernel_size: Kernel size for the first convolution block.
        stride: Stride for the first convolution block.
        representation_dim: Number of output channels from the encoder.
        dropout_rate: Dropout rate applied after the first conv block.
        encoder_channels: Channel counts for the first two conv blocks.
            Must have exactly 2 elements.
        encoder_inner_kernels: Kernel sizes for the second and third conv
            blocks. Must have exactly 2 elements.
        normalization_layer_type: Normalization strategy. ``CHANNEL`` uses
            GroupNorm(1, C), which is batch-size independent and avoids
            degeneracy at small batch sizes. ``BATCH`` uses BatchNorm1d.
            Defaults to ``CHANNEL``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        conv_kernel_size: int,
        stride: int,
        representation_dim: int = 128,
        dropout_rate: float = 0.35,
        encoder_channels: tuple[int, ...] = (32, 64),
        encoder_inner_kernels: tuple[int, ...] = (8, 8),
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        self.representation_dim = representation_dim

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

        _norm1 = (
            nn.GroupNorm(num_groups=1, num_channels=encoder_channels[0])
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(encoder_channels[0])
        )
        _norm2 = (
            nn.GroupNorm(num_groups=1, num_channels=encoder_channels[1])
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(encoder_channels[1])
        )
        _norm3 = (
            nn.GroupNorm(num_groups=1, num_channels=representation_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(representation_dim)
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_dim,
                encoder_channels[0],
                kernel_size=conv_kernel_size,
                stride=stride,
                bias=False,
                padding=conv_kernel_size // 2,
            ),
            _norm1,
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
            _norm2,
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                encoder_channels[1],
                representation_dim,
                kernel_size=encoder_inner_kernels[1],
                stride=1,
                bias=False,
                padding=encoder_inner_kernels[1] // 2,
            ),
            _norm3,
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch and return the convolutional feature map.

        Args:
            x: ``(batch, seq_len, input_dim)`` — input data in (B,T,C) layout

        Returns:
            features: ``(batch, representation_dim, reduced_seq_len)``
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.conv_block3(x)
