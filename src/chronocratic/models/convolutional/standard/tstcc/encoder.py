__all__ = ["TCCEncoder"]

import torch
from torch import nn

# 3-block architecture requires exactly 2 channel/kernel values
_EXPECTED_CHANNEL_COUNT = 2


class TCCEncoder(nn.Module):
    """Three-block Conv1D encoder backbone for TS-TCC.

    Returns the convolutional feature map ``(B, output_dims, L')`` used for
    contrastive learning and downstream representation extraction.

    Args:
        input_dims: Number of input features (channels).
        conv_kernel_size: Kernel size for the first convolution block.
        stride: Stride for the first convolution block.
        output_dims: Number of output channels from the encoder.
        dropout_rate: Dropout rate applied after the first conv block.
        encoder_channels: Channel counts for the first two conv blocks.
            Must have exactly 2 elements.
        encoder_inner_kernels: Kernel sizes for the second and third conv
            blocks. Must have exactly 2 elements.
        norm: Normalization strategy. ``"layer"`` uses GroupNorm(1, C),
            which is batch-size independent and avoids degeneracy at
            small batch sizes. ``"batch"`` uses BatchNorm1d for backward
            compatibility. Defaults to ``"layer"``.
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
        *,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        self.output_dims = output_dims

        if norm not in ("layer", "batch"):
            msg = f"norm must be 'layer' or 'batch', got '{norm}'"
            raise ValueError(msg)
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
            if norm == "layer"
            else nn.BatchNorm1d(encoder_channels[0])
        )
        _norm2 = (
            nn.GroupNorm(num_groups=1, num_channels=encoder_channels[1])
            if norm == "layer"
            else nn.BatchNorm1d(encoder_channels[1])
        )
        _norm3 = (
            nn.GroupNorm(num_groups=1, num_channels=output_dims)
            if norm == "layer"
            else nn.BatchNorm1d(output_dims)
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_dims,
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
                output_dims,
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
            x: ``(batch, seq_len, input_dims)`` — input data in (B,T,C) layout

        Returns:
            features: ``(batch, output_dims, reduced_seq_len)``
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.conv_block3(x)
