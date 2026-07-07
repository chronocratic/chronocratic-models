__all__ = ["FCNEncoder"]

import torch
from torch import nn

from chronocratic.models.enums.layers import NormalizationLayerType


class FCNEncoder(nn.Module):
    """Configurable dilated Conv1D encoder for MCL.

    Builds ``len(encoder_channels)`` dilated Conv1D blocks dynamically,
    followed by adaptive average pooling and a final linear projection to
    ``output_dims``.

    Args:
        input_dims: Number of input feature channels.
        output_dims: Dimension of the flat encoder output.
        encoder_channels: Tuple of channel counts for each Conv1d block.
        encoder_kernels: Tuple of kernel sizes for each Conv1d block.
        encoder_dilations: Tuple of dilation rates for each Conv1d block.
        normalization_layer_type: Normalization strategy. Use ``CHANNEL``
            for GroupNorm(1, C) which works correctly at batch_size=1,
            or ``BATCH`` for BatchNorm1d(C). Defaults to ``CHANNEL``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 128,
        encoder_channels: tuple[int, ...] = (128, 256, 128),
        encoder_kernels: tuple[int, ...] = (7, 5, 3),
        encoder_dilations: tuple[int, ...] = (2, 4, 8),
        *,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.encoder_kernels = encoder_kernels
        self.encoder_dilations = encoder_dilations

        layers: list[nn.Module] = []
        in_ch = input_dims
        for ch, k, d in zip(encoder_channels, encoder_kernels, encoder_dilations, strict=True):
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2 * d, dilation=d))
            if normalization_layer_type == NormalizationLayerType.CHANNEL:
                layers.append(nn.GroupNorm(num_groups=1, num_channels=ch))
            else:
                layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            in_ch = ch

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch, output_dims))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of time series into flat FCN representations.

        Args:
            x: Input batch of shape ``(batch, seq_len, input_dims)``.

        Returns:
            Flat representations of shape ``(batch, output_dims)``.
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        return self.layers(x)
