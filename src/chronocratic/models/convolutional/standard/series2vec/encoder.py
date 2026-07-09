import torch
from torch import nn
from torch.nn import init

from chronocratic.models.enums.layers import NormalizationLayerType


class DisjoinEncoder(nn.Module):
    """DisjoinEncoder with a selectable normalization strategy.

    Args:
        input_dim: Number of input channels (spatial conv kernel height).
        embedding_dim: Channel count of the temporal/spatial conv blocks.
        representation_dim: Channel count of the representation conv block.
            This is the **per-branch** dimension (each DisjoinEncoder produces
            this many features); the final concatenated output is
            ``2 * representation_dim``.
        kernel_size: Temporal convolution kernel width.
        normalization_layer_type: Normalization strategy. ``CHANNEL``
            (default) uses GroupNorm(num_groups=1, C), which is per-sample
            and works correctly at batch_size=1. ``BATCH`` uses
            BatchNorm2d/BatchNorm1d to reproduce the upstream Series2Vec
            architecture exactly.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        representation_dim: int,
        kernel_size: int,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()

        _temporal_norm = (
            nn.GroupNorm(num_groups=1, num_channels=embedding_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm2d(embedding_dim)
        )
        _spatial_norm = (
            nn.GroupNorm(num_groups=1, num_channels=embedding_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm2d(embedding_dim)
        )
        _rep_norm = (
            nn.GroupNorm(num_groups=1, num_channels=representation_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(representation_dim)
        )

        self.temporal_CNN = nn.Sequential(
            nn.Conv2d(1, embedding_dim, kernel_size=(1, kernel_size), padding="valid"),
            _temporal_norm,
            nn.GELU(),
        )

        self.spatial_CNN = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(input_dim, 1), padding="valid"),
            _spatial_norm,
            nn.GELU(),
        )

        self.rep_CNN = nn.Sequential(
            nn.Conv1d(embedding_dim, representation_dim, kernel_size=3), _rep_norm, nn.GELU()
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input shaped ``(batch, channels, time)``."""
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x)
        x = self.rep_CNN(x.squeeze(2))
        return x

    def initialize_weights(self) -> None:
        """Initialize convolution weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
