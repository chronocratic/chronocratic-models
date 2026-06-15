import torch
from torch import nn
from torch.nn import init


class DisjoinEncoder(nn.Module):
    def __init__(
        self, input_dims: int, embedding_dims: int, representation_dims: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.temporal_CNN = nn.Sequential(
            nn.Conv2d(1, embedding_dims, kernel_size=(1, kernel_size), padding="valid"),
            nn.BatchNorm2d(embedding_dims),
            nn.GELU(),
        )

        self.spatial_CNN = nn.Sequential(
            nn.Conv2d(embedding_dims, embedding_dims, kernel_size=(input_dims, 1), padding="valid"),
            nn.BatchNorm2d(embedding_dims),
            nn.GELU(),
        )

        self.rep_CNN = nn.Sequential(
            nn.Conv1d(embedding_dims, representation_dims, kernel_size=3),
            nn.BatchNorm1d(representation_dims),
            nn.GELU(),
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
