__all__ = ['TCCEncoder']

import torch
from torch import nn


class TCCEncoder(nn.Module):
    """Three-block Conv1D encoder backbone for TS-TCC.

    Returns ``(logits, features)`` where ``features`` is the pre-classification
    representation used for contrastive learning.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        stride: int,
        final_out_channels: int,
        features_len: int,
        num_classes: int,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                32,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.logits = nn.Linear(features_len * final_out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch and return logits plus convolutional features.

        Args:
            x: ``(batch, input_channels, seq_len)``

        Returns:
            logits:   ``(batch, num_classes)``
            features: ``(batch, final_out_channels, reduced_seq_len)``
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        logits = self.logits(x.reshape(x.shape[0], -1))
        return logits, x
