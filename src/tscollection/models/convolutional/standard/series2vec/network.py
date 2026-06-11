__all__ = ['Series2VecNetwork']

import torch
from torch import nn

from tscollection.models.convolutional.standard.series2vec.encoder import DisjoinEncoder


class Series2VecNetwork(nn.Module):
    """Pure PyTorch Series2Vec architecture.

    Public methods accept input shaped ``(batch, time, channels)``. The original
    Series2Vec convolution blocks operate on ``(batch, channels, time)``, so this
    class transposes internally before calling the encoders.

    The network produces representations only; downstream classification is
    implemented as a separate head (see ``series2vec/heads.py``).
    """

    def __init__(
        self,
        input_dims: int,
        embedding_dims: int,
        num_heads: int,
        feedforward_dims: int,
        representation_dims: int,
        dropout_rate: float,
        encoder_kernel_size: int = 8,
    ) -> None:
        super().__init__()

        self.embed_layer = DisjoinEncoder(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            representation_dims=representation_dims,
            kernel_size=encoder_kernel_size,
        )
        self.embed_layer_f = DisjoinEncoder(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            representation_dims=representation_dims,
            kernel_size=encoder_kernel_size,
        )

        self.layer_norm = nn.LayerNorm(representation_dims, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(representation_dims, eps=1e-5)
        self.attention_layer = nn.MultiheadAttention(representation_dims, num_heads, dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(representation_dims, feedforward_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dims, representation_dims),
            nn.Dropout(dropout_rate),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap_f = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
        expected_ndim = 3
        if x.ndim != expected_ndim:
            msg = 'Series2Vec input must have shape (batch, time, channels).'
            raise ValueError(msg)
        return x.transpose(1, 2)

    @staticmethod
    def _real_fft(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x).real

    def _temporal_representation(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_channels_first(x)
        out = self.embed_layer(x)
        return self.gap(out).squeeze(-1)

    def _frequency_representation(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_channels_first(x)
        x_f = self._real_fft(x)
        out_f = self.embed_layer_f(x_f)
        return self.gap_f(out_f).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, 2 * representation_dims)`` temporal + frequency concat."""
        temporal_representation = self._temporal_representation(x)
        frequency_representation = self._frequency_representation(x)
        return torch.cat((temporal_representation, frequency_representation), dim=1)

    def pretrain_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pairwise distances and representations used by the pretraining loss."""
        x_channels_first = self._to_channels_first(x)
        x_src = self.embed_layer(x_channels_first)
        x_src = self.gap(x_src)
        x_src = x_src.permute(2, 0, 1)

        attention_output, _ = self.attention_layer(x_src, x_src, x_src)
        attention_output = attention_output + x_src
        attention_output = self.layer_norm(attention_output)
        out = attention_output + self.feed_forward(attention_output)
        out = self.layer_norm_2(out)

        x_f = self._real_fft(x_channels_first)
        x_f = self.embed_layer_f(x_f)
        x_f = self.gap_f(x_f)
        x_f = x_f.permute(2, 0, 1)

        temporal_representation = out[:, 0, :]
        frequency_representation = x_f[:, 0, :]
        temporal_distance = torch.cdist(temporal_representation, temporal_representation)
        frequency_distance = torch.cdist(frequency_representation, frequency_representation)
        return (
            temporal_distance,
            frequency_distance,
            temporal_representation,
            frequency_representation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return representations of shape ``(batch, 2 * representation_dims)``."""
        return self.encode(x)
