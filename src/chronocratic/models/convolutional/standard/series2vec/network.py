__all__ = ["Series2VecNetwork"]

import torch
from torch import nn

from chronocratic.models.convolutional.standard.series2vec.encoder import DisjoinEncoder
from chronocratic.models.enums.layers import NormalizationLayerType


class Series2VecNetwork(nn.Module):
    """Pure PyTorch Series2Vec architecture.

    Public methods accept input shaped ``(batch, time, channels)``. The original
    Series2Vec convolution blocks operate on ``(batch, channels, time)``, so this
    class transposes internally before calling the encoders.

    Both :class:`DisjoinEncoder` instances (temporal and frequency branches)
    use GroupNorm for normalization, ensuring correct gradient flow at
    batch_size=1.

    The network produces representations only; downstream classification and
    regression use :class:`SupervisedModule` from
    ``chronocratic.models.supervised``.

    The ``representation_dims`` parameter controls the **output dimension** of
    :meth:`encode` (temporal + frequency concatenated). Internally, each branch
    encodes to ``representation_dims // 2`` features, so the parameter must be
    even.

    Args:
        input_dims: Number of input features (channels).
        embedding_dims: Token embedding dimensionality for the conv encoders.
        num_heads: Number of attention heads in the cross-attention layer.
        feedforward_dims: Hidden dimensionality of the feed-forward network.
        representation_dims: Output dimensionality of :meth:`encode`
            (temporal + frequency concatenated). Must be even.
        dropout_rate: Dropout probability applied in attention and FFN.
        encoder_kernel_size: Kernel size for the convolutional tokenizer.
        normalization_layer_type: Normalization strategy for the
            DisjoinEncoder instances. ``CHANNEL`` (default) uses GroupNorm
            for batch_size=1 safety. ``BATCH`` uses BatchNorm.
    """

    def __init__(
        self,
        input_dims: int,
        embedding_dims: int = 16,
        num_heads: int = 8,
        feedforward_dims: int = 256,
        representation_dims: int = 320,
        dropout_rate: float = 0.01,
        encoder_kernel_size: int = 8,
        *,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        if representation_dims % 2 != 0:
            msg = f"representation_dims must be even, got {representation_dims}"
            raise ValueError(msg)

        branch_dims = representation_dims // 2
        if branch_dims % num_heads != 0:
            msg = (
                f"representation_dims // 2 ({branch_dims}) must be divisible by "
                f"num_heads ({num_heads}) for MultiheadAttention"
            )
            raise ValueError(msg)
        self._branch_representation_dim = branch_dims

        self.embed_layer = DisjoinEncoder(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            representation_dims=branch_dims,
            kernel_size=encoder_kernel_size,
            normalization_layer_type=normalization_layer_type,
        )
        self.embed_layer_f = DisjoinEncoder(
            input_dims=input_dims,
            embedding_dims=embedding_dims,
            representation_dims=branch_dims,
            kernel_size=encoder_kernel_size,
            normalization_layer_type=normalization_layer_type,
        )

        self.layer_norm = nn.LayerNorm(branch_dims, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(branch_dims, eps=1e-5)
        self.attention_layer = nn.MultiheadAttention(branch_dims, num_heads, dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(branch_dims, feedforward_dims),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dims, branch_dims),
            nn.Dropout(dropout_rate),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap_f = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
        expected_ndim = 3
        if x.ndim != expected_ndim:
            msg = "Series2Vec input must have shape (batch, time, channels)."
            raise ValueError(msg)
        return x.transpose(1, 2)

    @staticmethod
    def _real_fft(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, norm="backward").real

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
        """Return ``(batch, representation_dims)`` temporal + frequency concat."""
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

        temporal_representation = out.squeeze(0)
        frequency_representation = x_f.squeeze(0)
        temporal_distance = torch.cdist(temporal_representation, temporal_representation)
        frequency_distance = torch.cdist(frequency_representation, frequency_representation)
        return (
            temporal_distance,
            frequency_distance,
            temporal_representation,
            frequency_representation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return representations of shape ``(batch, representation_dims)``."""
        return self.encode(x)

    @property
    def representation_dim(self) -> int:
        """Output representation size (temporal + frequency concatenated).

        Returns:
            The number of features produced by :meth:`encode`.
        """
        return self._branch_representation_dim * 2
