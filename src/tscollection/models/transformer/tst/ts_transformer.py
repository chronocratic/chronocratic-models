from collections.abc import Callable
import math
from typing import cast

import torch
from torch import nn, Tensor
from torch.nn import functional
from torch.nn.modules import (
    BatchNorm1d,
    Dropout,
    Linear,
    MultiheadAttention,
    TransformerEncoderLayer,
)

ActivationFn = Callable[[Tensor], Tensor]


def _get_activation_fn(activation: str) -> ActivationFn:
    if activation == 'relu':
        return functional.relu
    if activation == 'gelu':
        return functional.gelu
    msg = f'activation should be relu/gelu, not {activation}'
    raise ValueError(msg)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx).

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """  # noqa: D205

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 1024, scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.pe: torch.Tensor
        self.register_buffer(
            'pe', pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim].
        """  # noqa: D205
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim].
        """  # noqa: D205
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


type PositionalEncoder = type[FixedPositionalEncoding | LearnablePositionalEncoding]


def get_pos_encoder(pos_encoding: str) -> PositionalEncoder:
    """Return the positional encoding class for ``pos_encoding``."""
    if pos_encoding == 'learnable':
        return LearnablePositionalEncoding
    if pos_encoding == 'fixed':
        return FixedPositionalEncoding

    msg = f"pos_encoding should be 'learnable'/'fixed', not '{pos_encoding}'"
    raise NotImplementedError(msg)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it
    replaces LayerNorm with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """  # noqa: D205

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(
            d_model, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore pickled state with a default activation for older checkpoints."""
        if 'activation' not in state:
            state['activation'] = functional.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: Accepted for compatibility with ``TransformerEncoder``.

        Shape:
            see the docs in Transformer class.
        """
        del is_causal
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pos_encoding: str = 'fixed',
        activation: str = 'gelu',
        norm: str = 'BatchNorm',
        *,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(
            cast('TransformerEncoderLayer', encoder_layer), num_layers
        )

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, x: Tensor, padding_masks: Tensor) -> Tensor:
        """Encode and reconstruct a padded batch.

        Args:
            x: ``(batch_size, seq_length, feat_dim)`` tensor of masked features.
            padding_masks: ``(batch_size, seq_length)`` boolean tensor. ``1`` means
                keep the vector at this position; ``0`` means padding.

        Returns:
            ``(batch_size, seq_length, feat_dim)`` reconstruction tensor.
        """
        # PyTorch transformers use [seq_length, batch_size, feat_dim].
        inp = x.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model
        )
        inp = self.pos_enc(inp)  # add positional encoding
        # Padding-mask logic is reversed for MultiHeadAttention / TransformerEncoderLayer.
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output
