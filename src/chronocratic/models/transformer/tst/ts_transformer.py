from collections.abc import Callable
import math
from typing import cast

__all__ = [
    "FixedPositionalEncoding",
    "LearnablePositionalEncoding",
    "TSTransformerEncoder",
    "TransformerBatchNormEncoderLayer",
    "_get_activation_fn",
    "get_pos_encoder",
]

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
    if activation == "relu":
        return functional.relu
    if activation == "gelu":
        return functional.gelu
    msg = f"activation should be relu/gelu, not {activation}"
    raise ValueError(msg)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/hidden_dims))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/hidden_dims))
        \text{where pos is the word position and i is the embed idx).

    Args:
        hidden_dims: the embed dim (required).
        dropout_rate: the dropout value (default=0.1).
        sequence_length: the max. length of the incoming sequence (default=1024).
    """  # noqa: D205

    def __init__(
        self,
        hidden_dims: int,
        dropout_rate: float = 0.1,
        sequence_length: int = 1024,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(sequence_length, hidden_dims)  # positional encoding
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dims, 2).float() * (-math.log(10000.0) / hidden_dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.pe: torch.Tensor
        self.register_buffer(
            "pe", pe
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
    def __init__(
        self, hidden_dims: int, dropout_rate: float = 0.1, sequence_length: int = 1024
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        # Each position gets its own embedding
        # Since indices are always 0 ... sequence_length, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(sequence_length, 1, hidden_dims)
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
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    if pos_encoding == "fixed":
        return FixedPositionalEncoding

    msg = f"pos_encoding should be 'learnable'/'fixed', not '{pos_encoding}'"
    raise NotImplementedError(msg)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it
    replaces LayerNorm with BatchNorm.

    Args:
        hidden_dims: the number of expected features in the input (required).
        num_heads: the number of heads in the multiheadattention models (required).
        feedforward_dims: the dimension of the feedforward network model (default=2048).
        dropout_rate: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """  # noqa: D205

    def __init__(
        self,
        hidden_dims: int,
        num_heads: int,
        feedforward_dims: int = 2048,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_dims, num_heads, dropout=dropout_rate)
        # Implementation of Feedforward model
        self.linear1 = Linear(hidden_dims, feedforward_dims)
        self.dropout = Dropout(dropout_rate)
        self.linear2 = Linear(feedforward_dims, hidden_dims)

        self.norm1 = BatchNorm1d(
            hidden_dims, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(hidden_dims, eps=1e-5)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore pickled state with a default activation for older checkpoints."""
        if "activation" not in state:
            state["activation"] = functional.relu
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
        input_dims: int,
        sequence_length: int,
        hidden_dims: int,
        num_heads: int,
        depth: int,
        feedforward_dims: int,
        dropout_rate: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        *,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads

        self.project_inp = nn.Linear(input_dims, hidden_dims)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            hidden_dims, dropout_rate=dropout_rate * (1.0 - freeze), sequence_length=sequence_length
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                hidden_dims,
                self.num_heads,
                feedforward_dims,
                dropout_rate * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                hidden_dims,
                self.num_heads,
                feedforward_dims,
                dropout_rate * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(
            cast("TransformerEncoderLayer", encoder_layer), depth, enable_nested_tensor=False
        )

        self.output_layer = nn.Linear(hidden_dims, input_dims)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout_rate)

        self.input_dims = input_dims

    def forward(self, x: Tensor, padding_masks: Tensor) -> Tensor:
        """Encode and reconstruct a padded batch.

        Args:
            x: ``(batch_size, seq_length, input_dims)`` tensor of masked features.
            padding_masks: ``(batch_size, seq_length)`` boolean tensor. ``1`` means
                keep the vector at this position; ``0`` means padding.

        Returns:
            ``(batch_size, seq_length, input_dims)`` reconstruction tensor.
        """
        # PyTorch transformers use [seq_length, batch_size, input_dims].
        inp = x.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.hidden_dims)
        inp = self.pos_enc(inp)  # add positional encoding
        # Padding-mask logic is reversed for MultiHeadAttention / TransformerEncoderLayer.
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, hidden_dims)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_dims)
        output = self.dropout1(output)
        output = self.output_layer(output)  # (batch_size, seq_length, input_dims)

        return output

    def encode_representations(self, x: Tensor, padding_masks: Tensor) -> Tensor:
        """Return transformer representations before output_layer.

        Args:
            x: ``(batch_size, seq_length, input_dims)`` tensor of masked features.
            padding_masks: ``(batch_size, seq_length)`` boolean tensor. ``1`` means
                keep the vector at this position; ``0`` means padding.

        Returns:
            ``(batch_size, seq_length, hidden_dims)`` representation tensor.
        """
        # PyTorch transformers use [seq_length, batch_size, input_dims].
        inp = x.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.hidden_dims)
        inp = self.pos_enc(inp)  # add positional encoding
        # Padding-mask logic is reversed for MultiHeadAttention / TransformerEncoderLayer.
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, hidden_dims)
        output = self.act(output)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_dims)
        return self.dropout1(output)
