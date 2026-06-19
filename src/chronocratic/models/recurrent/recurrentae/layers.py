"""Pure ``nn`` building blocks for the recurrent autoencoder (no Lightning)."""

from __future__ import annotations

import torch
from torch import nn

from chronocratic.models.recurrent.enums import ReconstructionLoss, RecurrentCellType

_RNN_CLASSES: dict[RecurrentCellType, type[nn.Module]] = {
    RecurrentCellType.LSTM: nn.LSTM,
    RecurrentCellType.GRU: nn.GRU,
    RecurrentCellType.RNN: nn.RNN,
}

_LOSS_FNS: dict[ReconstructionLoss, type[nn.Module]] = {
    ReconstructionLoss.MSE: nn.MSELoss,
    ReconstructionLoss.MAE: nn.L1Loss,
}


def _prepare_dropout(dropout: float | tuple[float, ...], n_layers: int) -> tuple[float, ...]:
    """Expand a scalar dropout to one value per layer, or validate a per-layer tuple.

    Args:
        dropout: A single probability applied to every layer, or one per layer.
        n_layers: Number of RNN layers the dropout must cover.

    Returns:
        A tuple of length ``n_layers``.

    Raises:
        ValueError: If a tuple is given whose length does not match ``n_layers``.
    """
    if isinstance(dropout, (int, float)):
        return (float(dropout),) * n_layers
    if len(dropout) != n_layers:
        msg = f"Expected {n_layers} dropout values, got {len(dropout)}."
        raise ValueError(msg)
    return tuple(dropout)


class _RNNLayer(nn.Module):
    """Wraps an RNN cell and discards the hidden state, enabling nn.Sequential chaining."""

    def __init__(self, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return output


def _build_encoder(
    rnn_cls: type, input_dims: int, layers: tuple[int, ...], dropout: tuple[float, ...]
) -> nn.Sequential:
    encoder_layers: list[nn.Module] = [_RNNLayer(rnn_cls(input_dims, layers[0], batch_first=True))]
    for i in range(1, len(layers)):
        encoder_layers.append(_RNNLayer(rnn_cls(layers[i - 1], layers[i], batch_first=True)))
        if dropout[i] > 0:
            encoder_layers.append(nn.Dropout(dropout[i]))
    return nn.Sequential(*encoder_layers)


def _build_decoder(
    rnn_cls: type, input_dims: int, layers: tuple[int, ...], dropout: tuple[float, ...]
) -> nn.Sequential:
    decoder_layers: list[nn.Module] = [_RNNLayer(rnn_cls(layers[0], layers[0], batch_first=True))]
    for i in range(1, len(layers)):
        if i > 1 and dropout[i] > 0:
            decoder_layers.append(nn.Dropout(dropout[i]))
        decoder_layers.append(_RNNLayer(rnn_cls(layers[i - 1], layers[i], batch_first=True)))
    decoder_layers.append(nn.Linear(layers[-1], input_dims))
    return nn.Sequential(*decoder_layers)
