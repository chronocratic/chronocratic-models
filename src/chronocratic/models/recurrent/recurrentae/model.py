"""Recurrent autoencoder based on https://github.com/PyLink88/Recurrent-Autoencoder."""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoder"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from lightning.pytorch import LightningModule
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.recurrent.enums import RecurrentCellType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler


_OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "radam": torch.optim.RAdam,
}


class _RNNLayer(nn.Module):
    """Wraps an RNN cell and discards the hidden state, enabling nn.Sequential chaining."""

    def __init__(self, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return output


def _build_encoder(
    rnn_cls: type,
    n_features: int,
    layers: tuple[int],
    dropout: list[float],
) -> nn.Sequential:
    encoder_layers: list[nn.Module] = [_RNNLayer(rnn_cls(n_features, layers[0], batch_first=True))]
    for i in range(1, len(layers)):
        encoder_layers.append(_RNNLayer(rnn_cls(layers[i - 1], layers[i], batch_first=True)))
        if dropout[i] > 0:
            encoder_layers.append(nn.Dropout(dropout[i]))
    return nn.Sequential(*encoder_layers)


def _build_decoder(
    rnn_cls: type,
    n_features: int,
    layers: tuple[int],
    dropout: list[float],
) -> nn.Sequential:
    decoder_layers: list[nn.Module] = [_RNNLayer(rnn_cls(layers[0], layers[0], batch_first=True))]
    for i in range(1, len(layers)):
        if i > 1 and dropout[i] > 0:
            decoder_layers.append(nn.Dropout(dropout[i]))
        decoder_layers.append(_RNNLayer(rnn_cls(layers[i - 1], layers[i], batch_first=True)))
    decoder_layers.append(nn.Linear(layers[-1], n_features))
    return nn.Sequential(*decoder_layers)


class RecurrentAutoEncoder(LightningModule, BasicEncodingMixin):
    """Recurrent autoencoder for time series representation learning.

    Architecture based on https://github.com/PyLink88/Recurrent-Autoencoder.
    Supports LSTM, GRU, and RNN variants selected via ``recurrent_cell_type``.
    The encoder maps the input to a latent sequence via stacked RNN layers;
    the decoder reconstructs the original sequence from the time-reversed
    latent sequence using a mirrored RNN stack followed by a linear projection.

    Args:
        n_features: Number of input features (channels) per timestep.
        layers: Hidden sizes for each encoder RNN layer, e.g. ``[64, 32]``.
            The decoder uses the reversed order.
        recurrent_cell_type: RNN variant — LSTM, GRU, or RNN.
        dropout: Dropout probability applied between successive layers. A single
            float applies uniformly; a list must match ``len(layers)``.
        loss: Reconstruction objective — ``'mse'`` or ``'mae'``.
        optimizer: Optimizer — ``'adam'``, ``'adamw'``, or ``'radam'``.
        learning_rate: Base learning rate for the optimizer.
        sync_dist: Whether to sync logged metrics across devices.
    """

    def __init__(
        self,
        n_features: int,
        layers: tuple[int],
        recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM,
        dropout: float | list[float] = 0.2,
        loss: Literal["mse", "mae"] = "mse",
        optimizer: Literal["adam", "adamw", "radam"] = "adam",
        learning_rate: float = 1e-3,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.sync_dist = sync_dist

        dropout_list: list[float]
        if isinstance(dropout, list):
            dropout_list = dropout
        else:
            dropout_list = [dropout] * len(layers)
        inverse_layers = layers[::-1]
        inverse_dropout = dropout_list[::-1]

        rnn_cls = getattr(nn, str(recurrent_cell_type).upper())
        self.encoder = _build_encoder(rnn_cls, n_features, layers, dropout_list)
        self.decoder = _build_decoder(rnn_cls, n_features, inverse_layers, inverse_dropout)
        self.loss_fn: nn.Module = nn.MSELoss() if loss == "mse" else nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x``, reverse the latent sequence, and reconstruct."""
        encoded = self.encoder(x)
        return self.decoder(torch.flip(encoded, dims=[1]))

    def _get_encoder(self) -> nn.Module:
        return self.encoder

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, -1, :]

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        x = batch
        loss = self.loss_fn(self(x), x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        x = batch
        loss = self.loss_fn(self(x), x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return _OPTIMIZERS[self.optimizer](self.parameters(), lr=self.learning_rate)
