"""Recurrent autoencoder based on https://github.com/PyLink88/Recurrent-Autoencoder."""

from __future__ import annotations

__all__ = ["RecurrentAutoEncoder"]

from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.recurrent.enums import OptimizerName, ReconstructionLoss, RecurrentCellType

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRScheduler


_OPTIMIZERS: dict[OptimizerName, Callable[..., torch.optim.Optimizer]] = {
    OptimizerName.ADAM: torch.optim.Adam,
    OptimizerName.ADAMW: torch.optim.AdamW,
    OptimizerName.RADAM: torch.optim.RAdam,
}

_RNN_CLASSES: dict[RecurrentCellType, type[nn.Module]] = {
    RecurrentCellType.LSTM: nn.LSTM,
    RecurrentCellType.GRU: nn.GRU,
    RecurrentCellType.RNN: nn.RNN,
}

_LOSS_FNS: dict[ReconstructionLoss, type[nn.Module]] = {
    ReconstructionLoss.MSE: nn.MSELoss,
    ReconstructionLoss.MAE: nn.L1Loss,
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


class RecurrentAutoEncoder(LightningModule, BasicEncodingMixin):
    """Recurrent autoencoder for time series representation learning.

    Architecture based on https://github.com/PyLink88/Recurrent-Autoencoder.
    Supports LSTM, GRU, and RNN variants selected via ``recurrent_cell_type``.
    The encoder maps the input to a latent sequence via stacked RNN layers;
    the decoder reconstructs the original sequence from the time-reversed
    latent sequence using a mirrored RNN stack followed by a linear projection.

    Args:
        input_dims: Number of input features (channels) per timestep.
        layers: Hidden sizes for each encoder RNN layer, e.g. ``(64, 32)``.
            The decoder uses the reversed order.
        recurrent_cell_type: RNN variant — LSTM, GRU, or RNN.
        dropout: Dropout probability applied between successive layers. A single
            float applies uniformly; a tuple must match ``len(layers)``.
        loss: Reconstruction objective — ``'mse'`` or ``'mae'``.
        optimizer: Optimizer — ``'adam'``, ``'adamw'``, or ``'radam'``.
        learning_rate: Base learning rate for the optimizer.
        sync_dist: Whether to sync logged metrics across devices.
    """

    def __init__(
        self,
        input_dims: int,
        layers: tuple[int, ...],
        recurrent_cell_type: RecurrentCellType = RecurrentCellType.LSTM,
        dropout: float | tuple[float, ...] = 0.2,
        loss: ReconstructionLoss = ReconstructionLoss.MSE,
        optimizer: OptimizerName = OptimizerName.ADAM,
        learning_rate: float = 1e-3,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        recurrent_cell_type = RecurrentCellType(str(recurrent_cell_type).lower())
        self.input_dims = input_dims
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.sync_dist = sync_dist

        dropout_tuple: tuple[float, ...]
        dropout_tuple = (dropout,) * len(layers) if isinstance(dropout, float) else tuple(dropout)
        inverse_layers = layers[::-1]
        inverse_dropout = dropout_tuple[::-1]

        rnn_cls = _RNN_CLASSES[recurrent_cell_type]
        self.encoder = _build_encoder(rnn_cls, input_dims, layers, dropout_tuple)
        self.decoder = _build_decoder(rnn_cls, input_dims, inverse_layers, inverse_dropout)
        self.loss_fn: nn.Module = _LOSS_FNS[loss]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x``, reverse the latent sequence, and reconstruct."""
        encoded = self.encoder(x)
        return self.decoder(torch.flip(encoded, dims=[1]))

    def _get_encoder(self) -> nn.Module:
        return self.encoder

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, -1, :]

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log reconstruction loss for a training batch."""
        x = batch
        loss = self.loss_fn(self(x), x)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log reconstruction loss for a validation batch."""
        x = batch
        loss = self.loss_fn(self(x), x)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the optimizer configured with the model's learning rate."""
        return _OPTIMIZERS[self.optimizer](self.parameters(), lr=self.learning_rate)
