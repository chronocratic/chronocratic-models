"""Recurrent autoencoder for time series representation learning.

Architecture based on https://github.com/PyLink88/Recurrent-Autoencoder.
"""

from __future__ import annotations

__all__ = ['LSTMAutoEncoder']

from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
import torch
from torch import nn

from tscollection.models._mixin import BasicEncodingMixin

if TYPE_CHECKING:
    from typing import Literal

    from lightning.pytorch.utilities.types import OptimizerLRScheduler


class RecurrentEncoder(nn.Module):
    """Single-layer recurrent encoder that maps a sequence to a latent sequence.

    Supports LSTM, GRU, and RNN variants. The full output sequence is returned
    so that ``BasicEncodingMixin`` can extract the last timestep as a fixed-size
    representation, and so that the encoder output is compatible with the TSRC
    student interface.
    """

    def __init__(self, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full RNN output sequence of shape ``(B, T, latent_dim)``."""
        output, _ = self.rnn(x)
        return output


class RecurrentDecoder(nn.Module):
    """Single-layer recurrent decoder that maps a latent sequence to the input space.

    Accepts the time-reversed encoder output and reconstructs the original
    sequence via a single RNN layer followed by a linear projection.
    """

    def __init__(self, latent_dim: int, n_features: int, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn
        self.linear = nn.Linear(latent_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input sequence from a reversed latent sequence."""
        output, _ = self.rnn(x)
        return self.linear(output)


class LSTMAutoEncoder(LightningModule, BasicEncodingMixin):
    """Recurrent autoencoder for time series representation learning.

    Architecture based on https://github.com/PyLink88/Recurrent-Autoencoder.
    Supports LSTM, GRU, and RNN variants selected via ``rnn_type``. The
    encoder maps the input to a full latent sequence via a single RNN layer.
    The decoder reconstructs the original sequence from the time-reversed
    latent sequence using another RNN layer followed by a linear projection.

    The reconstruction target is the time-reversed input sequence, following
    the original repository's convention.

    This architecture is compatible with the TSRC student interface: the
    encoder exposes a full sequence tensor ``(B, T, latent_dim)`` so that
    ``TSRC.forward()`` can flip it and pass it to the decoder, and
    ``ReconstructionWithHintLoss`` can index ``r2[:, -1, :]`` as the
    fixed-size student representation.

    Args:
        n_features: Number of input features (channels) per timestep.
        latent_dim: Dimensionality of the RNN hidden state used as the
            bottleneck representation.
        rnn_type: Recurrent cell variant — ``'LSTM'``, ``'GRU'``, or
            ``'RNN'``.
        loss_type: Training objective — ``'MSE'`` (mean squared error) or
            ``'MAE'`` (mean absolute error).
        learning_rate: Base learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        rnn_type: Literal['LSTM', 'GRU', 'RNN'] = 'LSTM',
        loss_type: Literal['MSE', 'MAE'] = 'MSE',
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        rnn_cls = getattr(nn, rnn_type)
        self.encoder = RecurrentEncoder(rnn_cls(n_features, latent_dim, batch_first=True))
        self.decoder = RecurrentDecoder(
            latent_dim, n_features, rnn_cls(latent_dim, latent_dim, batch_first=True)
        )
        self.loss_fn: nn.Module = nn.MSELoss() if loss_type == 'MSE' else nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x``, reverse the latent sequence, and reconstruct."""
        encoded = self.encoder(x)
        return self.decoder(torch.flip(encoded, dims=[1]))

    def _get_encoder(self) -> nn.Module:
        """Expose the recurrent encoder to ``BasicEncodingMixin.encode``."""
        return self.encoder

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """Return the last timestep of the encoder output as the representation."""
        return output[:, -1, :]

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training reconstruction loss."""
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the validation reconstruction loss."""
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the Adam optimizer used to train the autoencoder."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
