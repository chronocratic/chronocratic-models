from __future__ import annotations

__all__ = ["TimeNet"]

from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn.modules.container import Sequential


class GRUWrapper(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, *, batch_first: bool = True
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the GRU output sequence."""
        output, _ = self.gru(x)
        return output


class TimeNet(LightningModule, BasicEncodingMixin):
    """TimeNet Model.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/paudan/TimeNet under MIT License.
    """

    def __init__(
        self,
        hidden_dims: int,
        depth: int,
        input_dims: int = 1,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._input_dims: int = input_dims
        self._hidden_dims: int = hidden_dims
        self._depth: int = depth
        self._dropout_rate: float = dropout_rate
        self._encoder: Sequential = self._build_encoder()
        self._decoder: Sequential = self._build_decoder()
        self._learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    @property
    def encoder(self) -> Sequential:
        """Return the GRU encoder."""
        return self._encoder

    @property
    def decoder(self) -> Sequential:
        """Return the GRU decoder."""
        return self._decoder

    def _build_encoder(self) -> nn.Sequential:
        encoder_layers: list[nn.Module] = [
            GRUWrapper(self._input_dims, self._hidden_dims, batch_first=True)
        ]
        for _ in range(1, self._depth):
            if self._dropout_rate > 0:
                encoder_layers.append(nn.Dropout(self._dropout_rate))
            encoder_layers.append(
                GRUWrapper(self._hidden_dims, self._hidden_dims, batch_first=True)
            )
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self) -> nn.Sequential:
        decoder_layers: list[nn.Module] = [
            GRUWrapper(self._hidden_dims, self._hidden_dims, batch_first=True)
        ]
        for i in range(1, self._depth):
            if i > 1 and self._dropout_rate > 0:
                decoder_layers.append(nn.Dropout(self._dropout_rate))
            decoder_layers.append(
                GRUWrapper(self._hidden_dims, self._hidden_dims, batch_first=True)
            )
        decoder_layers.append(nn.Linear(self._hidden_dims, self._input_dims))
        return nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x`` from the reversed encoder sequence."""
        encode = self._encoder(x)
        return self._decoder(torch.flip(encode, dims=[1]))  # reconstruction target

    def _get_encoder(self) -> nn.Module:
        """Expose the GRU encoder to ``BasicEncodingMixin.encode``."""
        return self._encoder

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """Select the final timestep as the pooled representation."""
        return output[:, -1, :]

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training reconstruction loss."""
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the validation reconstruction loss."""
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the Adam optimizer used to train TimeNet."""
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)
