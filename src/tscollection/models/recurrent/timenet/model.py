from __future__ import annotations

__all__ = ['TimeNet']

from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
import torch
from torch import nn

from tscollection.models._mixin import SimpleEncodingMixin

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn.modules.container import Sequential


class GRUWrapper(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output


class TimeNet(LightningModule, SimpleEncodingMixin):
    """TimeNet Model.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/paudan/TimeNet under MIT License.
    """

    def __init__(
        self, hidden_dims: int, num_layers: int, dropout: float = 0.1, learning_rate: float = 1e-3
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dims: int = hidden_dims
        self.num_layers: int = num_layers
        self.dropout: int | float = dropout
        self.encoder: Sequential = self._build_encoder()
        self.decoder: Sequential = self._build_decoder()
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def _build_encoder(self) -> nn.Sequential:
        encoder_layers: list[nn.Module] = [GRUWrapper(1, self.hidden_dims, batch_first=True)]
        for _ in range(1, self.num_layers):
            if self.dropout > 0:
                encoder_layers.append(nn.Dropout(self.dropout))
            encoder_layers.append(GRUWrapper(self.hidden_dims, self.hidden_dims, batch_first=True))
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self) -> nn.Sequential:
        decoder_layers: list[nn.Module] = [
            GRUWrapper(self.hidden_dims, self.hidden_dims, batch_first=True)
        ]
        for i in range(1, self.num_layers):
            if i > 1 and self.dropout > 0:
                ## Add dropout only between GRU layers, not after the last GRU layer
                decoder_layers.append(nn.Dropout(self.dropout))
            decoder_layers.append(GRUWrapper(self.hidden_dims, self.hidden_dims, batch_first=True))
        decoder_layers.append(nn.Linear(self.hidden_dims, 1))
        return nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encode = self.encoder(x)
        return self.decoder(torch.flip(encode, dims=[1]))  # reconstruction target

    def _encode_batch(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Encode one batch — returns the GRU encoder output ``(batch, T, hidden_dims)``."""
        return self.encoder(batch_x.to(self.device))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        output = self(x)
        loss = self.loss_fn(output, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
