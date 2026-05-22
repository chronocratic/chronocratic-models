__all__ = ['get_mcl_model']

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn

from tscollection.models.convolutional.standard.mcl.losses import MixUpLoss


class FCN(pl.LightningModule):
    def __init__(
        self,
        n_in: int,
        output_dims: int = 320,
        batch_size: int = 8,
        device: str = 'cuda',
        alpha: float = 1.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._device = device
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.criterion = MixUpLoss(device=self._device, batch_size=batch_size)

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_dims),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(output_dims, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128)
        )

        self.output_representation = False  # Control flag

    def switch_to_representation_mode(self) -> None:
        self.output_representation = True  # Switch to representation mode

    def switch_to_training_mode(self) -> None:
        self.output_representation = False  # Switch back to training mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out = self.proj_head(h)

        if self.output_representation:
            return h.unsqueeze(1)
        return out

    def _step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch

        x_1 = x
        x_2 = x[torch.randperm(len(x))]

        lam = np.random.beta(self.alpha, self.alpha)

        x_aug = lam * x_1 + (1 - lam) * x_2

        z_1 = self(x_1)
        z_2 = self(x_2)
        z_aug = self(x_aug)

        loss = self.criterion(z_aug, z_1, z_2, lam)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, batch_idx)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def get_mcl_model(model_params: dict) -> FCN:
    """Function to create an MCL model instance based on the provided parameters."""
    return FCN(**model_params)
