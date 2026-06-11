__all__ = ['FCN']

import lightning.pytorch as pl
import torch
from torch import nn

from tscollection.models._mixin import BasicEncodingMixin
from tscollection.models.convolutional.standard.mcl.encoder import FCNEncoder
from tscollection.models.convolutional.standard.mcl.losses import MixUpLoss


class FCN(pl.LightningModule, BasicEncodingMixin):
    """FCN encoder for Mixup Contrastive Learning (MCL).

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Wickstrom/MixupContrastiveLearning.
    """

    def __init__(
        self,
        n_in: int,
        output_dims: int = 320,
        alpha: float = 1.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._alpha = alpha
        self._learning_rate = learning_rate

        self.criterion = MixUpLoss()

        self.encoder = FCNEncoder(input_channels=n_in, output_dims=output_dims)
        self.proj_head = nn.Sequential(
            nn.Linear(output_dims, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected MCL representations for ``x``."""
        return self.proj_head(self.encoder(x))

    def _get_encoder(self) -> nn.Module:
        """Expose the FCN encoder (before the MixUp projection head)."""
        return self.encoder

    def _postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """Add a trailing singleton dim so the shape matches the flag-pattern convention."""
        return output.unsqueeze(1)

    def _step(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch

        x_1 = x
        x_2 = x[torch.randperm(len(x))]

        concentration = torch.tensor(self._alpha, device=x.device)
        lam = torch.distributions.Beta(concentration, concentration).sample()

        x_aug = lam * x_1 + (1 - lam) * x_2

        z_1 = self(x_1)
        z_2 = self(x_2)
        z_aug = self(x_aug)

        loss = self.criterion(z_aug, z_1, z_2, lam)

        return loss

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
        loss = self._step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the validation loss for one batch."""
        loss = self._step(batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the Adam optimizer used to train MCL."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
