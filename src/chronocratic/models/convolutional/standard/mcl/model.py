__all__ = ["MCL"]

import lightning.pytorch as pl
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder
from chronocratic.models.convolutional.standard.mcl.losses import MixUpLoss
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.utils import extract_features_from_batch
from chronocratic.models.utils.helpers import _warn_sequence_fallback


class MCL(pl.LightningModule, BasicEncodingMixin):
    """FCN-based encoder for Mixup Contrastive Learning (MCL).

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Wickstrom/MixupContrastiveLearning.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR}
    )

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 128,
        alpha: float = 1.0,
        learning_rate: float = 1e-3,
        encoder_channels: tuple[int, ...] = (128, 256, 128),
        encoder_kernels: tuple[int, ...] = (7, 5, 3),
        encoder_dilations: tuple[int, ...] = (2, 4, 8),
        projection_dims: int = 128,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._sync_dist = sync_dist

        self.criterion = MixUpLoss()

        self._encoder = FCNEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            encoder_dilations=encoder_dilations,
        )
        self.proj_head = nn.Sequential(
            nn.Linear(output_dims, projection_dims),
            nn.BatchNorm1d(projection_dims),
            nn.ReLU(),
            nn.Linear(projection_dims, projection_dims),
        )

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder."""
        return self._encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected MCL representations for ``x``."""
        return self.proj_head(self._encoder(x))

    def _get_encoder(self) -> nn.Module:
        """Expose the encoder (before the MixUp projection head)."""
        return self.encoder

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Return flat representation for VECTOR, unsqueeze for SEQUENCE."""
        flat = encoder(batch_x)  # (B, D) - D=latent_dim
        if output == EncodingOutputShape.VECTOR:
            return flat  # (B, D) — VECTOR
        if output == EncodingOutputShape.SEQUENCE:
            _warn_sequence_fallback(type(self))
            return flat.unsqueeze(1)  # (B, 1, D) — SEQUENCE (fake temporal axis)
        msg = f"MCL does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    def _step(self, batch: torch.Tensor) -> torch.Tensor:
        x = extract_features_from_batch(batch)

        x_1 = x
        x_2 = x[torch.randperm(len(x))]  # device-ok: CPU permutation index

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

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute and log the validation loss for one batch."""
        with torch.no_grad():
            loss = self._step(batch)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the Adam optimizer used to train MCL."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
