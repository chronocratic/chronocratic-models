__all__ = ["MCL"]

import lightning.pytorch as pl
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.mcl.encoder import FCNEncoder
from chronocratic.models.convolutional.standard.mcl.losses import MixUpLoss
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.utils import extract_features_from_batch, zero_fill_padding
from chronocratic.models.utils.helpers import _warn_sequence_fallback


class MCL(pl.LightningModule, BasicEncodingMixin):
    """FCN-based encoder for Mixup Contrastive Learning (MCL).

    This model was implemented based on the code available on this GitHub
    repo https://github.com/Wickstrom/MixupContrastiveLearning.

    Args:
        input_dim: Number of input feature channels.
        representation_dim: Dimension of the flat encoder output.
        alpha: Beta-distribution parameter for MixUp interpolation.
        learning_rate: Base learning rate for the Adam optimizer.
        encoder_channels: Tuple of channel counts for each Conv1d block.
        encoder_kernels: Tuple of kernel sizes for each Conv1d block.
        encoder_dilations: Tuple of dilation rates for each Conv1d block.
        projection_dim: Hidden dimension of the projection head.
        sync_dist: Whether to synchronize metrics across processes.
        normalization_layer_type: Normalization strategy for encoder and
            projection head. Use ``CHANNEL`` for GroupNorm (batch_size=1
            safe) or ``BATCH`` for BatchNorm1d. Defaults to ``CHANNEL``.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        *,
        input_dim: int,
        representation_dim: int = 128,
        alpha: float = 1.0,
        learning_rate: float = 1e-3,
        encoder_channels: tuple[int, ...] = (128, 256, 128),
        encoder_kernels: tuple[int, ...] = (7, 5, 3),
        encoder_dilations: tuple[int, ...] = (2, 4, 8),
        projection_dim: int = 128,
        sync_dist: bool = False,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._input_dim = input_dim
        self._representation_dim = representation_dim
        self._projection_dim = projection_dim
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._sync_dist = sync_dist

        self.criterion = MixUpLoss()

        self._encoder = FCNEncoder(
            input_dim=input_dim,
            representation_dim=representation_dim,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            encoder_dilations=encoder_dilations,
            normalization_layer_type=normalization_layer_type,
        )
        proj_norm = (
            nn.GroupNorm(num_groups=1, num_channels=projection_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(projection_dim)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(representation_dim, projection_dim),
            proj_norm,
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    @property
    def representation_dim(self) -> int:
        """Return the feature dim of encode()'s output."""
        return self._representation_dim

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
        if output not in type(self).supported_outputs:
            msg = f"MCL does not support output={output}; supported: {type(self).supported_outputs}"
            raise ValueError(msg)
        flat = encoder(batch_x)  # (B, D) - D=representation_dim
        if output == EncodingOutputShape.VECTOR:
            return flat  # (B, D) — VECTOR
        _warn_sequence_fallback(type(self))
        return flat.unsqueeze(1)  # (B, 1, D) — SEQUENCE (fake temporal axis)

    def _step(self, batch: torch.Tensor) -> torch.Tensor:
        """Run one contrastive training step.

        Note:
            At ``batch_size=1``, ``randperm(1)`` returns identity, so
            MixUp interpolation collapses to ``x_aug = x_1`` and
            ``z_1 == z_2 == z_aug``, producing trivially near-zero loss.
            Use ``batch_size >= 2`` for meaningful contrastive training.
        """
        x = extract_features_from_batch(batch)
        x, _ = zero_fill_padding(x)

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
