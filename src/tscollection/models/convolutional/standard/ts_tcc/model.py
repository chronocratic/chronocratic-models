from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ['TSTCC', 'get_ts_tcc_model']

from typing import cast, Literal

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.autotsrc.models.ts_tcc.encoder import TCCEncoder
from src.autotsrc.models.ts_tcc.losses import NTXentLoss
from src.autotsrc.models.ts_tcc.temporal_contrast import TemporalContrast

if TYPE_CHECKING:
    from lightning.pytorch.core.optimizer import LightningOptimizer
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

TrainingMode = Literal['self_supervised', 'supervised', 'fine_tuning']


class TSTCC(pl.LightningModule):
    """PyTorch Lightning module for TS-TCC.

    Three training modes controlled by ``training_mode``:

    - ``self_supervised``: temporal + contextual contrastive pre-training on
      augmented views; labels are ignored.
    - ``supervised``: standard cross-entropy training on labeled data.
    - ``fine_tuning``: cross-entropy training with only the logits head
      trainable; backbone weights are frozen.

    Batch format (all modes): ``(data, labels, aug1, aug2)``
    where ``aug1`` / ``aug2`` are only used in ``self_supervised`` mode.

    Uses ``automatic_optimization = False`` because two separate optimizers
    (one per sub-module) must be stepped independently.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        stride: int,
        final_out_channels: int,
        features_len: int,
        num_classes: int,
        dropout: float = 0.35,
        tc_hidden_dim: int = 100,
        tc_timesteps: int = 6,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
        training_mode: TrainingMode = 'self_supervised',
        learning_rate: float = 3e-4,
        lambda1: float = 1.0,
        lambda2: float = 0.7,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self._training_mode = training_mode
        self._learning_rate = learning_rate
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._sync_dist = sync_dist

        self._encoder = TCCEncoder(
            input_channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
            final_out_channels=final_out_channels,
            features_len=features_len,
            num_classes=num_classes,
            dropout=dropout,
        )
        self._tc_model = TemporalContrast(
            num_channels=final_out_channels, hidden_dim=tc_hidden_dim, timesteps=tc_timesteps
        )
        self._nt_xent_loss = NTXentLoss(
            temperature=temperature, use_cosine_similarity=use_cosine_similarity
        )
        self._criterion = nn.CrossEntropyLoss()

        if training_mode == 'fine_tuning':
            for name, param in self._encoder.named_parameters():
                param.requires_grad = name.startswith('logits')

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder. Returns ``(logits, features)``."""
        return self._encoder(x)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        data, labels, aug1, aug2 = batch

        if self._training_mode == 'self_supervised':
            aug1 = aug1.float()
            aug2 = aug2.float()
            _, features1 = self._encoder(aug1)
            _, features2 = self._encoder(aug2)
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_loss1, proj1 = self._tc_model(features1, features2)
            temp_loss2, proj2 = self._tc_model(features2, features1)

            temporal_loss = temp_loss1 + temp_loss2
            contextual_loss = self._nt_xent_loss(proj1, proj2)
            return self._lambda1 * temporal_loss + self._lambda2 * contextual_loss

        # supervised / fine_tuning
        predictions, _ = self._encoder(data.float())
        return self._criterion(predictions, labels.long())

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Manual optimization step for both sub-module optimizers."""
        model_opt, tc_opt = cast('list[LightningOptimizer]', self.optimizers())
        model_opt.zero_grad()
        tc_opt.zero_grad()

        loss = self._compute_loss(batch)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.manual_backward(loss)
        model_opt.step()
        tc_opt.step()
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute and log validation loss."""
        loss = self._compute_loss(batch)
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return one Adam optimizer per sub-module (encoder and TC model)."""
        return [
            torch.optim.Adam(self._encoder.parameters(), lr=self._learning_rate),
            torch.optim.Adam(self._tc_model.parameters(), lr=self._learning_rate),
        ]

    # ------------------------------------------------------------------
    # Representation extraction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, data: torch.Tensor, batch_size: int, num_workers: int = 0) -> torch.Tensor:
        """Extract feature maps for ``data`` of shape ``(N, C, T)``.

        Returns ``(N, final_out_channels, reduced_T)``: the Conv encoder output
        before the logits layer, suitable for downstream tasks.
        """
        was_training = self._encoder.training
        self._encoder.eval()

        loader = DataLoader(
            TensorDataset(data), batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        outputs = []
        for (batch_x,) in loader:
            _, features = self._encoder(batch_x.float().to(self.device))
            outputs.append(features.cpu())

        self._encoder.train(was_training)
        return torch.cat(outputs, dim=0)


def get_ts_tcc_model(model_params: dict) -> TSTCC:
    """Create a TSTCC instance from a parameter dictionary."""
    return TSTCC(**model_params)
