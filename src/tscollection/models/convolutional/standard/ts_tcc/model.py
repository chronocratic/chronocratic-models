from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ['TSTCC']


import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F

from tscollection.models._mixin import SimpleEncodingMixin
from tscollection.models.convolutional.standard.ts_tcc.encoder import TCCEncoder
from tscollection.models.convolutional.standard.ts_tcc.enums import TrainingMode
from tscollection.models.convolutional.standard.ts_tcc.losses import NTXentLoss
from tscollection.models.convolutional.standard.ts_tcc.temporal_contrast import TemporalContrast
from tscollection.models.utils import extract_features_from_batch

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from tscollection.models.augmentation.base import AugmentationMethod


class TSTCC(pl.LightningModule, SimpleEncodingMixin):
    """PyTorch Lightning module for TS-TCC.

    Three training modes controlled by ``training_mode``:

    - ``self_supervised``: temporal + contextual contrastive pre-training on
      augmented views; labels are ignored.
    - ``supervised``: standard cross-entropy training on labeled data.
    - ``fine_tuning``: cross-entropy training with only the logits head
      trainable; backbone weights are frozen.

    Batch format: ``(data, labels)``. In ``self_supervised`` mode, two
    augmented views of ``data`` are produced by the injected
    ``AugmentationMethod``. The default is a ``PairedAugmentation`` of
    Gaussian scaling (weak view) and segment-permutation + jitter
    (strong view), matching the original TS-TCC contract.

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
        training_mode: TrainingMode = TrainingMode.SELF_SUPERVISED,
        learning_rate: float = 3e-4,
        lambda1: float = 1.0,
        lambda2: float = 0.7,
        sync_dist: bool = False,
        augmentation: AugmentationMethod | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['augmentation'])
        self.automatic_optimization = False

        self._training_mode = training_mode
        self._learning_rate = learning_rate
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._sync_dist = sync_dist

        if augmentation is None:
            from tscollection.models.augmentation import (  # noqa: PLC0415
                ComposeAugmentation,
                Jitter,
                JitterParameters,
                PairedAugmentation,
                Permutation,
                PermutationParameters,
                Scaling,
                ScalingParameters,
            )

            # Weak view: per-(sample, channel) Gaussian scaling around mean=2.
            # Strong view: random segment permutation followed by additive jitter.
            # Data flows as (B, C, T), hence channel_dim=1 and time_dim=-1.
            weak = Scaling(
                ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=1)
            )
            strong = ComposeAugmentation(
                [
                    Permutation(PermutationParameters(max_segments=5, time_dim=-1)),
                    Jitter(JitterParameters(sigma=0.8)),
                ]
            )
            self._augmentation: AugmentationMethod = PairedAugmentation(weak, strong)
        else:
            self._augmentation = augmentation

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

        if training_mode == TrainingMode.FINE_TUNING:
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
        data = extract_features_from_batch(batch).float()

        if self._training_mode == TrainingMode.SELF_SUPERVISED:
            views = self._augmentation.augment(data)
            aug1, aug2 = views.views[0], views.views[1]
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
        labels = batch[1]
        predictions, _ = self._encoder(data)
        return self._criterion(predictions, labels.long())

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Manual optimization step for both sub-module optimizers."""
        model_opt, tc_opt = self.optimizers()
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
    # Representation extraction (via SimpleEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _encode_batch(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Encode one batch — returns the conv-encoder features before the logits head.

        Input shape ``(batch, C, T)``; output ``(batch, final_out_channels, reduced_T)``.
        """
        _, features = self._encoder(batch_x.float().to(self.device))
        return features
