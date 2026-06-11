__all__ = ['TSTCC']

from typing import cast, TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional

from tscollection.models._mixin import BasicEncodingMixin
from tscollection.models.convolutional.standard.tstcc.encoder import TCCEncoder
from tscollection.models.convolutional.standard.tstcc.enums import TSTCCTrainingMode
from tscollection.models.convolutional.standard.tstcc.losses import NTXentLoss
from tscollection.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast
from tscollection.models.utils import extract_features_from_batch

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from tscollection.models.augmentation.dual import DualAugmentation


class TSTCC(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for TS-TCC.

    Three training modes controlled by ``training_mode``:

    - ``self_supervised``: temporal + contextual contrastive pre-training on
      augmented views; labels are ignored.
    - ``supervised``: standard cross-entropy training on labeled data.
    - ``fine_tuning``: cross-entropy training with only the logits head
      trainable; backbone weights are frozen.

    Batch format: ``(data, labels)``. In ``self_supervised`` mode, two
    augmented views of ``data`` are produced by the injected
    ``DualAugmentation`` (one augmentation per view). The default is
    ``TSTCCDualAugmentation``, which provides Gaussian scaling (weak)
    and segment-permutation + jitter (strong) views, matching the
    original TS-TCC contract.

    Uses ``automatic_optimization = False`` because two separate optimizers
    (one per sub-module) must be stepped independently.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/emadeldeen24/TS-TCC under MIT License.
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
        *,
        use_cosine_similarity: bool = True,
        training_mode: TSTCCTrainingMode = TSTCCTrainingMode.SELF_SUPERVISED,
        learning_rate: float = 3e-4,
        lambda1: float = 1.0,
        lambda2: float = 0.7,
        sync_dist: bool = False,
        augmentation: 'DualAugmentation | None' = None,
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
            from tscollection.models.convolutional.standard.tstcc.augmentations import (  # noqa: PLC0415
                TSTCCDualAugmentation,
            )

            self._augmentation = TSTCCDualAugmentation()
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

        if training_mode == TSTCCTrainingMode.FINE_TUNING:
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

    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        data = extract_features_from_batch(batch).float()

        if self._training_mode == TSTCCTrainingMode.SELF_SUPERVISED:
            views = self._augmentation.augment(data)
            aug1, aug2 = views.views[0], views.views[1]
            _, features1 = self._encoder(aug1)
            _, features2 = self._encoder(aug2)
            features1 = functional.normalize(features1, dim=1)
            features2 = functional.normalize(features2, dim=1)

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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        """Manual optimization step for both sub-module optimizers."""
        optimizers = cast('list[torch.optim.Optimizer]', self.optimizers(use_pl_optimizer=False))
        model_opt, tc_opt = optimizers
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

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        """Compute and log validation loss."""
        with torch.no_grad():
            loss = self._compute_loss(batch)
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> 'OptimizerLRScheduler':
        """Return one Adam optimizer per sub-module (encoder and TC model)."""
        return [
            torch.optim.Adam(self._encoder.parameters(), lr=self._learning_rate),
            torch.optim.Adam(self._tc_model.parameters(), lr=self._learning_rate),
        ]

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> nn.Module:
        """Expose the conv encoder to ``BasicEncodingMixin.encode``."""
        return self._encoder

    def _prepare_inputs(self, batch_x: torch.Tensor) -> tuple[torch.Tensor]:
        """Cast to float — the TCC encoder expects float inputs."""
        return (batch_x.float(),)

    def _postprocess(self, output: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the pre-logits features from the ``(logits, features)`` encoder output."""
        return output[1]
