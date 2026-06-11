from __future__ import annotations

__all__ = ['TST']

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

from tscollection.models._mixin import BasicEncodingMixin
from tscollection.models.transformer.tst.loss import MaskedMSELoss
from tscollection.models.transformer.tst.ts_transformer import TSTransformerEncoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


class TST(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for the Time Series Transformer (TST).

    Representation-learning model trained with a masked-reconstruction
    pretraining objective. The same model supports both random-mask
    imputation and structured-mask transduction pretraining — the
    masking strategy is configured upstream in the dataloader and is
    transparent to the model.

    Batch format expected from the DataLoader:
        ``(X, targets, target_masks, padding_masks, IDs)``
    where ``target_masks`` marks the positions whose reconstruction is
    scored, and ``padding_masks`` marks valid (non-padded) timesteps.

    ``forward(x, padding_masks)`` returns transformer representations
    of shape ``(batch, seq_len, d_model)``, not the masked-reconstruction
    output. The reconstruction head is internal and used only during
    pretraining.

    For downstream classification / regression, use :class:`FineTuningModule`
    from ``tscollection.models._finetuning``.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/gzerveas/mvts_transformer under MIT License.
    """

    def __init__(
        self,
        feat_dim: int,
        max_seq_len: int,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pos_encoding: str = 'fixed',
        activation: str = 'gelu',
        norm: str = 'BatchNorm',
        *,
        freeze: bool = False,
        learning_rate: float = 1e-3,
        lr_step: list[int] | None = None,
        lr_factor: float = 0.1,
        l2_reg: float = 0.0,
        global_reg: bool = False,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._l2_reg = l2_reg
        self._global_reg = global_reg
        self._learning_rate = learning_rate
        self._lr_step = lr_step or [1_000_000]
        self._lr_factor = lr_factor
        self._sync_dist = sync_dist

        self._encoder = TSTransformerEncoder(
            feat_dim=feat_dim,
            max_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding=pos_encoding,
            activation=activation,
            norm=norm,
            freeze=freeze,
        )
        self._loss_fn: nn.Module = MaskedMSELoss(reduction='none')

        if freeze:
            for name, param in self._encoder.named_parameters():
                param.requires_grad = name.startswith('output_layer')

    # ------------------------------------------------------------------
    # Forward / representation extraction
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Return transformer representations of shape ``(batch, seq_len, d_model)``."""
        return self.get_representations(x, padding_masks)

    def get_representations(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Run the transformer trunk, skipping the reconstruction output layer."""
        return self._encoder.encode_representations(x, padding_masks)

    def reconstruct(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Run the full backbone, including the reconstruction output layer.

        Used during masked-reconstruction pretraining; downstream callers
        should use ``forward`` / ``get_representations`` instead.
        """
        return self._encoder(x, padding_masks)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        x, targets, target_masks, padding_masks, _ = batch
        predictions = self.reconstruct(x, padding_masks)
        combined_mask = target_masks * padding_masks.unsqueeze(-1)
        per_element_loss = self._loss_fn(predictions, targets, combined_mask)

        mean_loss = torch.sum(per_element_loss) / len(per_element_loss)

        # output-layer-only L2 (global L2 is handled via weight_decay in the optimizer)
        if self.training and self._l2_reg and not self._global_reg:
            for name, param in self._encoder.named_parameters():
                if name == 'output_layer.weight':
                    mean_loss = mean_loss + self._l2_reg * torch.sum(torch.square(param))

        return mean_loss

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the masked-reconstruction training loss for one batch."""
        loss = self._compute_loss(batch)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        return loss

    def validation_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the masked-reconstruction validation loss for one batch."""
        loss = self._compute_loss(batch)
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    # ------------------------------------------------------------------
    # Gradient clipping (original used max_norm=4.0)
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients by global norm to stabilise training."""
        del optimizer, gradient_clip_algorithm
        if gradient_clip_val is None:
            gradient_clip_val = 4.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_val)

    # ------------------------------------------------------------------
    # Optimizers & LR scheduling
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Return Adam optimizer with MultiStepLR scheduler."""
        weight_decay = self._l2_reg if self._global_reg else 0.0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self._lr_step, gamma=self._lr_factor
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> Callable[..., torch.Tensor]:
        """Expose representation extraction to ``BasicEncodingMixin.encode``."""
        return self.get_representations

    def _get_encoder_module(self) -> nn.Module:
        """Underlying module for state management — ``get_representations`` is a bound method."""
        return self._encoder

    def _prepare_inputs(self, batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize all-true padding masks; ``encode()`` carries no mask info."""
        padding_masks = torch.ones(
            batch_x.shape[0], batch_x.shape[1], dtype=torch.bool, device=self.device
        )
        return (batch_x, padding_masks)

    @property
    def representation_dim(self) -> int:
        """Flattened representation size handed to a downstream head.

        Returns:
            ``d_model * max_len`` — the number of features after flattening
            the ``(batch, seq_len, d_model)`` representation.
        """
        return self._encoder.d_model * self._encoder.max_len
