"""Downstream classification and regression heads for a pretrained TST.

Each head is a standalone LightningModule that wraps a (possibly
pretrained) :class:`TST` backbone, attaches a final ``nn.Linear`` over
the flattened representation, and supplies its own loss and training
loop. The default is to freeze the backbone and train only the head's
output layer (linear-probe pattern); set ``freeze_backbone=False`` for
full fine-tuning.

Both heads expect batches of the form ``(X, targets, padding_masks, IDs)``.
"""

from __future__ import annotations

__all__ = ['TSTClassificationHead', 'TSTRegressionHead']

from abc import abstractmethod
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

if TYPE_CHECKING:
    from tscollection.models.transformer.tst.model import TST


class _TSTHead(pl.LightningModule):
    """Shared base for downstream TST heads.

    Subclasses provide the loss and target preparation by overriding
    :meth:`_compute_loss`.
    """

    def __init__(
        self,
        backbone: TST,
        num_outputs: int,
        *,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])

        self._backbone = backbone
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sync_dist = sync_dist

        d_model = backbone.hparams['d_model']
        max_seq_len = backbone.hparams['max_seq_len']
        self._output_layer = nn.Linear(d_model * max_seq_len, num_outputs)

        if freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` through the backbone and return ``(batch, num_outputs)`` logits."""
        representations = self._backbone.get_representations(x, padding_masks)
        # Zero out padding embeddings, then flatten across time and feature dims.
        representations = representations * padding_masks.unsqueeze(-1)
        flat = representations.reshape(representations.shape[0], -1)
        return self._output_layer(flat)

    @abstractmethod
    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return the scalar loss for this head."""

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
        x, targets, padding_masks, _ = batch
        predictions = self(x, padding_masks)
        loss = self._compute_loss(predictions, targets)
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
        """Compute and log the validation loss for one batch."""
        x, targets, padding_masks, _ = batch
        predictions = self(x, padding_masks)
        loss = self._compute_loss(predictions, targets)
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam over the trainable parameters of this module."""
        trainable = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(
            trainable, lr=self._learning_rate, weight_decay=self._weight_decay
        )


class TSTClassificationHead(_TSTHead):
    """Classification head on top of a TST backbone.

    Args:
        backbone: A (possibly pretrained) :class:`TST` instance.
        num_classes: Number of target classes.
        freeze_backbone: Freeze backbone weights (linear-probe). Default ``True``.
        learning_rate: Learning rate for Adam.
        weight_decay: L2 weight decay applied via the optimizer.
        sync_dist: Whether to synchronize logged metrics across processes.
    """

    def __init__(
        self,
        backbone: TST,
        num_classes: int,
        *,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        sync_dist: bool = False,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_outputs=num_classes,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sync_dist=sync_dist,
        )
        self._loss_fn = nn.CrossEntropyLoss()

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_fn(predictions, targets.long().squeeze())


class TSTRegressionHead(_TSTHead):
    """Regression head on top of a TST backbone.

    Args:
        backbone: A (possibly pretrained) :class:`TST` instance.
        num_outputs: Number of regression targets per sample.
        freeze_backbone: Freeze backbone weights (linear-probe). Default ``True``.
        learning_rate: Learning rate for Adam.
        weight_decay: L2 weight decay applied via the optimizer.
        sync_dist: Whether to synchronize logged metrics across processes.
    """

    def __init__(
        self,
        backbone: TST,
        num_outputs: int,
        *,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        sync_dist: bool = False,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_outputs=num_outputs,
            freeze_backbone=freeze_backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sync_dist=sync_dist,
        )
        self._loss_fn = nn.MSELoss()

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_fn(predictions, targets)
