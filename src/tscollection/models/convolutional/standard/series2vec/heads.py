"""Downstream classification head for a pretrained Series2Vec.

The head is a standalone LightningModule that wraps a (possibly
pretrained) :class:`Series2Vec` backbone, attaches a final
``nn.Linear`` over the ``(B, 2 * representation_dims)`` representation,
and supplies its own loss and training loop. The default is to freeze
the backbone and train only the head's output layer (linear-probe
pattern); set ``freeze_backbone=False`` for full fine-tuning.

Expects batches of the form ``(X, targets)``.
"""

from __future__ import annotations

__all__ = ['Series2VecClassificationHead']

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

if TYPE_CHECKING:
    from tscollection.models.convolutional.standard.series2vec.model import Series2Vec


class Series2VecClassificationHead(pl.LightningModule):
    """Classification head on top of a Series2Vec backbone.

    Args:
        backbone: A (possibly pretrained) :class:`Series2Vec` instance.
        num_classes: Number of target classes.
        freeze_backbone: Freeze backbone weights (linear-probe). Default ``True``.
        learning_rate: Learning rate for Adam.
        weight_decay: L2 weight decay applied via the optimizer.
        sync_dist: Whether to synchronize logged metrics across processes.
    """

    def __init__(
        self,
        backbone: Series2Vec,
        num_classes: int,
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

        representation_dims = backbone.hparams['representation_dims']
        # Series2VecNetwork.encode concatenates temporal + frequency reps.
        self._output_layer = nn.Linear(2 * representation_dims, num_classes)
        self._loss_fn = nn.CrossEntropyLoss()

        if freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, num_classes)`` classification logits."""
        representations = self._backbone.network.encode(x)
        return self._output_layer(representations)

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(predictions, targets.long().squeeze())

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
        x, targets = batch
        predictions = self(x)
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
        x, targets = batch
        predictions = self(x)
        loss = self._compute_loss(predictions, targets)
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam over the trainable parameters of this module."""
        trainable = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(trainable, lr=self._learning_rate, weight_decay=self._weight_decay)
