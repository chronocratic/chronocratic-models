"""Supervised-training wrapper and reusable head for downstream tasks.

Provides a single :class:`SupervisedModule` (LightningModule) that owns
the train/val loop, optimizer, logging, and freeze logic. Model-specific
concerns are injected via a :class:`BatchAdapter`, a representation
function, and an :class:`nn.Module` head.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['BatchAdapter', 'FlattenLinearHead', 'RepresentationBackbone', 'SupervisedModule']


@runtime_checkable
class RepresentationBackbone(Protocol):
    """A backbone that can report the flattened feature size of its representation.

    Implementations:
        - :class:`~chronocratic.models.transformer.tst.model.TST`
        - :class:`~chronocratic.models.convolutional.standard.series2vec.model.Series2Vec`
        - :class:`~chronocratic.models.convolutional.standard.tstcc.model.TSTCC`
    """

    @property
    def representation_dim(self) -> int:
        """Flattened feature size handed to a downstream head."""


class BatchAdapter(Protocol):
    """Strategy: decode a model-specific batch tuple into encoder inputs + targets.

    Each model has a different batch format from its DataLoader.
    The adapter normalizes it into ``((encoder_inputs, ...), targets)``
    so :class:`SupervisedModule` never sees model-specific tuple shapes.
    """

    def __call__(self, batch: tuple) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return ``((encoder_inputs, ...), targets)``."""


class FlattenLinearHead(nn.Module):
    """Flatten a representation across all non-batch dims, then a single linear layer.

    Reused by every model whose representation is a tensor of shape ``(B, ...)``.
    Series2Vec reps are already ``(B, 2*rep)`` so the flatten is a no-op there.

    Args:
        in_features: Flattened representation size (``backbone.representation_dim``).
        num_outputs: Number of classes (classification) or targets (regression).
    """

    def __init__(self, in_features: int, num_outputs: int) -> None:
        super().__init__()
        self._fc = nn.Linear(in_features, num_outputs)

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        """Compute logits from a representation tensor.

        Args:
            reps: Representation of shape ``(B, ...)``.

        Returns:
            Logits of shape ``(B, num_outputs)``.
        """
        return self._fc(reps.flatten(start_dim=1))


class SupervisedModule(pl.LightningModule):
    """Generic supervised-training wrapper for labeled downstream tasks.

    Trains a ``backbone`` + ``head`` on labels for classification or
    regression. Owns the train/val loop, optimizer, logging, and (static)
    freeze. Everything model-specific is injected.

    The four supervised modes are *configuration*, not subclasses::

        Mode                  backbone state   freeze_backbone   Trainer callback
        Linear probe          pretrained       True              none
        Full fine-tune        pretrained       False             none
        Gradual unfreeze      pretrained       False             BackboneUnfreeze
        Supervised (scratch)  fresh / random   False             none

    "Fine-tune" vs "supervised from scratch" is solely whether the injected
    backbone was pretrained — same class, same call. The from-scratch path
    (a freshly constructed, un-pretrained backbone with ``freeze_backbone=False``)
    replaces the old TS-TCC ``SUPERVISED`` training mode.

    Args:
        backbone: A (possibly pretrained) model exposing the representation fn used below.
        head: Maps a representation tensor to ``(B, num_outputs)``
            (e.g. :class:`FlattenLinearHead`).
        representation_fn: ``(backbone, *encoder_inputs) -> Tensor``. Differentiable. MUST
            NOT route through ``encode()`` (that path is inference-mode / offline only).
        batch_adapter: Decodes the batch tuple into ``((encoder_inputs, ...), targets)``.
        loss_fn: ``(predictions, targets) -> scalar``.
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        freeze_backbone: Freeze backbone params (linear probe). Set ``False`` for full
            fine-tuning, for supervised-from-scratch (fresh backbone), or when a
            gradual-unfreeze callback owns freezing (see :class:`BackboneUnfreeze`).
            Never have both the bool and a callback flip ``requires_grad``.
        sync_dist: Sync logged metrics across processes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        representation_fn: Callable[..., torch.Tensor],
        batch_adapter: BatchAdapter,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        freeze_backbone: bool = True,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['backbone', 'head', 'representation_fn', 'batch_adapter', 'loss_fn']
        )
        self._backbone = backbone
        self._head = head
        self._representation_fn = representation_fn
        self._batch_adapter = batch_adapter
        self._loss_fn = loss_fn
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sync_dist = sync_dist
        if freeze_backbone:
            self._backbone.requires_grad_(requires_grad=False)

    @property
    def backbone(self) -> nn.Module:
        """The wrapped backbone (read-only access for finetuning callbacks)."""
        return self._backbone

    def forward(self, *encoder_inputs: torch.Tensor) -> torch.Tensor:
        """Run representations through the head.

        Args:
            encoder_inputs: Model-specific tensors passed to ``representation_fn``.

        Returns:
            Predictions of shape ``(B, num_outputs)``.
        """
        reps = self._representation_fn(self._backbone, *encoder_inputs)
        return self._head(reps)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """Run one training or validation step.

        Args:
            batch: Raw batch tuple from the DataLoader.
            stage: ``'train'`` or ``'val'`` for the log key prefix.

        Returns:
            Scalar loss tensor.
        """
        encoder_inputs, targets = self._batch_adapter(batch)
        predictions = self(*encoder_inputs)
        loss = self._loss_fn(predictions, targets)
        self.log(
            f'{stage}_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        return loss

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch.

        Args:
            batch: Raw batch tuple from the DataLoader.
            _batch_idx: Index of this batch within the epoch (unused).

        Returns:
            Scalar training loss.
        """
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the validation loss for one batch.

        Args:
            batch: Raw batch tuple from the DataLoader.
            _batch_idx: Index of this batch within the epoch (unused).

        Returns:
            Scalar validation loss.
        """
        return self._shared_step(batch, stage='val')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam over the trainable parameters of this module.

        Uses a generator expression so frozen backbone params are excluded
        automatically. Compatible with gradual-unfreeze callbacks that add
        param groups later via :meth:`BaseFinetuning.unfreeze_and_add_param_group`.
        """
        trainable = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.Adam(trainable, lr=self._learning_rate, weight_decay=self._weight_decay)
