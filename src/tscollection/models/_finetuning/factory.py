"""Factory constructors for :class:`FineTuningModule`.

Each factory wires the correct :class:`FlattenLinearHead`, batch adapter,
representation function, and loss function for a given backbone — so
callers don't hand-assemble four collaborators.

Per D-01, TS-TCC uses a fresh :class:`FlattenLinearHead` (not encoder
logits reuse). All three factories follow the same pattern.
"""

from __future__ import annotations

import torch
from torch import nn

from tscollection.models._finetuning.adapters import (
    classification_loss,
    series2vec_representations,
    supervised_batch_adapter,
    tst_batch_adapter,
    tst_representations,
    tstcc_representations,
)
from tscollection.models._finetuning.finetuning import FineTuningModule, FlattenLinearHead

__all__ = ['make_series2vec_finetuner', 'make_tst_finetuner', 'make_tstcc_finetuner']


def make_tst_finetuner(
    backbone: FineTuningModule,  # type: ignore[type-arg]
    *,
    num_outputs: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> FineTuningModule:
    """Build a :class:`FineTuningModule` for a TST backbone.

    Args:
        backbone: A :class:`TST` instance with ``representation_dim``.
        num_outputs: Number of classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`FineTuningModule` ready for training.
    """
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_outputs)
    loss_fn = classification_loss if task == 'classification' else _mse_loss
    return FineTuningModule(
        backbone=backbone,
        head=head,
        representation_fn=tst_representations,
        batch_adapter=tst_batch_adapter,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        sync_dist=sync_dist,
    )


def make_series2vec_finetuner(
    backbone: FineTuningModule,  # type: ignore[type-arg]
    *,
    num_outputs: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> FineTuningModule:
    """Build a :class:`FineTuningModule` for a Series2Vec backbone.

    Args:
        backbone: A :class:`Series2Vec` instance with ``representation_dim``.
        num_outputs: Number of classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`FineTuningModule` ready for training.
    """
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_outputs)
    loss_fn = classification_loss if task == 'classification' else _mse_loss
    return FineTuningModule(
        backbone=backbone,
        head=head,
        representation_fn=series2vec_representations,
        batch_adapter=supervised_batch_adapter,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        sync_dist=sync_dist,
    )


def make_tstcc_finetuner(
    backbone: FineTuningModule,  # type: ignore[type-arg]
    *,
    num_classes: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> FineTuningModule:
    """Build a :class:`FineTuningModule` for a TS-TCC backbone.

    Per D-01, uses a fresh :class:`FlattenLinearHead` (not encoder logits reuse).

    Args:
        backbone: A :class:`TSTCC` instance with ``representation_dim``.
        num_classes: Number of output classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`FineTuningModule` ready for training.
    """
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_classes)
    loss_fn = classification_loss if task == 'classification' else _mse_loss
    return FineTuningModule(
        backbone=backbone,
        head=head,
        representation_fn=tstcc_representations,
        batch_adapter=supervised_batch_adapter,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        sync_dist=sync_dist,
    )


# -- private helpers -----------------------------------------------------------


def _mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss for regression tasks."""
    return nn.MSELoss()(predictions, targets)
