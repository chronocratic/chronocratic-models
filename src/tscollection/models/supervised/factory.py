"""Factory constructors for :class:`SupervisedModule`.

Each factory wires the correct :class:`FlattenLinearHead`, batch adapter,
representation function, and loss function for a given backbone — so
callers don't hand-assemble four collaborators.

TS-TCC uses a fresh :class:`FlattenLinearHead` (not encoder
logits reuse). All three factories follow the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tscollection.models.supervised._adapters import (
    series2vec_representations,
    supervised_batch_adapter,
    tst_batch_adapter,
    tst_representations,
    tstcc_representations,
)
from tscollection.models.supervised._utils import classification_loss, regression_loss
from tscollection.models.supervised.supervised import FlattenLinearHead, SupervisedModule

if TYPE_CHECKING:
    from tscollection.models.convolutional.standard.series2vec.model import Series2Vec
    from tscollection.models.convolutional.standard.tstcc.model import TSTCC
    from tscollection.models.transformer.tst.model import TST

__all__ = ['make_series2vec_supervised', 'make_tst_supervised', 'make_tstcc_supervised']

_VALID_TASKS = ('classification', 'regression')


def _validate_task(task: str) -> None:
    """Raise if *task* is not one of the recognized values."""
    if task not in _VALID_TASKS:
        msg = f"task must be 'classification' or 'regression', got '{task}'"
        raise ValueError(msg)


def make_tst_supervised(
    backbone: TST,
    *,
    num_outputs: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> SupervisedModule:
    """Build a :class:`SupervisedModule` for a TST backbone.

    Args:
        backbone: A :class:`TST` instance with ``representation_dim``.
        num_outputs: Number of classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`SupervisedModule` ready for training.

    Example:
        Linear probe on a pretrained backbone (default)::

            module = make_tst_supervised(pretrained_tst, num_outputs=5)

        Supervised from scratch — fresh backbone, train end-to-end::

            module = make_tst_supervised(
                TST(...), num_outputs=5, freeze_backbone=False
            )
    """
    _validate_task(task)
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_outputs)
    loss_fn = classification_loss if task == 'classification' else regression_loss
    return SupervisedModule(
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


def make_series2vec_supervised(
    backbone: Series2Vec,
    *,
    num_outputs: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> SupervisedModule:
    """Build a :class:`SupervisedModule` for a Series2Vec backbone.

    Args:
        backbone: A :class:`Series2Vec` instance with ``representation_dim``.
        num_outputs: Number of classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`SupervisedModule` ready for training.

    Example:
        Regression from scratch on a fresh backbone::

            module = make_series2vec_supervised(
                Series2Vec(...), num_outputs=1, task='regression', freeze_backbone=False
            )
    """
    _validate_task(task)
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_outputs)
    loss_fn = classification_loss if task == 'classification' else regression_loss
    return SupervisedModule(
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


def make_tstcc_supervised(
    backbone: TSTCC,
    *,
    num_outputs: int,
    task: str = 'classification',
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    sync_dist: bool = False,
) -> SupervisedModule:
    """Build a :class:`SupervisedModule` for a TS-TCC backbone.

    Uses a fresh :class:`FlattenLinearHead` (not encoder logits reuse).
    With ``freeze_backbone=False`` on a fresh (un-pretrained) :class:`TSTCC`, this
    is the explicit replacement for the removed ``TSTCCTrainingMode.SUPERVISED``.

    Args:
        backbone: A :class:`TSTCC` instance with ``representation_dim``.
        num_outputs: Number of classes (classification) or targets (regression).
        task: ``'classification'`` or ``'regression'``.
        freeze_backbone: Freeze backbone params at construction (linear probe).
        learning_rate: Adam LR.
        weight_decay: Adam weight decay.
        sync_dist: Sync logged metrics across processes.

    Returns:
        Configured :class:`SupervisedModule` ready for training.

    Example:
        Supervised from scratch (old ``SUPERVISED`` mode)::

            module = make_tstcc_supervised(
                TSTCC(...), num_outputs=6, freeze_backbone=False
            )
    """
    _validate_task(task)
    head = FlattenLinearHead(in_features=backbone.representation_dim, num_outputs=num_outputs)
    loss_fn = classification_loss if task == 'classification' else regression_loss
    return SupervisedModule(
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
