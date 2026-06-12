"""Centralized helpers for trainable augmentation producers.

These functions are the **only** place in the codebase that branch on
``isinstance(..., TrainableAugmentationProducer)``. Models should
call these helpers instead of checking the type themselves, keeping the
model code branchless on the augmentation type.

Exported symbols:
    - ``maybe_train_augmentation``
    - ``maybe_configure_augmentation_optimizer``
"""

from __future__ import annotations

__all__ = [
    'maybe_configure_augmentation_optimizer',
    'maybe_train_augmentation',
]

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.optim import Optimizer

from tscollection.models.augmentation.base import (
    AugmentationProducer,
    TrainableAugmentationProducer,
)


def maybe_train_augmentation(
    augmentation: AugmentationProducer[Any],
    *,
    x: torch.Tensor,
    encoder: nn.Module,
    epoch: int,
    batch_idx: int,
) -> torch.Tensor | None:
    """Run one augmentation-network training step if the producer is trainable.

    For stateless producers this function returns ``None`` immediately.
    For trainable producers it checks ``should_train_augmentation()``
    and delegates to ``train_step()`` when the strategy permits.

    Mode management: sets ``encoder`` to eval and ``augmentation`` to train
    during the forward pass, then restores augmentation to eval. The caller
    is responsible for setting encoder back to train for Phase 2.

    This is the sole code path in the codebase that uses
    ``isinstance(..., TrainableAugmentationProducer)`` for the training
    loop.

    Args:
        augmentation: The augmentation producer to optionally train.
        x: Original input data.
        encoder: The main encoder module to compute embeddings.
        epoch: Current training epoch.
        batch_idx: Current batch index within the epoch.

    Returns:
        Loss tensor if the producer is trainable and should train this
        step, otherwise ``None``.
    """
    if not isinstance(augmentation, TrainableAugmentationProducer):
        return None
    if not augmentation.should_train_augmentation(epoch=epoch, batch_idx=batch_idx):
        return None
    encoder.eval()
    augmentation.train()
    loss = augmentation.train_step(x=x, encoder=encoder, batch_idx=batch_idx)
    augmentation.eval()
    return loss


def maybe_configure_augmentation_optimizer(
    augmentation: AugmentationProducer[Any],
    *,
    lr: float,
) -> Optimizer | None:
    """Configure an optimizer for the augmentation network if trainable.

    For stateless producers this function returns ``None`` immediately.
    For trainable producers it delegates to ``configure_optimizer()``.

    This is the sole code path in the codebase that uses
    ``isinstance(..., TrainableAugmentationProducer)`` for optimizer
    configuration.

    Args:
        augmentation: The augmentation producer to optionally configure.
        lr: Learning rate for the augmentation network optimizer.

    Returns:
        An optimizer for the augmentation network parameters, or ``None``
        if the producer is not trainable.
    """
    if not isinstance(augmentation, TrainableAugmentationProducer):
        return None
    return augmentation.configure_optimizer(lr=lr)
