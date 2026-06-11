"""Gradual unfreeze callback for fine-tuning.

Provides :class:`BackboneUnfreeze`, a Lightning callback that freezes the
backbone at training start and unfreezes it at a target epoch with a
reduced learning rate (discriminative LR pattern).

Note:
    When this callback is used, construct :class:`FineTuningModule` with
    ``freeze_backbone=False`` so the callback is the sole owner of freeze
    state. Never let the module bool AND a callback both flip ``requires_grad``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import BaseFinetuning

if TYPE_CHECKING:
    import lightning.pytorch as pl
    import torch

__all__ = ['BackboneUnfreeze']


class BackboneUnfreeze(BaseFinetuning):
    """Freeze the backbone, then unfreeze at a target epoch with a reduced LR.

    Attach at the Trainer for the gradual strategy. When this callback is
    used, construct the :class:`FineTuningModule` with ``freeze_backbone=False``
    so this callback is the SOLE owner of freeze state.

    Args:
        unfreeze_at_epoch: Epoch at which the backbone is unfrozen.
        initial_denom_lr: Backbone enters at ``optimizer_lr / initial_denom_lr``
            (discriminative LR; 10.0 is the ULMFiT default).
    """

    def __init__(self, unfreeze_at_epoch: int = 10, initial_denom_lr: float = 10.0) -> None:
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._initial_denom_lr = initial_denom_lr

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """Freeze the backbone before training starts.

        This method runs before optimizer construction, so the initial
        Adam holds only head params.

        Args:
            pl_module: The :class:`FineTuningModule` instance.
        """
        self.freeze(pl_module._backbone)

    def finetune_function(
        self, pl_module: pl.LightningModule, current_epoch: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Unfreeze the backbone at the target epoch.

        Uses :meth:`BaseFinetuning.unfreeze_and_add_param_group` to add
        the backbone as a new optimizer param group — no re-init, no
        LR clobber.

        Args:
            pl_module: The :class:`FineTuningModule` instance.
            current_epoch: Current training epoch.
            optimizer: The active optimizer.
        """
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module._backbone,
                optimizer=optimizer,
                initial_denom_lr=self._initial_denom_lr,
            )
