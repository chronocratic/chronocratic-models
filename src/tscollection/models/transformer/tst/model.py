from __future__ import annotations

__all__ = ['TST']

import math
from typing import cast, Literal, TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tscollection.models.transformer.tst.loss import MaskedMSELoss
from tscollection.models.transformer.tst.ts_transformer import (
    TSTransformerEncoder,
    TSTransformerEncoderClassiregressor,
)

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

Task = Literal['imputation', 'transduction', 'classification', 'regression']


class TST(pl.LightningModule):
    """PyTorch Lightning module for the Time Series Transformer (TST).

    Supports four tasks controlled by the ``task`` parameter:

    - imputation / transduction  → MaskedMSELoss, TSTransformerEncoder backbone
    - classification / regression → CrossEntropyLoss / MSELoss, TSTransformerEncoderClassiregressor

    Batch format expected from the DataLoader:
    - unsupervised (imputation / transduction):
        (X, targets, target_masks, padding_masks, IDs)
    - supervised (classification / regression):
        (X, targets, padding_masks, IDs)

    All tensors are on CPU when returned from the DataLoader;
    Lightning moves them to the correct device before each step.
    """

    def __init__(
        self,
        feat_dim: int,
        max_seq_len: int,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        num_classes: int | None = None,
        task: Task = 'imputation',
        dropout: float = 0.1,
        pos_encoding: str = 'fixed',
        activation: str = 'gelu',
        norm: str = 'BatchNorm',
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

        self._task = task
        self._l2_reg = l2_reg
        self._global_reg = global_reg
        self._learning_rate = learning_rate
        self._lr_step = lr_step or [1_000_000]
        self._lr_factor = lr_factor
        self._sync_dist = sync_dist

        if task in ('imputation', 'transduction'):
            self._encoder: TSTransformerEncoder | TSTransformerEncoderClassiregressor = (
                TSTransformerEncoder(
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
            )
            self._loss_fn: nn.Module = MaskedMSELoss(reduction='none')

        elif task in ('classification', 'regression'):
            if num_classes is None:
                msg = f'num_classes is required for task "{task}"'
                raise ValueError(msg)
            self._encoder = TSTransformerEncoderClassiregressor(
                feat_dim=feat_dim,
                max_len=max_seq_len,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                num_classes=num_classes,
                dropout=dropout,
                pos_encoding=pos_encoding,
                activation=activation,
                norm=norm,
                freeze=freeze,
            )
            self._loss_fn = (
                nn.CrossEntropyLoss(reduction='none')
                if task == 'classification'
                else nn.MSELoss(reduction='none')
            )
        else:
            msg = f"Unknown task '{task}'"
            raise ValueError(msg)

        if freeze:
            for name, param in self._encoder.named_parameters():
                param.requires_grad = name.startswith('output_layer')

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Run the encoder on masked input ``x`` with the given padding masks."""
        return self._encoder(x, padding_masks)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        if self._task in ('imputation', 'transduction'):
            x, targets, target_masks, padding_masks, _ = batch
            predictions = self(x, padding_masks)
            combined_mask = target_masks * padding_masks.unsqueeze(-1)
            per_element_loss = self._loss_fn(predictions, targets, combined_mask)  # (num_active,)
        else:
            x, targets, padding_masks, _ = batch
            predictions = self(x, padding_masks)
            if self._task == 'classification':
                per_element_loss = self._loss_fn(predictions, targets.long().squeeze())
            else:
                per_element_loss = self._loss_fn(predictions, targets)

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

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
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

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute and log the validation loss for one batch."""
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
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=4.0)

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
    # Representation extraction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, data: torch.Tensor, batch_size: int, num_workers: int = 0) -> torch.Tensor:
        """Extract backbone representations for ``data`` of shape ``(N, T, C)``.

        Returns a tensor of shape ``(N, T, d_model)``: the transformer output
        before the task-specific output layer, suitable for downstream tasks.
        """
        was_training = self._encoder.training
        self._encoder.eval()

        loader = DataLoader(
            TensorDataset(data), batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        outputs = []
        for (batch_x,) in loader:
            inp = batch_x.to(self.device)
            padding_masks = torch.ones(
                inp.shape[0], inp.shape[1], dtype=torch.bool, device=self.device
            )
            outputs.append(self._backbone(inp, padding_masks).cpu())

        self._encoder.train(was_training)
        return torch.cat(outputs, dim=0)

    def _backbone(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Run the shared transformer trunk, stopping before the output layer.

        Both TSTransformerEncoder and TSTransformerEncoderClassiregressor share
        the same trunk attributes (project_inp, pos_enc, transformer_encoder,
        act, dropout1), so this works regardless of task.
        """
        enc = cast('TSTransformerEncoder', self._encoder)
        inp = x.permute(1, 0, 2)
        inp = enc.project_inp(inp) * math.sqrt(enc.d_model)
        inp = enc.pos_enc(inp)
        out = enc.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        out = enc.act(out)
        out = out.permute(1, 0, 2)
        return enc.dropout1(out)
