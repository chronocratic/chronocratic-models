from __future__ import annotations

__all__ = ["TST"]

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.transformer.tst.loss import MaskedMSELoss
from chronocratic.models.transformer.tst.ts_transformer import TSTransformerEncoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


class TST(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for TST.

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
    of shape ``(batch, seq_len, hidden_dim)``, not the masked-reconstruction
    output. The reconstruction head is internal and used only during
    pretraining.

    For downstream classification / regression, use :class:`SupervisedModule`
    from ``chronocratic.models.supervised``.

    Args:
        input_dim: Number of input features (channels).
        sequence_length: Maximum sequence length supported by the positional
            encoding.
        hidden_dim: Transformer model (token) dimensionality.
        num_heads: Number of attention heads.
        depth: Number of stacked transformer encoder layers.
        feedforward_dim: Hidden dimensionality of the transformer
            feed-forward block.
        dropout_rate: Dropout probability used throughout the transformer.
        pos_encoding: Positional-encoding type (e.g. ``'fixed'`` or
            ``'learnable'``) passed to the encoder.
        activation: Activation function name passed to the transformer
            feed-forward block.
        normalization_layer_type: Normalization layer used inside the
            encoder. ``BATCH`` (default) uses custom BatchNorm transformer
            layers. ``CHANNEL`` uses PyTorch's LayerNorm-based
            TransformerEncoderLayer.
        freeze: When ``True``, freezes the backbone weights and only
            trains the output layer.
        learning_rate: Base learning rate for the Adam optimizer.
        lr_step: Milestones (in epochs) for the MultiStepLR scheduler.
            ``None`` means no decay (defaults to a single far-future
            milestone internally).
        lr_factor: Multiplicative decay factor applied at each
            ``lr_step`` milestone.
        weight_decay: L2 regularization coefficient. Applied to the output
            layer only when ``global_reg=False``, or to all parameters
            (via optimizer weight decay) when ``global_reg=True``.
        global_reg: Whether ``weight_decay`` is applied globally as
            weight decay (``True``) or only to the output layer
            (``False``).
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
        augmentation: Optional custom augmentation function.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/gzerveas/mvts_transformer under MIT License.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        depth: int = 3,
        feedforward_dim: int = 256,
        dropout_rate: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.BATCH,
        *,
        freeze: bool = False,
        learning_rate: float = 1e-3,
        lr_step: tuple[int, ...] | None = None,
        lr_factor: float = 0.1,
        weight_decay: float = 0.0,
        global_reg: bool = False,
        sync_dist: bool = False,
        augmentation: Callable | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])

        self._weight_decay = weight_decay
        self._global_reg = global_reg
        self._learning_rate = learning_rate
        self._lr_step = list(lr_step) if lr_step is not None else [1_000_000]
        self._lr_factor = lr_factor
        self._sync_dist = sync_dist

        self._augmentation = augmentation

        self._encoder = TSTransformerEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout_rate,
            pos_encoding=pos_encoding,
            activation=activation,
            normalization_layer_type=normalization_layer_type,
            freeze=freeze,
        )
        self._loss_fn: nn.Module = MaskedMSELoss(reduction="none")

        if freeze:
            for name, param in self._encoder.named_parameters():
                param.requires_grad = name.startswith("output_layer")

    # ------------------------------------------------------------------
    # Forward / representation extraction
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        """Return transformer representations of shape ``(batch, seq_len, hidden_dim)``."""
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
        if self.training and self._weight_decay and not self._global_reg:
            for name, param in self._encoder.named_parameters():
                if name == "output_layer.weight":
                    mean_loss = mean_loss + self._weight_decay * torch.sum(torch.square(param))

        return mean_loss

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, _batch_idx: int) -> torch.Tensor:
        """Compute and log the masked-reconstruction training loss for one batch."""
        loss = self._compute_loss(batch)
        self.log(
            "train_loss",
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
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
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
        weight_decay = self._weight_decay if self._global_reg else 0.0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self._lr_step, gamma=self._lr_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> nn.Module:
        """Return the transformer encoder module for ``BasicEncodingMixin.encode``."""
        return self._encoder

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Build padding mask and call ``encoder.encode_representations``.

        Args:
            encoder: The TSTransformerEncoder module.
            batch_x: Batch tensor of shape ``(B, seq_len, input_dim)``.
            output: Requested output shape. Defaults to VECTOR (2-D).

        Returns:
            Representations of shape ``(B, D)`` for VECTOR
            or ``(B, T, D)`` for SEQUENCE (B=batch, T=seq_len, D=hidden_dim).
        """
        padding_masks = torch.ones(batch_x.shape[:2], dtype=torch.bool, device=batch_x.device)
        full_sequence = encoder.encode_representations(batch_x, padding_masks)  # (B, T, D)
        if output == EncodingOutputShape.VECTOR:
            return full_sequence.mean(dim=1)  # (B, D) - mean over T
        if output == EncodingOutputShape.SEQUENCE:
            return full_sequence  # (B, T, D)
        msg = f"TST does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    @property
    def encoder(self) -> nn.Module:
        """Return the transformer encoder for inspection and checkpointing."""
        return self._encoder

    @property
    def representation_dim(self) -> int:
        """Feature dim of the pooled encode() representation.

        Returns:
            ``hidden_dim`` — the feature dimension of the vector produced
            by ``encode()`` with ``output=VECTOR``.
        """
        return self._encoder.hidden_dim
