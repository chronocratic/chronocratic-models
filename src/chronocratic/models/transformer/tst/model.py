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
from chronocratic.models.utils import (
    extract_features_from_batch,
    process_sample_length,
    zero_fill_padding,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


class TST(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for TST.

    Representation-learning model trained with a masked-reconstruction
    pretraining objective. Input masking is generated INTERNALLY from
    ``masking_ratio`` (Bernoulli, independent per element); the dataloader
    supplies no masks.

    Accepts any batch format handled by ``extract_features_from_batch``:
    a bare ``(B, T, F)`` tensor, or a tuple/list whose first element is
    that tensor (e.g. ``(X, y)`` from UEA/UCR loaders). Labels are ignored.

    Padded timesteps must arrive as NaN (see ``pad_tensor_with_nan``); that
    is the only signal separating padding from genuine zeros. They are
    excluded from attention, from the reconstruction loss, and from VECTOR
    pooling. A sample with no valid timesteps raises ``ValueError``.

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
        masking_ratio: Fraction of input elements zeroed during
            masked-reconstruction pretraining. Each element is masked
            independently (Bernoulli). Must be in the open interval
            ``(0.0, 1.0)``. Default ``0.15`` matches the upstream default.
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
        weight_decay: L2 regularization coefficient. Must be non-negative.
            Inactive at the default ``0.0``. When positive, applied to all
            parameters via optimizer weight decay if ``global_reg=True``, or
            added to the training loss as an L2 penalty on the output layer
            alone if ``global_reg=False``.
        global_reg: Selects where a positive ``weight_decay`` is applied:
            globally via the optimizer (``True``) or to the output layer
            only (``False``). No effect when ``weight_decay=0.0``.
        sync_dist: Whether to synchronize logged metrics across
            distributed processes.
        augmentation: Optional custom augmentation function.
        max_train_length: Maximum sequence length used during training; longer
            batches are randomly cropped to this length. ``None`` means no
            cap, which will fail on inputs longer than ``sequence_length``.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/gzerveas/mvts_transformer under MIT License.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        depth: int = 3,
        feedforward_dim: int = 256,
        dropout_rate: float = 0.1,
        masking_ratio: float = 0.15,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.BATCH,
        freeze: bool = False,
        learning_rate: float = 1e-3,
        lr_step: tuple[int, ...] | None = None,
        lr_factor: float = 0.1,
        weight_decay: float = 0.0,
        global_reg: bool = False,
        sync_dist: bool = False,
        augmentation: Callable | None = None,
        max_train_length: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])

        # torch.optim rejects a negative weight_decay; mirror that on the
        # global_reg=False path, which never reaches the optimizer.
        if weight_decay < 0.0:
            msg = f"weight_decay must be non-negative, got {weight_decay}."
            raise ValueError(msg)
        self._weight_decay = weight_decay
        self._global_reg = global_reg
        self._learning_rate = learning_rate
        self._lr_step = list(lr_step) if lr_step is not None else [1_000_000]
        self._lr_factor = lr_factor
        self._sync_dist = sync_dist

        self._augmentation = augmentation

        if masking_ratio <= 0.0 or masking_ratio >= 1.0:
            msg = f"masking_ratio must be in the open interval (0.0, 1.0), got {masking_ratio}."
            raise ValueError(msg)
        self._masking_ratio = masking_ratio

        self._sequence_length = sequence_length
        self._max_train_length = max_train_length

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
        self._loss_fn: nn.Module = MaskedMSELoss(reduction="mean")

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

    @staticmethod
    def _split_padding(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-fill NaN-padded timesteps and recover the real padding mask.

        Padding reaches the model as NaN (see ``pad_tensor_with_nan``), which is
        the only signal distinguishing padded timesteps from genuine zeros. It
        must be resolved before the trunk runs, so attention can be told to skip
        those positions rather than pooling them into real ones.

        Args:
            x: Batch of shape ``(B, T, F)``, NaN at padded timesteps.

        Returns:
            ``(x_filled, padding_masks)`` where ``padding_masks`` is ``(B, T)``
            with ``True`` at valid timesteps.

        Raises:
            ValueError: If any sample has no valid timesteps. Such a row makes
                attention pool over nothing and yields NaN, which BatchNorm then
                spreads across the whole batch.
        """
        x_filled, padding_masks = zero_fill_padding(x)
        if not padding_masks.any(dim=1).all():
            bad = (~padding_masks.any(dim=1)).nonzero().flatten().tolist()
            msg = f"Samples {bad} are entirely NaN (no valid timesteps); cannot build a mask."
            raise ValueError(msg)
        return x_filled, padding_masks

    def _make_masked_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bernoulli-mask x in place of upstream's collate_unsuperv.

        Args:
            x: Batch of shape ``(B, T, F)``.

        Returns:
            ``(masked_x, targets, target_masks, padding_masks)`` where
            ``target_masks`` is ``(B, T, F)`` with ``True`` at scored positions
            and ``padding_masks`` is ``(B, T)`` with ``True`` at valid timesteps.
        """
        x_filled, padding_masks = self._split_padding(x)
        # ponytail: Bernoulli mask; upstream also offers geometric (lm=3). Add if repr quality lags.
        keep = torch.rand(x_filled.shape, device=x.device) >= self._masking_ratio
        masked_x = x_filled * keep
        return masked_x, x_filled, ~keep, padding_masks

    def _compute_loss(self, batch: torch.Tensor | tuple | list) -> torch.Tensor:
        x = extract_features_from_batch(batch)
        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)
        masked_x, targets, target_masks, padding_masks = self._make_masked_inputs(x)
        predictions = self.reconstruct(masked_x, padding_masks)
        combined_mask = target_masks & padding_masks.unsqueeze(-1)
        if not combined_mask.any():
            return x.new_zeros((), requires_grad=True)
        mean_loss = self._loss_fn(predictions, targets, combined_mask)

        # output-layer-only L2 (global L2 is handled via weight_decay in the optimizer)
        if self.training and self._weight_decay and not self._global_reg:
            mean_loss = mean_loss + self._weight_decay * torch.sum(
                torch.square(self._encoder.output_layer.weight)
            )

        return mean_loss

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor | tuple | list, _batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: torch.Tensor | tuple | list, _batch_idx: int) -> torch.Tensor:
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
        x_filled, padding_masks = self._split_padding(batch_x)
        assert isinstance(encoder, TSTransformerEncoder)  # noqa: S101  # narrows Module -> TSTransformerEncoder
        full_sequence = encoder.encode_representations(x_filled, padding_masks)  # (B, T, D)
        if output == EncodingOutputShape.VECTOR:
            # Mean over real timesteps only; padded ones would drag the average.
            keep = padding_masks.unsqueeze(-1).to(full_sequence.dtype)  # (B, T, 1)
            return (full_sequence * keep).sum(dim=1) / keep.sum(dim=1)  # (B, D)
        if output == EncodingOutputShape.SEQUENCE:
            return full_sequence  # (B, T, D)
        msg = f"TST does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    @property
    def encoder(self) -> nn.Module:
        """Return the transformer encoder for inspection and checkpointing."""
        return self._encoder

    @property
    def sequence_length(self) -> int:
        """Return the maximum sequence length supported by this model."""
        return self._sequence_length

    @property
    def representation_dim(self) -> int:
        """Feature dim of the pooled encode() representation.

        Returns:
            ``hidden_dim`` — the feature dimension of the vector produced
            by ``encode()`` with ``output=VECTOR``.
        """
        return self._encoder.hidden_dim
