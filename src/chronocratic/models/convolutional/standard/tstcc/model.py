__all__ = ["TSTCC"]

from typing import cast, TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.losses import NTXentLoss
from chronocratic.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.utils import extract_features_from_batch

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair


class TSTCC(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for TS-TCC (self-supervised pretraining only).

    Single-purpose model for temporal + contextual contrastive pre-training
    on augmented views. Labels are ignored during pretraining.

    Batch format: ``(data, labels)`` where ``labels`` is ignored.
    Two augmented views of ``data`` are produced by the injected
    ``AugmentationProducer[ViewPair]`` (e.g. :func:`_default_tstcc_pair`),
    which provides Gaussian scaling (weak) and segment-permutation + jitter
    (strong) views.

    Uses ``automatic_optimization = False`` because two separate optimizers
    (one per sub-module) must be stepped independently.

    For downstream classification or regression, use :class:`SupervisedModule`
    from ``chronocratic.models.supervised``.

    This model was implemented based on the code available on this GitHub
    repo https://github.com/emadeldeen24/TS-TCC under MIT License.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        input_dims: int,
        conv_kernel_size: int,
        stride: int,
        output_dims: int = 128,
        encoder_channels: tuple[int, ...] = (32, 64),
        encoder_inner_kernels: tuple[int, ...] = (8, 8),
        dropout_rate: float = 0.35,
        temporal_contrast_hidden_dim: int = 100,
        temporal_contrast_timesteps: int = 6,
        temperature: float = 0.2,
        *,
        use_cosine_similarity: bool = True,
        learning_rate: float = 3e-4,
        temporal_loss_weight: float = 1.0,
        contextual_loss_weight: float = 0.7,
        weight_decay: float = 0.0003,
        sync_dist: bool = False,
        augmentation: "AugmentationProducer[ViewPair] | None" = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])
        self.automatic_optimization = False

        self._learning_rate = learning_rate
        self._temporal_loss_weight = temporal_loss_weight
        self._contextual_loss_weight = contextual_loss_weight
        self._weight_decay = weight_decay
        self._sync_dist = sync_dist

        if augmentation is None:
            from chronocratic.models.convolutional.standard.tstcc.augmentations import (  # noqa: PLC0415
                _default_tstcc_pair,
            )

            self._augmentation: AugmentationProducer[ViewPair] = _default_tstcc_pair()
        else:
            self._augmentation = augmentation

        self._encoder = TCCEncoder(
            input_dims=input_dims,
            conv_kernel_size=conv_kernel_size,
            stride=stride,
            output_dims=output_dims,
            encoder_channels=encoder_channels,
            encoder_inner_kernels=encoder_inner_kernels,
            dropout_rate=dropout_rate,
        )
        self._tc_model = TemporalContrast(
            num_channels=output_dims,
            hidden_dim=temporal_contrast_hidden_dim,
            timesteps=temporal_contrast_timesteps,
        )
        self._nt_xent_loss = NTXentLoss(
            temperature=temperature, use_cosine_similarity=use_cosine_similarity
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder. Returns convolutional feature map ``(B, C, L')``."""
        return self._encoder(x)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive pretraining loss.

        Labels in the batch are ignored — this model handles self-supervised
        pretraining only. For downstream supervised tasks, use SupervisedModule.
        """
        data = extract_features_from_batch(batch).float()

        pair = self._augmentation.produce(data)
        aug1, aug2 = pair.first, pair.second
        features1 = self._encoder(aug1)
        features2 = self._encoder(aug2)
        features1 = functional.normalize(features1, dim=1)
        features2 = functional.normalize(features2, dim=1)

        temp_loss1, proj1 = self._tc_model(features1, features2)
        temp_loss2, proj2 = self._tc_model(features2, features1)

        temporal_loss = temp_loss1 + temp_loss2
        contextual_loss = self._nt_xent_loss(proj1, proj2)
        return (
            self._temporal_loss_weight * temporal_loss
            + self._contextual_loss_weight * contextual_loss
        )

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        """Manual optimization step for both sub-module optimizers."""
        optimizers = cast("list[torch.optim.Optimizer]", self.optimizers(use_pl_optimizer=False))
        model_opt, tc_opt = optimizers
        model_opt.zero_grad()
        tc_opt.zero_grad()

        loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        if not torch.isfinite(loss):
            msg = f"Loss is {loss.item()}, skipping optimization step"
            raise RuntimeError(msg)
        self.manual_backward(loss)
        model_opt.step()
        tc_opt.step()
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        """Compute and log validation loss."""
        with torch.no_grad():
            loss = self._compute_loss(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> "OptimizerLRScheduler":
        """Return one Adam optimizer per sub-module (encoder and TC model)."""
        return [
            torch.optim.Adam(
                self._encoder.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
            ),
            torch.optim.Adam(
                self._tc_model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
            ),
        ]

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> nn.Module:
        """Expose the conv encoder to ``BasicEncodingMixin.encode``."""
        return self._encoder

    @property
    def encoder(self) -> nn.Module:
        """Return the TCC encoder for inspection and checkpointing."""
        return self._encoder

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Cast to float and encode the batch.

        The TCC encoder expects float inputs, so we cast batch_x to float
        before encoding. The feature map ``(B, C, L')`` is then
        pooled to ``(B, C)`` for VECTOR, or transposed to
        ``(B, L', C)`` for SEQUENCE, where:

        - ``B``: batch size
        - ``C``: encoder output channels (``output_dims``)
        - ``L'``: conv-downsampled sequence length (``L' = seq_len // stride``)
        """
        features = encoder(batch_x.float())  # (B, C, L')
        if output == EncodingOutputShape.VECTOR:
            return features.mean(dim=-1)  # (B, C)
        if output == EncodingOutputShape.SEQUENCE:
            return features.transpose(1, 2)  # (B, L', C)
        msg = f"TSTCC does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    @property
    def representation_dim(self) -> int:
        """Representation dimension after global average pooling.

        Returns:
            The encoder's ``output_dims``, matching the pooled feature shape
            ``(B, output_dims)``.
        """
        return self._encoder.output_dims
