__all__ = ["TSTCC", "_tstcc_encoder_output_length"]

from typing import cast, TYPE_CHECKING
import warnings

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.tstcc.encoder import TCCEncoder
from chronocratic.models.convolutional.standard.tstcc.losses import NTXentLoss
from chronocratic.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.utils import extract_features_from_batch, zero_fill_padding

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair


# ---------------------------------------------------------------------------
# Encoder output length computation
# ---------------------------------------------------------------------------


def _tstcc_encoder_output_length(seq_len: int) -> int:
    """Compute the output sequence length of TCCEncoder for a given input length.

    TCCEncoder has 3 MaxPool1d(kernel_size=2, stride=2, padding=1) stages.
    Each stage: output = input // 2 + 1.

    Args:
        seq_len: Input sequence length.

    Returns:
        Output sequence length after three pooling stages.
    """

    def pool(length: int) -> int:
        return length // 2 + 1

    return pool(pool(pool(seq_len)))


def _clamp_timesteps(temporal_contrast_timesteps: int, seq_len: int) -> int:
    """Auto-clamp temporal contrast timesteps based on encoder output.

    TemporalContrast.forward requires encoder_output_seq_len > timesteps.
    If the encoder shrinks the sequence too much, clamp timesteps to
    max(1, encoder_output_seq_len - 1).

    Args:
        temporal_contrast_timesteps: Original timesteps from config.
        seq_len: Input sequence length.

    Returns:
        Clamped timesteps value.
    """
    encoder_out = _tstcc_encoder_output_length(seq_len)
    if encoder_out <= temporal_contrast_timesteps:
        return max(1, encoder_out - 1)
    return temporal_contrast_timesteps


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

    Args:
        input_dim: Number of input features (channels).
        conv_kernel_size: Kernel size for the first convolution block.
        stride: Stride for the first convolution block.
        representation_dim: Number of output channels from the encoder.
        encoder_channels: Channel counts for the first two conv blocks.
        encoder_inner_kernels: Kernel sizes for the second and third conv
            blocks.
        dropout_rate: Dropout rate applied after the first conv block.
        temporal_contrast_hidden_dim: Hidden dimension for the transformer
            and projection head in TemporalContrast.
        temporal_contrast_timesteps: Number of timesteps for temporal
            contrastive prediction.
        temperature: Temperature parameter for NT-Xent loss.
        use_cosine_similarity: Whether to use cosine similarity in NT-Xent.
        learning_rate: Learning rate for Adam optimizers.
        temporal_loss_weight: Weight for the temporal contrastive loss.
        contextual_loss_weight: Weight for the contextual NT-Xent loss.
        weight_decay: Weight decay for Adam optimizers.
        sync_dist: Whether to synchronize metrics across processes.
        normalization_layer_type: Normalization strategy. ``CHANNEL`` uses
            GroupNorm(1, C) in the encoder and LayerNorm in the projection
            head, which is batch-size independent and avoids degeneracy at
            small batch sizes. ``BATCH`` uses BatchNorm1d. Defaults to
            ``CHANNEL``.
        augmentation: Optional custom augmentation producer. Defaults to
            the standard TSTCC pair (Gaussian scaling + segment permutation
            with jitter).
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        *,
        input_dim: int,
        conv_kernel_size: int = 8,
        stride: int = 1,
        representation_dim: int = 128,
        encoder_channels: tuple[int, ...] = (32, 64),
        encoder_inner_kernels: tuple[int, ...] = (8, 8),
        dropout_rate: float = 0.35,
        temporal_contrast_hidden_dim: int = 100,
        temporal_contrast_timesteps: int = 6,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
        learning_rate: float = 3e-4,
        temporal_loss_weight: float = 1.0,
        contextual_loss_weight: float = 0.7,
        weight_decay: float = 0.0003,
        sync_dist: bool = False,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
        augmentation: "AugmentationProducer[ViewPair] | None" = None,
        sequence_length: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])
        self.automatic_optimization = False

        self._learning_rate = learning_rate
        self._temporal_loss_weight = temporal_loss_weight
        self._contextual_loss_weight = contextual_loss_weight
        self._weight_decay = weight_decay
        self._sync_dist = sync_dist

        # Auto-clamp timesteps if sequence_length is provided
        self._original_timesteps = temporal_contrast_timesteps
        if sequence_length is not None:
            clamped = _clamp_timesteps(temporal_contrast_timesteps, sequence_length)
            if clamped != temporal_contrast_timesteps:
                warnings.warn(
                    f"TSTCC: encoder output length "
                    f"({_tstcc_encoder_output_length(sequence_length)}) is "
                    f"<= temporal_contrast_timesteps "
                    f"({temporal_contrast_timesteps}). Clamping timesteps "
                    f"to {clamped}.",
                    UserWarning,
                    stacklevel=2,
                )
            temporal_contrast_timesteps = clamped

        self.temporal_contrast_timesteps = temporal_contrast_timesteps
        self._timesteps_warned = sequence_length is not None
        # Store for potential runtime _tc_model reconstruction
        self._tc_hidden_dim = temporal_contrast_hidden_dim
        self._tc_normalization_layer_type = normalization_layer_type

        if augmentation is None:
            from chronocratic.models.convolutional.standard.tstcc.augmentations import (  # noqa: PLC0415
                _default_tstcc_pair,
            )

            self._augmentation: AugmentationProducer[ViewPair] = _default_tstcc_pair()
        else:
            self._augmentation = augmentation

        self._encoder = TCCEncoder(
            input_dim=input_dim,
            conv_kernel_size=conv_kernel_size,
            stride=stride,
            representation_dim=representation_dim,
            encoder_channels=encoder_channels,
            encoder_inner_kernels=encoder_inner_kernels,
            dropout_rate=dropout_rate,
            normalization_layer_type=normalization_layer_type,
        )
        self._tc_model = TemporalContrast(
            num_channels=representation_dim,
            hidden_dim=temporal_contrast_hidden_dim,
            timesteps=temporal_contrast_timesteps,
            normalization_layer_type=normalization_layer_type,
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

        # NaN defense: zero-fill padded timesteps before augmentation
        data, _ = zero_fill_padding(data)

        # Runtime timesteps clamping (when sequence_length not provided at init)
        if not self._timesteps_warned:
            seq_len = data.shape[1]
            clamped = _clamp_timesteps(self.temporal_contrast_timesteps, seq_len)
            if clamped != self.temporal_contrast_timesteps:
                warnings.warn(
                    f"TSTCC: encoder output length ({_tstcc_encoder_output_length(seq_len)}) "
                    f"is <= temporal_contrast_timesteps ({self.temporal_contrast_timesteps}). "
                    f"Clamping timesteps to {clamped}.",
                    UserWarning,
                    stacklevel=2,
                )
                self.temporal_contrast_timesteps = clamped
                self._tc_model = TemporalContrast(
                    num_channels=self._encoder.representation_dim,
                    hidden_dim=self._tc_hidden_dim,
                    timesteps=clamped,
                    normalization_layer_type=self._tc_normalization_layer_type,
                )
            self._timesteps_warned = True

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
        - ``C``: encoder output channels (``representation_dim``)
        - ``L'``: conv-downsampled sequence length (``L' = seq_len // stride``)
        """
        # ponytail: zero-fill only. Padding contaminates the WHOLE feature map, not
        # just its receptive field: GroupNorm(1, C) reduces over (C, L'), so padded
        # values enter the norm statistics and shift every output position (measured
        # 66/66 at seq_len=512, valid=300; 31/66 with the norms stripped). Masking
        # this pooling would therefore not help — the values are already contaminated
        # upstream. Real fix is masked normalization across all 3 blocks, which
        # changes the encoder and invalidates trained checkpoints. Training pools
        # nothing and contaminates identically, so this is a representation-quality
        # ceiling, not a correctness bug.
        batch_x, _ = zero_fill_padding(batch_x)
        features = encoder(batch_x.float())  # (B, C, L')
        if output == EncodingOutputShape.VECTOR:
            return features.mean(dim=-1)  # (B, C) — VECTOR
        if output == EncodingOutputShape.SEQUENCE:
            return features.transpose(1, 2)  # (B, L', C) — SEQUENCE
        msg = f"TSTCC does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)

    @property
    def representation_dim(self) -> int:
        """Representation dimension after global average pooling.

        Returns:
            The encoder's ``representation_dim``, matching the pooled feature
            shape ``(B, representation_dim)``.
        """
        return self._encoder.representation_dim
