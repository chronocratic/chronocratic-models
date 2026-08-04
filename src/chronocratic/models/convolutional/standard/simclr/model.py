__all__ = ["SimCLR"]

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.simclr.augmentations import _default_simclr_pair
from chronocratic.models.convolutional.standard.simclr.encoder import ResNet1dEncoder
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.losses import NTXentLoss
from chronocratic.models.utils import (
    extract_features_from_batch,
    process_sample_length,
    zero_fill_padding,
)
from chronocratic.models.utils.helpers import _warn_sequence_fallback

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair


class SimCLR(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for SimCLR (self-supervised pretraining only).

    Instance-level contrastive learning: two augmented views of each batch are
    encoded by a shared ResNet-1D backbone, projected through a two-layer head,
    and contrasted with NT-Xent. The other ``2N - 2`` views in the batch act as
    negatives, so batches of one carry no learning signal — use
    ``batch_size >= 2``.

    ``encode()`` returns backbone features, not projections. The projection
    head exists only to shape the contrastive objective and is discarded
    downstream, which is the arrangement SimCLR prescribes.

    Batch format: a bare tensor, or ``(data, labels)`` with labels ignored.
    For downstream classification or regression, use :class:`SupervisedModule`
    from ``chronocratic.models.supervised``.

    This model was implemented based on the code available on this GitHub repo
    https://github.com/mqwfrog/ULTS (``models/SimCLR/``), the time-series
    reference for SimCLR. Deliberate divergences from it:

    - **1-D convolutions.** The reference stacks ``Conv2d`` over an input
      unsqueezed to ``(B, C, T, 1)``. Numerically identical; see
      :mod:`~chronocratic.models.convolutional.standard.simclr.encoder`.
    - **Each encoder stage honours its own configured depth.** The reference
      repeats the third stage's index in place of the fourth. See
      :class:`ResNet1dEncoder`.
    - **Linear warmup into cosine annealing**, the schedule the original
      SimCLR specifies (google-research/simclr, ``WarmUpAndCosineDecay``).
      The reference instead builds ``ExponentialLR(gamma=decay_lr)`` reusing
      the weight-decay value ``1e-6`` as the decay factor, which multiplies
      the learning rate by ``1e-6`` every epoch — it is at ``1e-15`` after
      two. It then calls ``scheduler.step(v_loss)``, an argument
      ``ExponentialLR`` ignores. See ``configure_optimizers``.
    - **Views come from the injected producer, per batch.** The reference
      augments the whole training set once, offline. Its training loop also
      binds four names to a dataset item that yields five
      (``x, y, aug1, aug2, index``), which raises ``ValueError`` before the
      first batch — the published SimCLR path does not run as written. Had it
      bound correctly it would have paired the raw series with ``aug1``,
      contrasting an unaugmented view against one augmented view rather than
      the two augmented views against each other. Both augmented views are
      used here.
    - **Normalization defaults to ``GroupNorm(1, C)``** rather than the
      reference's hardcoded ``BatchNorm``. See
      :func:`~chronocratic.models.convolutional.standard.simclr.encoder._norm_layer`;
      ``normalization_layer_type=BATCH`` restores the reference.
    - **Default scaling uses ``sigma=0.1`` and a per-sample factor.** The
      reference's ``sigma=1.1`` places a substantial fraction of its scale
      factors below zero, inverting the sign of the series they scale. See
      :func:`~chronocratic.models.convolutional.standard.simclr.augmentations._default_simclr_pair`.
    - **``encode()`` does not L2-normalize.** The reference returns
      ``F.normalize(h_out)``. Every other model in this library returns raw
      features, and the projection path normalizes internally either way, so
      the loss is unaffected.

    Args:
        input_dim: Number of input features (channels) in the time series.
        conv_kernel_size: Kernel size of the stem convolution and of every
            residual convolution that is not a 1-tap bottleneck projection.
        stem_conv_channels: Number of channels produced by the stem
            convolution — the single convolution that precedes the residual
            stages and lifts the ``input_dim`` raw input channels to the
            width the first stage expects. Distinct from
            ``encoder_stage_channels``, which governs the residual stages
            only.
        encoder_stage_channels: Width of each residual stage, one entry per
            stage. The number of entries fixes the number of stages, and
            ``encoder_stage_depths`` and ``encoder_stage_strides`` must match
            it in length. A stage's output width is its entry multiplied by
            the block's ``expansion``, so the final entry together with
            ``residual_block_type`` determines ``representation_dim``.
        encoder_stage_depths: Number of residual blocks stacked in each
            stage — the stage's depth. One entry per stage, same length as
            ``encoder_stage_channels``.
        encoder_stage_strides: Temporal stride applied by each stage, one
            entry per stage. Only the first block of a stage applies it; the
            remaining blocks use stride 1. A stride of ``2`` halves the
            sequence length at that stage.
        residual_block_type: ``BOTTLENECK`` (``expansion = 4``) or ``BASIC``
            (``expansion = 1``). Depth comes from ``encoder_stage_depths``, so this
            selects the block and not the ResNet variant: the defaults
            together give ResNet-50, and switching to ``BASIC`` alone gives a
            ResNet-34-shaped backbone rather than a ResNet-18. Together with
            ``encoder_stage_channels[-1]`` it fixes ``representation_dim``, which is
            therefore a read-only property rather than a constructor argument.
        projection_hidden_dim: Hidden width of the two-layer projection head.
        projection_dim: Output width of the projection head — the space the
            NT-Xent loss operates in. Never used downstream; ``encode()``
            returns encoder features.
        temperature: Temperature of the NT-Xent loss.
        learning_rate: Learning rate for the Adam optimizer.
        weight_decay: Weight decay for the Adam optimizer.
        use_lr_scheduler: Whether to warm up and then cosine-anneal the
            learning rate to zero over training. Defaults to ``True``.
        warmup_epochs: Length of the linear warmup, in epochs, capped at half
            the trainer's ``max_epochs``. Defaults to ``10``.
        max_train_length: Random-crop length applied to training batches only.
            ``None`` disables cropping. ``encode()`` never crops.
        sync_dist: Whether to synchronize logged metrics across processes.
        normalization_layer_type: Normalization strategy for the encoder and
            the projection head. ``CHANNEL`` uses GroupNorm(1, C), which is
            batch-size independent. ``BATCH`` uses BatchNorm1d and reproduces
            the reference.
        augmentation: Optional custom augmentation producer. Defaults to the
            reference weak/strong pair (Gaussian scaling; segment permutation
            with jitter).
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    def __init__(
        self,
        *,
        input_dim: int,
        conv_kernel_size: int = 3,
        stem_conv_channels: int = 64,
        encoder_stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        encoder_stage_depths: tuple[int, ...] = (3, 4, 6, 3),
        encoder_stage_strides: tuple[int, ...] = (1, 2, 2, 2),
        residual_block_type: ResidualBlockType = ResidualBlockType.BOTTLENECK,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
        temperature: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        use_lr_scheduler: bool = True,
        warmup_epochs: int = 10,
        max_train_length: int | None = None,
        sync_dist: bool = False,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
        augmentation: "AugmentationProducer[ViewPair] | None" = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])

        self._input_dim = input_dim
        self._conv_kernel_size = conv_kernel_size
        self._stem_channels = stem_conv_channels
        self._encoder_channels = encoder_stage_channels
        self._encoder_depths = encoder_stage_depths
        self._encoder_strides = encoder_stage_strides
        self._residual_block_type = residual_block_type
        self._projection_hidden_dim = projection_hidden_dim
        self._projection_dim = projection_dim
        self._temperature = temperature
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._use_lr_scheduler = use_lr_scheduler
        self._warmup_epochs = warmup_epochs
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist
        self._normalization_layer_type = normalization_layer_type

        self._encoder = ResNet1dEncoder(
            input_dim=input_dim,
            conv_kernel_size=conv_kernel_size,
            stem_conv_channels=stem_conv_channels,
            encoder_stage_channels=encoder_stage_channels,
            encoder_stage_depths=encoder_stage_depths,
            encoder_stage_strides=encoder_stage_strides,
            residual_block_type=residual_block_type,
            normalization_layer_type=normalization_layer_type,
        )

        projection_norm = (
            nn.GroupNorm(num_groups=1, num_channels=projection_hidden_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(projection_hidden_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self._encoder.representation_dim, projection_hidden_dim),
            projection_norm,
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

        self._augmentation: AugmentationProducer[ViewPair] = (
            _default_simclr_pair() if augmentation is None else augmentation
        )

        self.criterion = NTXentLoss(temperature=temperature, use_cosine_similarity=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized projections for ``x``.

        Args:
            x: Input batch of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Unit-norm projections of shape ``(batch, projection_dim)``.
        """
        # DIVERGENCE: the reference returns the pair ``(F.normalize(h),
        # F.normalize(z))`` and its loss consumes only ``z``. This returns
        # ``z`` alone; ``h`` is reached through ``encode()``, which returns it
        # unnormalized in keeping with every other model in this library.
        #
        # The normalization here is required, not incidental. The reference's
        # loss takes plain dot products and subtracts ``exp(1 / tau)`` from
        # each row sum to remove the self-similarity term, which is correct
        # only when every row has unit norm — the same condition that makes the
        # dot product a cosine similarity. Removing this call would leave the
        # loss optimizing an incorrect denominator without raising.
        return functional.normalize(self.projection_head(self._encoder(x)), dim=-1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the NT-Xent loss over two augmented views of the batch.

        Labels, if present in the batch, are ignored — this model handles
        self-supervised pretraining only.
        """
        data = extract_features_from_batch(batch).float()
        # Crop first: any later mask must describe the cropped tensor.
        data = process_sample_length(sample=data, max_sample_length=self._max_train_length)
        # NaN defense: zero-fill padded timesteps before augmentation, because
        # the augmentation primitives and Conv1d both propagate NaN.
        data, _ = zero_fill_padding(data)

        pair = self._augmentation.produce(data)
        return self.criterion(self(pair.first), self(pair.second))

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], _batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the training loss for one batch."""
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

    def validation_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...], _batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the validation loss for one batch."""
        with torch.no_grad():
            loss = self._compute_loss(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist
        )
        return loss

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> "torch.optim.Optimizer | OptimizerLRSchedulerConfig":
        """Return Adam with a linear-warmup, cosine-annealed learning rate."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        # DIVERGENCE: the learning-rate schedule follows the original SimCLR
        # (google-research/simclr, ``WarmUpAndCosineDecay``) rather than this
        # port's immediate reference. ULTS wraps the optimizer in
        # ``ExponentialLR(gamma=decay_lr)``, reusing the weight-decay value
        # ``1e-6`` as the per-epoch decay factor, which drives the learning
        # rate to ``1e-15`` within two epochs; it then calls
        # ``scheduler.step(v_loss)``, an argument ``ExponentialLR`` ignores.
        # Porting it verbatim would suppress training entirely.
        #
        # Omitting the schedule is not a neutral alternative. At a constant
        # rate the parameters are still in motion when the epoch budget ends,
        # so the final weights depend on where the trajectory happened to be
        # rather than on a converged optimum, and arbitrarily small numerical
        # differences accumulate into materially different models. Annealing to
        # zero removes that dependence. ULTS itself applies this same
        # warmup-then-cosine schedule to SwAV
        # (``models/SwAV/train_SwAV.py``), though not to SimCLR.
        if not self._use_lr_scheduler:
            return optimizer

        # `self.trainer` raises when the module is used outside a Trainer, and
        # max_epochs is None or -1 for step-bounded or open-ended runs. In
        # either case there is no horizon to anneal over, so stay constant.
        trainer = getattr(self, "_trainer", None)
        max_epochs = getattr(trainer, "max_epochs", None)
        if not max_epochs or max_epochs < 0:
            return optimizer

        # Keep at least half the budget for annealing: the default 10-epoch
        # warmup would otherwise consume a short run entirely and leave the
        # learning rate high at the end, which is the case this guards against.
        warmup_epochs = min(self._warmup_epochs, max_epochs // 2)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=0.0
        )
        if warmup_epochs == 0:
            scheduler: torch.optim.lr_scheduler.LRScheduler = cosine
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    # start_factor must be strictly positive, so warmup starts
                    # from a negligible rate rather than the reference's zero.
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs
                    ),
                    cosine,
                ],
                milestones=[warmup_epochs],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> nn.Module:
        """Expose the backbone (before the projection head) to ``encode()``."""
        return self._encoder

    @property
    def encoder(self) -> nn.Module:
        """Return the ResNet-1D backbone for inspection and checkpointing."""
        return self._encoder

    @property
    def representation_dim(self) -> int:
        """Width of ``encode()``'s output.

        Fixed by the architecture as ``encoder_stage_channels[-1] * expansion``,
        where ``expansion`` follows from ``residual_block_type``. Change it by
        changing those two, not by assignment.
        """
        return self._encoder.representation_dim

    def _encode_batch(
        self,
        encoder: nn.Module,
        batch_x: torch.Tensor,
        *,
        output: EncodingOutputShape = EncodingOutputShape.VECTOR,
    ) -> torch.Tensor:
        """Return pooled backbone features, not projections.

        The backbone pools globally over time, so there is no temporal axis to
        return: SEQUENCE is a length-1 fallback.
        """
        batch_x, _ = zero_fill_padding(batch_x)
        flat = encoder(batch_x.float())  # (B, D) — D = representation_dim
        if output == EncodingOutputShape.VECTOR:
            return flat
        if output == EncodingOutputShape.SEQUENCE:
            _warn_sequence_fallback(type(self))
            return flat.unsqueeze(1)  # (B, 1, D) — fake temporal axis
        msg = f"SimCLR does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)
