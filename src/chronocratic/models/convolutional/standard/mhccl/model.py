__all__ = ["MHCCL"]

from typing import TYPE_CHECKING
import warnings

import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional

from chronocratic.models._mixin import BasicEncodingMixin
from chronocratic.models.convolutional.standard.encoders.resnet import Conv1dResNetEncoder
from chronocratic.models.convolutional.standard.mhccl.augmentations import _default_mhccl_pair
from chronocratic.models.convolutional.standard.mhccl.clustering import finch
from chronocratic.models.convolutional.standard.mhccl.config import _MIN_SINGLETON_SPLIT_COUNT
from chronocratic.models.convolutional.standard.mhccl.losses import (
    instance_contrast_terms,
    masked_bce_with_logits,
    prototype_contrast_terms,
)
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.clustering import OutlierMaskMode
from chronocratic.models.enums.encoding import EncodingOutputShape
from chronocratic.models.enums.layers import NormalizationLayerType
from chronocratic.models.utils import (
    ensure_pairable_batch,
    extract_features_from_batch,
    process_sample_length,
    zero_fill_padding,
)

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair
    from chronocratic.models.convolutional.standard.mhccl.clustering import ClusterHierarchy

# Each view of the batch contributes one block of rows to the clustering pool:
# the raw series and its two augmented views.
_VIEWS_PER_SAMPLE = 3
# Multiplier on the step-decay branch of the learning-rate schedule.
_LR_STEP_GAMMA = 0.1

# Once-per-class dedup so a shallow hierarchy warns once rather than once per
# training step, mirroring the tracker in ``utils/helpers.py``.
_warned_shallow_hierarchy: set[str] = set()


def _reset_hierarchy_warning() -> None:
    """Clear the once-per-class shallow-hierarchy warning, for test isolation."""
    _warned_shallow_hierarchy.clear()


def _warn_shallow_hierarchy(cls: type, *, requested: int, available: int, pool_size: int) -> None:
    """Warn once per model class that the clustering realized fewer partitions."""
    if cls.__name__ in _warned_shallow_hierarchy:
        return
    _warned_shallow_hierarchy.add(cls.__name__)
    warnings.warn(
        f"MHCCL requested {requested} hierarchy levels but the clustering realized "
        f"{available} usable ones over a pool of {pool_size} rows. Cluster-level contrast "
        f"is averaged over the levels actually used; raise feature_bank_size or the batch "
        f"size to deepen the hierarchy. Logged every step as mhccl/hierarchy_levels_used.",
        category=UserWarning,
        stacklevel=3,
    )


# ── Parameter names vs. the reference ────────────────────────────────────────
# Parameters below are named per this library's canonical vocabulary rather
# than the reference's. These are renames only; none of them changes
# behaviour. Reproduced in docs/api/conv_standard.md.
#
#   this library                        MHCCL (main.py argparse, framework.py)
#   ----------------------------------  ------------------------------------
#   input_dim                           in_channels (from a dataset_name chain)
#   stem_conv_kernel_size               inline 8 in the net.conv1 replacement
#   stem_conv_channels                  inline 64 in net.conv1
#   encoder_stage_channels              torchvision ResNet internals
#   encoder_stage_depths                positional list of four stage depths
#   encoder_stage_strides               torchvision ResNet internals
#   residual_block_type                 resnet.BasicBlock
#   conv_kernel_size                    torchvision conv3x3
#   projection_dim                      low_dim / num_classes
#   projection_hidden_dim               dim_mlp
#   use_projection_mlp                  --mlp
#   key_momentum                        --moco_m
#   optimizer_momentum                  --momentum
#   positive_instance_count             --posi
#   negative_instance_count             --negi
#   positive_prototype_count            --posp
#   negative_prototype_count            --negp
#   hierarchy_levels                    --layers
#   use_instance_loss                   --protoNCE_only, inverted
#   instance_temperature                --tempi, gated by --usetemp
#   prototype_temperature               --tempp, gated by --usetemp
#   mask_outliers_at_base_level         --mask_layer0
#   mask_outliers_at_upper_levels       --mask_others
#   outlier_mask_mode                   --mask_mode
#   outlier_distance_threshold          --dist_threshold
#   outlier_mask_proportion             --proportion
#   replace_centroids_with_nearest_member  --replace_centroids
#   use_lr_scheduler, lr_step_milestones   --cos / --schedule
#   feature_bank_size                   no equivalent; see the divergences
#
# The reference also exposes --warmup_epoch and --req_clust, which this port
# omits: the first raises before its first step at any value above its default,
# and the second only feeds a CSV dump.


class MHCCL(pl.LightningModule, BasicEncodingMixin):
    """PyTorch Lightning module for MHCCL (self-supervised pretraining only).

    Cluster-level contrastive learning. A query encoder and a momentum-updated
    key encoder embed two augmented views; FINCH hierarchical clustering
    partitions those embeddings at several granularities; and the anchor is
    contrasted both against instances sharing its finest-partition cluster and
    against the prototypes of each partition. Two masking strategies shape the
    pairs: *upward* masking drops per-cluster outliers before recomputing a
    prototype, and *downward* masking withholds any candidate negative prototype
    that shares the anchor's parent cluster, supplementing it into the positives
    instead.

    Both terms are multi-label sigmoid objectives — binary cross-entropy over a
    multi-hot target — rather than softmax InfoNCE.

    ``encode()`` returns backbone features, not projections. The projection head
    shapes the contrastive objective and the clustering space, and is discarded
    downstream.

    Batch format: a bare tensor, or ``(data, labels)`` with labels ignored.
    For downstream classification or regression, use :class:`SupervisedModule`
    from ``chronocratic.models.supervised``.

    This model was implemented based on the code available on this GitHub repo
    https://github.com/mqwfrog/MHCCL, the reference for MHCCL. Deliberate
    divergences from it:

    - **Clustering runs per batch over a momentum feature bank.** The reference
      clusters the training set once per epoch and addresses the result through
      a dataset-global sample index this library's batch contract does not
      carry. FINCH runs instead on the batch's ``(3B, D)`` stack of raw and
      augmented embeddings — the same stacking, and the leading ``B`` rows are
      the anchors — unioned with a FIFO of recent momentum embeddings that
      restores the sample count a multi-level hierarchy needs. Prototypes are
      therefore re-derived every step rather than fixed per epoch.
    - **The feature bank is capped at one epoch of rows.** ``feature_bank_size``
      is a ceiling; the data sets the size below it. Retaining more would fill
      the pool with copies of the same samples from different encoder states.
    - **The realized hierarchy depth is reported, not assumed.** Each contrasted
      partition needs the one above it for downward masking, so
      ``min(hierarchy_levels, levels - 1)`` are used, logged as
      ``mhccl/hierarchy_levels_used``, and a shortfall warns once. The
      cluster-level term averages over the levels used; the reference divides by
      the configured count, which ties the loss scale to the batch's geometry.
    - **Logits are inner products.** The reference reshapes its ``(B, M, D)``
      candidate stack to ``(B, D, M)`` where the comment above the line asks for
      a transpose, so its einsum contracts mismatched elements. Computed here
      with ``einsum("bd,bmd->bm", ...)``.
    - **Negatives are everything outside the anchor's cluster.** The reference
      excludes only the positives it sampled, leaving a large cluster's
      unsampled members as negatives of an anchor they share a cluster with.
    - **Shortfalls are masked, not raised.** The reference draws a fixed count
      and raises when a batch holds fewer candidates. Unfillable slots are
      excluded from the mean, which equals ``reduction="mean"`` when none are.
    - **No shuffle-BN and no ``torch.distributed``.** Batch shuffling is the
      identity at world size 1 and unnecessary under the ``GroupNorm`` default,
      which computes no batch statistics.
    - **Centroids never mutate the feature matrix** and divide by true
      per-cluster counts. See
      :mod:`~chronocratic.models.convolutional.standard.mhccl.clustering`.
    - **1-D convolutions.** The reference applies a torchvision ResNet-18 as
      ``Conv2d`` over an input unsqueezed to ``(B, C, T, 1)``, where the outer
      kernel columns only multiply zero padding. Numerically identical; see
      :class:`Conv1dResNetEncoder`.
    - **Normalization defaults to ``GroupNorm(1, C)``** rather than the
      reference's ``BatchNorm``, so the encoder stays well-defined at
      ``batch_size=1``. ``normalization_layer_type=BATCH`` restores it.
    - **Views come from the injected producer, per batch.** The reference
      freezes both views at ``Dataset`` construction and never seeds
      ``np.random``, so they are not reproducible across runs.
    - **Defaults follow the README's published command** where it and the CLI
      disagree — ``use_projection_mlp`` and ``use_lr_scheduler``.

    Args:
        input_dim: Number of input features (channels) in the time series.
        conv_kernel_size: Kernel size of every residual convolution that is not
            a 1-tap bottleneck projection.
        stem_conv_kernel_size: Kernel size of the stem convolution, which the
            reference sets wider than the residual convolutions.
        stem_conv_channels: Number of channels produced by the stem convolution.
        encoder_stage_channels: Width of each residual stage, one entry per
            stage.
        encoder_stage_depths: Number of residual blocks per stage. With
            ``BASIC`` blocks the default gives a ResNet-18, which is what the
            reference builds.
        encoder_stage_strides: Temporal stride applied by each stage.
        residual_block_type: ``BASIC`` (``expansion = 1``) or ``BOTTLENECK``
            (``expansion = 4``).
        projection_hidden_dim: Hidden width of the projection head, used only
            when ``use_projection_mlp`` is set.
        projection_dim: Width of the projected space the contrastive terms and
            the clustering both operate in.
        use_projection_mlp: Whether the projection head is a two-layer MLP
            rather than a single linear map.
        key_momentum: EMA coefficient for the momentum (key) encoder.
        feature_bank_size: Upper bound on the number of momentum-encoder feature
            rows retained from previous steps and clustered alongside the
            current batch. Bounded further at runtime so the bank never spans
            more than one epoch. ``0`` clusters the current batch alone.
        hierarchy_levels: Number of clustering partitions to contrast against.
            The number actually used is bounded by what the clustering realizes.
        positive_instance_count: Positive instances per anchor in the
            instance-level term.
        negative_instance_count: Negative instances per anchor in the
            instance-level term.
        positive_prototype_count: Positive prototypes per anchor per partition,
            including the anchor's own.
        negative_prototype_count: Negative prototypes per anchor per partition.
        use_instance_loss: Whether to include the instance-level term.
        instance_temperature: Divisor for the instance-level logits. ``None``
            leaves them as raw inner products.
        prototype_temperature: Divisor for the cluster-level logits.
        mask_outliers_at_base_level: Whether upward masking refines the finest
            partition's prototypes.
        mask_outliers_at_upper_levels: Whether upward masking refines every
            coarser partition's prototypes.
        outlier_mask_mode: Which cluster members upward masking excludes.
        outlier_distance_threshold: Absolute distance beyond which a member is
            an outlier, for ``OutlierMaskMode.THRESHOLD``.
        outlier_mask_proportion: Fraction of each cluster to exclude, for
            ``OutlierMaskMode.PROPORTION``.
        replace_centroids_with_nearest_member: Whether each prototype is
            replaced by the real member closest to it.
        learning_rate: Learning rate for the SGD optimizer.
        optimizer_momentum: Momentum for the SGD optimizer. Distinct from
            ``key_momentum``, which governs the momentum encoder.
        weight_decay: Weight decay for the SGD optimizer.
        use_lr_scheduler: Whether to decay the learning rate over training.
        lr_step_milestones: Epochs at which to multiply the learning rate by
            ``0.1``. ``None`` cosine-anneals to zero instead.
        max_train_length: Random-crop length applied to training batches only.
            ``None`` disables cropping. ``encode()`` never crops.
        sync_dist: Whether to synchronize logged metrics across processes.
        normalization_layer_type: Normalization strategy for the encoders and
            the projection head.
        augmentation: Optional custom augmentation producer. Defaults to the
            reference weak/strong pair.
        singleton_split_count: Number of contiguous windows to split a singleton
            batch into.
    """

    supported_outputs: frozenset[EncodingOutputShape] = frozenset(
        {EncodingOutputShape.VECTOR, EncodingOutputShape.SEQUENCE}
    )

    # PLR0915: this model carries roughly twice the hyperparameters of the next
    # largest one here, and the library requires each to be stored explicitly as
    # ``self._{name}``. That mandate and the statement-count limit cannot both
    # hold. ``PLR0913`` is already ignored project-wide for the same reason on
    # the signature side.
    def __init__(  # noqa: PLR0915
        self,
        *,
        input_dim: int,
        conv_kernel_size: int = 3,
        stem_conv_kernel_size: int = 8,
        stem_conv_channels: int = 64,
        encoder_stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        encoder_stage_depths: tuple[int, ...] = (2, 2, 2, 2),
        encoder_stage_strides: tuple[int, ...] = (1, 2, 2, 2),
        residual_block_type: ResidualBlockType = ResidualBlockType.BASIC,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
        use_projection_mlp: bool = True,
        key_momentum: float = 0.999,
        feature_bank_size: int = 256,
        hierarchy_levels: int = 3,
        positive_instance_count: int = 3,
        negative_instance_count: int = 4,
        positive_prototype_count: int = 3,
        negative_prototype_count: int = 4,
        use_instance_loss: bool = True,
        instance_temperature: float | None = None,
        prototype_temperature: float | None = None,
        mask_outliers_at_base_level: bool = False,
        mask_outliers_at_upper_levels: bool = False,
        outlier_mask_mode: OutlierMaskMode = OutlierMaskMode.FARTHEST,
        outlier_distance_threshold: float = 0.3,
        outlier_mask_proportion: float = 0.5,
        replace_centroids_with_nearest_member: bool = False,
        learning_rate: float = 0.03,
        optimizer_momentum: float = 0.9,
        weight_decay: float = 1e-4,
        use_lr_scheduler: bool = True,
        lr_step_milestones: tuple[int, ...] | None = None,
        max_train_length: int | None = None,
        sync_dist: bool = False,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
        augmentation: "AugmentationProducer[ViewPair] | None" = None,
        singleton_split_count: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])

        self._input_dim = input_dim
        self._conv_kernel_size = conv_kernel_size
        self._stem_conv_kernel_size = stem_conv_kernel_size
        self._stem_conv_channels = stem_conv_channels
        self._encoder_stage_channels = encoder_stage_channels
        self._encoder_stage_depths = encoder_stage_depths
        self._encoder_stage_strides = encoder_stage_strides
        self._residual_block_type = residual_block_type
        self._projection_hidden_dim = projection_hidden_dim
        self._projection_dim = projection_dim
        self._use_projection_mlp = use_projection_mlp
        self._key_momentum = key_momentum
        self._feature_bank_size = feature_bank_size
        self._hierarchy_levels = hierarchy_levels
        self._positive_instance_count = positive_instance_count
        self._negative_instance_count = negative_instance_count
        self._positive_prototype_count = positive_prototype_count
        self._negative_prototype_count = negative_prototype_count
        self._use_instance_loss = use_instance_loss
        self._instance_temperature = instance_temperature
        self._prototype_temperature = prototype_temperature
        self._mask_outliers_at_base_level = mask_outliers_at_base_level
        self._mask_outliers_at_upper_levels = mask_outliers_at_upper_levels
        self._outlier_mask_mode = outlier_mask_mode
        self._outlier_distance_threshold = outlier_distance_threshold
        self._outlier_mask_proportion = outlier_mask_proportion
        self._replace_centroids_with_nearest_member = replace_centroids_with_nearest_member
        self._learning_rate = learning_rate
        self._optimizer_momentum = optimizer_momentum
        self._weight_decay = weight_decay
        self._use_lr_scheduler = use_lr_scheduler
        self._lr_step_milestones = lr_step_milestones
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist
        self._normalization_layer_type = normalization_layer_type

        if singleton_split_count < _MIN_SINGLETON_SPLIT_COUNT:
            msg = f"singleton_split_count must be >= 2, got {singleton_split_count}"
            raise ValueError(msg)
        self._singleton_split_count = singleton_split_count

        encoder_kwargs = {
            "input_dim": input_dim,
            "conv_kernel_size": conv_kernel_size,
            "stem_conv_kernel_size": stem_conv_kernel_size,
            "stem_conv_channels": stem_conv_channels,
            "encoder_stage_channels": encoder_stage_channels,
            "encoder_stage_depths": encoder_stage_depths,
            "encoder_stage_strides": encoder_stage_strides,
            "residual_block_type": residual_block_type,
            "normalization_layer_type": normalization_layer_type,
        }
        self._encoder = Conv1dResNetEncoder(**encoder_kwargs)  # type: ignore[arg-type]
        self._key_encoder = Conv1dResNetEncoder(**encoder_kwargs)  # type: ignore[arg-type]
        self._projection_head = self._build_projection_head()
        self._key_projection_head = self._build_projection_head()

        # The key branch is an EMA of the query branch, never an optimization
        # target. Both module pairs need the copy-and-freeze: the reference gets
        # its head for free because there the head is a submodule of the encoder.
        for online, momentum in self._momentum_module_pairs():
            for param_online, param_momentum in zip(
                online.parameters(), momentum.parameters(), strict=True
            ):
                param_momentum.data.copy_(param_online.data)
                param_momentum.requires_grad = False

        self.feature_bank: torch.Tensor
        self.feature_bank_fill: torch.Tensor
        self.feature_bank_index: torch.Tensor
        self.rows_this_epoch: torch.Tensor
        self.rows_last_epoch: torch.Tensor
        # DIVERGENCE: the bank starts empty rather than randomly initialized as
        # CoST's queue does. Random unit vectors would form spurious clusters
        # that real samples get assigned to; an empty bank simply degrades the
        # first steps to batch-only clustering.
        self.register_buffer("feature_bank", torch.zeros(max(feature_bank_size, 1), projection_dim))
        self.register_buffer("feature_bank_fill", torch.zeros((), dtype=torch.long))
        self.register_buffer("feature_bank_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("rows_this_epoch", torch.zeros((), dtype=torch.long))
        self.register_buffer("rows_last_epoch", torch.zeros((), dtype=torch.long))

        self._augmentation: AugmentationProducer[ViewPair] = (
            _default_mhccl_pair() if augmentation is None else augmentation
        )

    def _build_projection_head(self) -> nn.Module:
        """Build the head that maps backbone features into the contrast space."""
        width = self._encoder.representation_dim
        if not self._use_projection_mlp:
            return nn.Linear(width, self._projection_dim)
        return nn.Sequential(
            nn.Linear(width, self._projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._projection_hidden_dim, self._projection_dim),
        )

    def _momentum_module_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        """Return the (online, momentum) module pairs the EMA update spans."""
        return (
            (self._encoder, self._key_encoder),
            (self._projection_head, self._key_projection_head),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-norm query projections for ``x``.

        The backbone's temporal axis is averaged away before the head, so the
        projection is a fixed-width vector. That is required rather than
        incidental here: cluster centroids are means of these vectors and are
        compared against them, so the space cannot vary with input length. It is
        also what the reference computes, by pooling to ``(B, C, 1, 1)`` before
        its final linear layer.

        Args:
            x: Input batch of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Unit-norm projections of shape ``(batch, projection_dim)``.
        """
        return self._project(encoder=self._encoder, head=self._projection_head, x=x)

    def _project(self, *, encoder: nn.Module, head: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Encode, pool over time, project, and L2-normalize.

        Normalization is load-bearing: the reference's temperature-free logits
        are only in a usable range because both sides are unit-norm, and the
        Euclidean clustering that consumes them assumes the same.
        """
        features = encoder(x)  # (B, C, T')
        return functional.normalize(head(features.mean(dim=-1)), dim=-1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Move the key branch one EMA step toward the query branch.

        Only parameters are averaged, matching the reference; normalization
        running statistics are left to each branch, which matters solely under
        ``normalization_layer_type=BATCH``.
        """
        for online, momentum in self._momentum_module_pairs():
            for param_online, param_momentum in zip(
                online.parameters(), momentum.parameters(), strict=True
            ):
                param_momentum.data.mul_(self._key_momentum).add_(
                    param_online.data, alpha=1.0 - self._key_momentum
                )

    # ------------------------------------------------------------------
    # Feature bank
    # ------------------------------------------------------------------

    def _bank_rows(self) -> torch.Tensor:
        """Return the retained bank rows, capped at one epoch's worth."""
        fill = int(self.feature_bank_fill)
        if self._feature_bank_size == 0 or fill == 0:
            return self.feature_bank.new_zeros((0, self._projection_dim))
        # An epoch enqueues three rows per sample, so retaining more than that
        # means retaining copies of the same samples from older encoder states.
        # Those copies do not merely waste the pool: each sample's copies cluster
        # together at the finest partition and add a level encoding duplication
        # rather than structure, so an uncapped bank can look like a deeper
        # hierarchy while carrying less information. Pinned in
        # ``tests/unit/test_mhccl.py::test_duplicate_rows_manufacture_spurious_depth``.
        cap = int(self.rows_last_epoch)
        retained = min(fill, cap) if cap > 0 else fill
        offsets = torch.arange(retained, device=self.feature_bank.device)
        positions = (self.feature_bank_index - 1 - offsets) % self.feature_bank.size(0)
        return self.feature_bank[positions]

    @torch.no_grad()
    def _enqueue(self, rows: torch.Tensor) -> None:
        """Append rows to the circular bank, overwriting the oldest."""
        if self._feature_bank_size == 0:
            return
        size = self.feature_bank.size(0)
        newest = rows[-size:].detach()
        offsets = torch.arange(newest.size(0), device=rows.device)
        positions = (self.feature_bank_index + offsets) % size
        self.feature_bank[positions] = newest.to(self.feature_bank.dtype)
        self.feature_bank_index.copy_((self.feature_bank_index + newest.size(0)) % size)
        self.feature_bank_fill.copy_(torch.clamp(self.feature_bank_fill + newest.size(0), max=size))
        self.rows_this_epoch.add_(newest.size(0))

    def on_train_epoch_end(self) -> None:
        """Record how many rows one epoch enqueues, which caps the bank."""
        self.rows_last_epoch.copy_(self.rows_this_epoch)
        self.rows_this_epoch.zero_()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the multi-level contrastive loss over two views of the batch.

        Labels, if present in the batch, are ignored — this model handles
        self-supervised pretraining only.
        """
        data = extract_features_from_batch(batch).float()
        # Crop first: any later mask must describe the cropped tensor.
        data = process_sample_length(sample=data, max_sample_length=self._max_train_length)
        # NaN defense: zero-fill padded timesteps before augmentation, because
        # the augmentation primitives, Conv1d and cdist all propagate NaN, and a
        # NaN row would silently corrupt the whole partition.
        data, _ = zero_fill_padding(data)
        # Both terms are defined against the rest of the batch. At B == 1 the
        # instance term has no negatives and the clustering has nothing to
        # separate, so splitting the series into contiguous windows restores a
        # real signal. Unlike NT-Xent the gradient is not exactly zero here —
        # BCE against an all-ones target still moves the encoder — so this is a
        # signal-quality guard rather than a correctness one.
        data = ensure_pairable_batch(
            data,
            split_count=self._singleton_split_count,
            min_window_len=max(self._conv_kernel_size, self._stem_conv_kernel_size),
        )
        batch_size = data.size(0)

        pair = self._augmentation.produce(data)
        with torch.no_grad():
            if self.training:
                self._momentum_update_key_encoder()
            # One key forward over all three views. Under BATCH normalization
            # this pools statistics across the views rather than computing them
            # per view; under the CHANNEL default it is bit-identical.
            stacked = self._project(
                encoder=self._key_encoder,
                head=self._key_projection_head,
                x=torch.cat([data, pair.first, pair.second], dim=0),
            )
            # Blocks are [raw | first view | second view]; the key branch's
            # contribution to the loss is the second view's block.
            key = stacked[(_VIEWS_PER_SAMPLE - 1) * batch_size :]
            pool = torch.cat([stacked, self._bank_rows()], dim=0)

        hierarchy = self._cluster(pool)
        # The anchors are the raw-view block, exactly the rows the reference's
        # dataset index addresses.
        query = self(pair.first)

        instance_loss = masked_bce_with_logits(
            terms=instance_contrast_terms(
                query=query,
                key=key,
                assignment=hierarchy.assignments[0][:batch_size],
                positive_count=self._positive_instance_count,
                negative_count=self._negative_instance_count,
            ),
            temperature=self._instance_temperature,
        )

        prototype_loss, usable = self._prototype_loss(
            hierarchy=hierarchy, query=query, batch_size=batch_size, pool_size=pool.size(0)
        )

        if self.training:
            self._enqueue(stacked)

        self._log_components(
            instance_loss=instance_loss,
            prototype_loss=prototype_loss,
            levels_used=usable,
            level0_clusters=int(hierarchy.assignments[0].max()) + 1,
        )
        return (instance_loss if self._use_instance_loss else 0.0) + prototype_loss

    def _prototype_loss(
        self, *, hierarchy: "ClusterHierarchy", query: torch.Tensor, batch_size: int, pool_size: int
    ) -> tuple[torch.Tensor, int]:
        """Average cluster-level contrast over the partitions actually realized.

        Each contrasted partition needs the one above it to supply the sibling
        relation downward masking depends on, so the coarsest partition is never
        contrasted against and the usable count is bounded by ``len(parents)``.

        Args:
            hierarchy: The partition hierarchy for this batch.
            query: ``(B, D)`` unit-norm anchor projections.
            batch_size: Number of anchors, which are the pool's leading rows.
            pool_size: Rows the clustering saw, reported in the shortfall warning.

        Returns:
            The averaged loss and the number of partitions it covers.

        Raises:
            RuntimeError: If no partition is usable and the instance-level term
                is also disabled, leaving nothing to train on.
        """
        usable = min(self._hierarchy_levels, len(hierarchy.parents))
        # Hold the warning until the bank has seen a full epoch: during the
        # first one the pool is still filling, so a shallow hierarchy there says
        # nothing about the configuration.
        if usable < self._hierarchy_levels and int(self.rows_last_epoch) > 0:
            _warn_shallow_hierarchy(
                type(self), requested=self._hierarchy_levels, available=usable, pool_size=pool_size
            )
        if usable == 0 and not self._use_instance_loss:
            msg = (
                "MHCCL has no trainable term: the clustering realized no partition with a "
                "parent, so cluster-level contrast is unavailable, and use_instance_loss is "
                "False. Enable use_instance_loss or enlarge the batch or feature bank."
            )
            raise RuntimeError(msg)

        total = torch.zeros((), device=query.device)
        for level in range(usable):
            total = total + masked_bce_with_logits(
                terms=prototype_contrast_terms(
                    query=query,
                    assignment=hierarchy.assignments[level][:batch_size],
                    centroids=hierarchy.centroids[level],
                    parents=hierarchy.parents[level],
                    positive_count=self._positive_prototype_count,
                    negative_count=self._negative_prototype_count,
                ),
                temperature=self._prototype_temperature,
            )
        if usable == 0:
            return total, usable
        # DIVERGENCE: averaged over the levels actually used, not over
        # ``hierarchy_levels``. Dividing by the configured count would shrink the
        # loss whenever the hierarchy is shallow, tying the objective's scale to
        # the batch's geometry.
        return total / usable, usable

    def _cluster(self, pool: torch.Tensor) -> "ClusterHierarchy":
        """Build the partition hierarchy the two contrastive terms consume."""
        return finch(
            pool,
            # One extra partition beyond those contrasted against: the topmost
            # supplies the sibling relation for the level below and is not
            # itself used.
            max_levels=self._hierarchy_levels + 1,
            mask_base_level=self._mask_outliers_at_base_level,
            mask_upper_levels=self._mask_outliers_at_upper_levels,
            mask_mode=self._outlier_mask_mode,
            distance_threshold=self._outlier_distance_threshold,
            mask_proportion=self._outlier_mask_proportion,
            replace_centroids_with_nearest_member=self._replace_centroids_with_nearest_member,
        )

    def _log_components(
        self,
        *,
        instance_loss: torch.Tensor,
        prototype_loss: torch.Tensor,
        levels_used: int,
        level0_clusters: int,
    ) -> None:
        """Log the two terms and the clustering's realized shape."""
        if getattr(self, "_trainer", None) is None:
            return
        self.log_dict(
            {
                "mhccl/instance_loss": instance_loss,
                "mhccl/prototype_loss": prototype_loss,
                "mhccl/hierarchy_levels_used": float(levels_used),
                "mhccl/level0_clusters": float(level0_clusters),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=self._sync_dist,
        )

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
        """Compute and log the validation loss for one batch.

        Neither the momentum update nor the feature bank is touched here, so
        validation cannot move training state.
        """
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
        """Return SGD with the reference's cosine or step learning-rate decay."""
        # The key branch is frozen, so it has no place in the optimizer. The
        # reference passes ``model.parameters()``, which includes it; harmless,
        # since it receives no gradients, but filtered here as CoST does.
        optimizer = torch.optim.SGD(
            [param for param in self.parameters() if param.requires_grad],
            lr=self._learning_rate,
            momentum=self._optimizer_momentum,
            weight_decay=self._weight_decay,
        )
        if not self._use_lr_scheduler:
            return optimizer

        if self._lr_step_milestones is not None:
            # The reference's non-``--cos`` branch: multiply by 0.1 at each
            # milestone. Needs no horizon, so it survives without a trainer.
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=list(self._lr_step_milestones), gamma=_LR_STEP_GAMMA
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        # `self.trainer` raises when the module is used outside a Trainer, and
        # max_epochs is None or -1 for step-bounded or open-ended runs. Cosine
        # annealing needs a horizon, so without one the rate stays constant.
        trainer = getattr(self, "_trainer", None)
        max_epochs = getattr(trainer, "max_epochs", None)
        if not max_epochs or max_epochs < 0:
            return optimizer

        # The reference's ``--cos`` branch computes
        # ``lr * 0.5 * (1 + cos(pi * epoch / epochs))`` by hand, which is
        # exactly CosineAnnealingLR stepped once per epoch.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=0.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Representation extraction (via BasicEncodingMixin.encode)
    # ------------------------------------------------------------------

    def _get_encoder(self) -> nn.Module:
        """Expose the query backbone (before the projection head) to ``encode()``."""
        return self._encoder

    @property
    def encoder(self) -> nn.Module:
        """Return the query ResNet-1D backbone for inspection and checkpointing."""
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
        """Reduce the backbone feature map to the requested output shape.

        Returns backbone features, never projections: the projection head shapes
        the contrastive objective and the clustering space, and is discarded
        downstream.

        Args:
            encoder: The backbone returned by :meth:`_get_encoder`.
            batch_x: Batch of shape ``(batch, seq_len, input_dim)``, already on
                the model's device.
            output: Requested output shape.

        Returns:
            ``(B, representation_dim)`` for VECTOR, or
            ``(B, reduced_len, representation_dim)`` for SEQUENCE.

        Raises:
            ValueError: If ``output`` is not a supported shape.
        """
        batch_x, _ = zero_fill_padding(batch_x)
        features = encoder(batch_x.float())  # (B, C, T')
        if output == EncodingOutputShape.VECTOR:
            return features.mean(dim=-1)  # (B, C) — VECTOR
        if output == EncodingOutputShape.SEQUENCE:
            return features.transpose(1, 2)  # (B, T', C) — SEQUENCE
        msg = f"MHCCL does not support output={output}; supported: {type(self).supported_outputs}"
        raise ValueError(msg)
