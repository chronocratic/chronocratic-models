"""Configuration for the MHCCL model.

Provides MHCCLModelParameters with all settings for the ResNet-1D encoder pair,
the projection head, the FINCH clustering hierarchy, and the two contrastive
terms.
"""

__all__ = ["MHCCLModelParameters"]

from dataclasses import dataclass

from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.clustering import OutlierMaskMode
from chronocratic.models.enums.layers import NormalizationLayerType

# Two windows is the smallest split that yields a non-degenerate batch
# (exactly one negative pair).
_MIN_SINGLETON_SPLIT_COUNT = 2
# One partition supplies prototypes; the level above it supplies the sibling
# relation downward masking needs, and is counted separately.
_MIN_HIERARCHY_LEVELS = 1
# The anchor's own prototype is always positive, so at least one is required.
_MIN_POSITIVE_COUNT = 1


@dataclass(kw_only=True)
class MHCCLModelParameters:
    """Configuration for the MHCCL model.

    Defaults are sourced from the reference implementation's code, not from the
    MHCCL paper: ``main.py``'s argparse for the training and contrast settings,
    and ``framework.py``'s ``ResNet18`` for the architecture. Where the CLI
    default and the README's published command disagree, the published command
    wins, since it is the only complete configuration the authors give;
    ``use_projection_mlp`` and ``use_lr_scheduler`` are the two such fields.

    Args:
        input_dim: Number of input features (channels) in the time series.
        conv_kernel_size: Kernel size of every residual convolution that is not
            a 1-tap bottleneck projection. Defaults to ``3``.
        stem_conv_kernel_size: Kernel size of the stem convolution, which the
            reference sets wider than the residual convolutions. Defaults to
            ``8``.
        stem_conv_channels: Number of channels produced by the stem
            convolution. Defaults to ``64``.
        encoder_stage_channels: Width of each residual stage, one entry per
            stage. Defaults to ``(64, 128, 256, 512)``.
        encoder_stage_depths: Number of residual blocks per stage. With
            ``BASIC`` blocks the default gives a ResNet-18, which is what the
            reference builds. Defaults to ``(2, 2, 2, 2)``.
        encoder_stage_strides: Temporal stride applied by each stage. Defaults
            to ``(1, 2, 2, 2)``.
        residual_block_type: ``BASIC`` (``expansion = 1``) or ``BOTTLENECK``
            (``expansion = 4``). Defaults to ``BASIC``.
        projection_hidden_dim: Hidden width of the projection head, used only
            when ``use_projection_mlp`` is set. Defaults to ``512``.
        projection_dim: Width of the projected space the contrastive terms and
            the clustering both operate in. Defaults to ``128``.
        use_projection_mlp: Whether the projection head is a two-layer MLP
            rather than a single linear map. Defaults to ``True``, following the
            reference's published command; its CLI default is ``False``.
        key_momentum: EMA coefficient for the momentum (key) encoder. Higher
            values keep the key encoder more stable across steps. Defaults to
            ``0.999``.
        feature_bank_size: Upper bound on the number of momentum-encoder
            feature rows retained from previous steps and clustered alongside
            the current batch. Bounded further at runtime so the bank never
            spans more than one epoch, because copies of the same sample from
            different encoder states would otherwise dominate the finest
            partition. ``0`` clusters the current batch alone. Defaults to
            ``256``.
        hierarchy_levels: Number of clustering partitions to contrast against.
            Each needs the partition above it to supply the sibling relation, so
            the number actually used is bounded by what the clustering realizes
            and is logged as ``mhccl/hierarchy_levels_used``. Defaults to ``3``.
        positive_instance_count: Positive instances per anchor in the
            instance-level term. Defaults to ``3``.
        negative_instance_count: Negative instances per anchor in the
            instance-level term. Defaults to ``4``.
        positive_prototype_count: Positive prototypes per anchor per partition,
            including the anchor's own. Defaults to ``3``.
        negative_prototype_count: Negative prototypes per anchor per partition.
            Defaults to ``4``.
        use_instance_loss: Whether to include the instance-level term. Setting
            it ``False`` reproduces the reference's ``--protoNCE_only``.
            Defaults to ``True``.
        instance_temperature: Divisor for the instance-level logits. ``None``
            (the default) leaves them as raw inner products, which is what the
            reference's published configuration does.
        prototype_temperature: Divisor for the cluster-level logits. ``None``
            by default, for the same reason as ``instance_temperature``.
        mask_outliers_at_base_level: Whether upward masking refines the finest
            partition's prototypes. Defaults to ``False``.
        mask_outliers_at_upper_levels: Whether upward masking refines every
            coarser partition's prototypes. Defaults to ``False``.
        outlier_mask_mode: Which cluster members upward masking excludes.
            Defaults to ``FARTHEST``.
        outlier_distance_threshold: Absolute distance beyond which a member is
            an outlier, for ``OutlierMaskMode.THRESHOLD``. Defaults to ``0.3``.
        outlier_mask_proportion: Fraction of each cluster to exclude, for
            ``OutlierMaskMode.PROPORTION``. Defaults to ``0.5``.
        replace_centroids_with_nearest_member: Whether each prototype is
            replaced by the real member closest to it. Defaults to ``False``.
        learning_rate: Learning rate for the SGD optimizer. Defaults to
            ``0.03``.
        optimizer_momentum: Momentum for the SGD optimizer. Distinct from
            ``key_momentum``, which governs the momentum encoder. Defaults to
            ``0.9``.
        weight_decay: Weight decay for the SGD optimizer. Defaults to ``1e-4``.
        use_lr_scheduler: Whether to decay the learning rate over training.
            Defaults to ``True``, following the reference's published command;
            its CLI default is ``False``.
        lr_step_milestones: Epochs at which to multiply the learning rate by
            ``0.1``. ``None`` (the default) cosine-anneals to zero instead,
            which is the reference's ``--cos`` branch.
        max_train_length: Random-crop length applied to training batches only.
            ``None`` (the default) disables cropping. ``encode()`` never crops.
        sync_dist: Whether to synchronize logged metrics across distributed
            processes.
        normalization_layer_type: Normalization strategy for the encoders and
            the projection head. ``CHANNEL`` (default) uses GroupNorm(1, C),
            which is batch-size independent. ``BATCH`` uses BatchNorm1d and
            reproduces the reference.
        augmentation: Custom augmentation producer. Defaults to ``None``, which
            builds the reference weak/strong pair at model init.
        singleton_split_count: Number of contiguous windows to split a
            singleton batch into. Defaults to ``3``.
    """

    input_dim: int
    conv_kernel_size: int = 3
    stem_conv_kernel_size: int = 8
    stem_conv_channels: int = 64
    encoder_stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    encoder_stage_depths: tuple[int, ...] = (2, 2, 2, 2)
    encoder_stage_strides: tuple[int, ...] = (1, 2, 2, 2)
    residual_block_type: ResidualBlockType = ResidualBlockType.BASIC
    projection_hidden_dim: int = 512
    projection_dim: int = 128
    use_projection_mlp: bool = True
    key_momentum: float = 0.999
    feature_bank_size: int = 256
    hierarchy_levels: int = 3
    positive_instance_count: int = 3
    negative_instance_count: int = 4
    positive_prototype_count: int = 3
    negative_prototype_count: int = 4
    use_instance_loss: bool = True
    instance_temperature: float | None = None
    prototype_temperature: float | None = None
    mask_outliers_at_base_level: bool = False
    mask_outliers_at_upper_levels: bool = False
    outlier_mask_mode: OutlierMaskMode = OutlierMaskMode.FARTHEST
    outlier_distance_threshold: float = 0.3
    outlier_mask_proportion: float = 0.5
    replace_centroids_with_nearest_member: bool = False
    learning_rate: float = 0.03
    optimizer_momentum: float = 0.9
    weight_decay: float = 1e-4
    use_lr_scheduler: bool = True
    lr_step_milestones: tuple[int, ...] | None = None
    max_train_length: int | None = None
    sync_dist: bool = False
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL
    augmentation: AugmentationProducer[ViewPair] | None = None
    singleton_split_count: int = 3

    def __post_init__(self) -> None:
        """Validate numeric constraints after construction."""
        self._validate_architecture()
        self._validate_clustering()
        self._validate_contrast()
        self._validate_optimization()

    def _validate_architecture(self) -> None:
        """Check the encoder and projection-head dimensions."""
        positive_dims = {
            "input_dim": self.input_dim,
            "conv_kernel_size": self.conv_kernel_size,
            "stem_conv_kernel_size": self.stem_conv_kernel_size,
            "stem_conv_channels": self.stem_conv_channels,
            "projection_dim": self.projection_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
        }
        for name, value in positive_dims.items():
            if value <= 0:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)

    def _validate_clustering(self) -> None:
        """Check the hierarchy, feature bank, and upward-masking settings."""
        if self.hierarchy_levels < _MIN_HIERARCHY_LEVELS:
            msg = f"hierarchy_levels must be at least 1, got {self.hierarchy_levels}"
            raise ValueError(msg)
        if self.feature_bank_size < 0:
            msg = f"feature_bank_size must be non-negative, got {self.feature_bank_size}"
            raise ValueError(msg)
        if not 0.0 < self.outlier_mask_proportion <= 1.0:
            msg = f"outlier_mask_proportion must lie in (0, 1], got {self.outlier_mask_proportion}"
            raise ValueError(msg)
        if self.outlier_distance_threshold <= 0.0:
            msg = (
                f"outlier_distance_threshold must be positive, "
                f"got {self.outlier_distance_threshold}"
            )
            raise ValueError(msg)

    def _validate_contrast(self) -> None:
        """Check the pair counts and temperatures of the two contrastive terms."""
        positives = {
            "positive_instance_count": self.positive_instance_count,
            "positive_prototype_count": self.positive_prototype_count,
        }
        for name, value in positives.items():
            if value < _MIN_POSITIVE_COUNT:
                msg = f"{name} must be at least 1, got {value}"
                raise ValueError(msg)
        negatives = {
            "negative_instance_count": self.negative_instance_count,
            "negative_prototype_count": self.negative_prototype_count,
        }
        for name, value in negatives.items():
            if value < 0:
                msg = f"{name} must be non-negative, got {value}"
                raise ValueError(msg)
        temperatures = {
            "instance_temperature": self.instance_temperature,
            "prototype_temperature": self.prototype_temperature,
        }
        for name, value in temperatures.items():
            if value is not None and value <= 0.0:
                msg = f"{name} must be positive when set, got {value}"
                raise ValueError(msg)

    def _validate_optimization(self) -> None:
        """Check the momentum coefficients, batch guard, and crop length."""
        if not 0.0 <= self.key_momentum < 1.0:
            msg = f"key_momentum must lie in [0, 1), got {self.key_momentum}"
            raise ValueError(msg)
        if self.singleton_split_count < _MIN_SINGLETON_SPLIT_COUNT:
            msg = f"singleton_split_count must be >= 2, got {self.singleton_split_count}"
            raise ValueError(msg)
        if self.max_train_length is not None and self.max_train_length <= 0:
            msg = f"max_train_length must be positive when set, got {self.max_train_length}"
            raise ValueError(msg)
