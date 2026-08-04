"""Configuration for the SimCLR model.

Provides SimCLRModelParameters with all settings for the ResNet-1D encoder,
the projection head, and the NT-Xent contrastive objective.
"""

__all__ = ["SimCLRModelParameters"]

from dataclasses import dataclass

from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.layers import NormalizationLayerType


@dataclass(kw_only=True)
class SimCLRModelParameters:
    """Configuration for the SimCLR model.

    Defaults are sourced from the reference implementation's code, not from
    the SimCLR paper: ``ULTS/main.py`` for the training hyperparameters and
    ``ULTS/models/SimCLR/models.py`` for the architecture. The active model
    line there is ``resnet50_cifar``, hence the Bottleneck default.

    Args:
        input_dim: Number of input features (channels) in the time series.
        conv_kernel_size: Kernel size of the stem convolution and of every
            residual convolution that is not a 1-tap bottleneck projection.
            Defaults to ``3``.
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
            Defaults to ``BOTTLENECK``.
        projection_hidden_dim: Hidden width of the two-layer projection head.
            Defaults to ``512``.
        projection_dim: Output width of the projection head — the space the
            NT-Xent loss operates in. Never used downstream; ``encode()``
            returns encoder features. Defaults to ``128``.
        temperature: Temperature of the NT-Xent loss. Defaults to ``0.5``.
        learning_rate: Learning rate for the Adam optimizer. Defaults to
            ``1e-3``.
        weight_decay: Weight decay for the Adam optimizer. Defaults to
            ``1e-6``.
        use_lr_scheduler: Whether to decay the learning rate over training
            with linear warmup followed by cosine annealing to zero, the
            schedule SimCLR is specified with. Defaults to ``True``. Set to
            ``False`` to train at a constant ``learning_rate``, which makes
            the final weights depend on where the optimizer happens to be
            when the epoch budget runs out; see ``configure_optimizers``.
        warmup_epochs: Number of epochs over which the learning rate ramps
            linearly from zero to ``learning_rate`` before cosine annealing
            begins. Clamped to at most half of the trainer's ``max_epochs``,
            so short runs still spend most of their budget annealing.
            Ignored when ``use_lr_scheduler`` is ``False``. Defaults to
            ``10``.
        max_train_length: Random-crop length applied to training batches
            only. ``None`` (the default) disables cropping.
        sync_dist: Whether to synchronize logged metrics across distributed
            processes.
        normalization_layer_type: Normalization strategy for the encoder and
            the projection head. ``CHANNEL`` (default) uses GroupNorm(1, C),
            which is batch-size independent. ``BATCH`` uses BatchNorm1d and
            reproduces the reference.
        augmentation: Custom augmentation producer. Defaults to ``None``,
            which builds the reference weak/strong pair at model init.
    """

    input_dim: int
    conv_kernel_size: int = 3
    stem_conv_channels: int = 64
    encoder_stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    encoder_stage_depths: tuple[int, ...] = (3, 4, 6, 3)
    encoder_stage_strides: tuple[int, ...] = (1, 2, 2, 2)
    residual_block_type: ResidualBlockType = ResidualBlockType.BOTTLENECK
    projection_hidden_dim: int = 512
    projection_dim: int = 128
    temperature: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    use_lr_scheduler: bool = True
    warmup_epochs: int = 10
    max_train_length: int | None = None
    sync_dist: bool = False
    normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL
    augmentation: AugmentationProducer[ViewPair] | None = None

    def __post_init__(self) -> None:
        """Validate numeric constraints after construction."""
        if self.input_dim <= 0:
            msg = f"input_dim must be positive, got {self.input_dim}"
            raise ValueError(msg)
        if self.conv_kernel_size <= 0:
            msg = f"conv_kernel_size must be positive, got {self.conv_kernel_size}"
            raise ValueError(msg)
        if self.stem_conv_channels <= 0:
            msg = f"stem_conv_channels must be positive, got {self.stem_conv_channels}"
            raise ValueError(msg)
        if self.projection_dim <= 0:
            msg = f"projection_dim must be positive, got {self.projection_dim}"
            raise ValueError(msg)
        if self.projection_hidden_dim <= 0:
            msg = f"projection_hidden_dim must be positive, got {self.projection_hidden_dim}"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = f"temperature must be positive, got {self.temperature}"
            raise ValueError(msg)
        if self.warmup_epochs < 0:
            msg = f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            raise ValueError(msg)
        if self.max_train_length is not None and self.max_train_length <= 0:
            msg = f"max_train_length must be positive when set, got {self.max_train_length}"
            raise ValueError(msg)
