"""ResNet-style Conv1D residual blocks and encoder.

A 1-D port of the ResNet blocks and stage-stacking encoder, shared by
standard-convolution models that need a residual backbone. :class:`SimCLR` is
the current consumer.

The reference implementation this is ported from (ULTS
``models/SimCLR/models.py::ResNet_CIFAR``) builds a **2-D** ResNet and feeds it
``(B, C, T, 1)`` — the time series is unsqueezed into an image of width 1.
Every ``Conv2d(k=3, padding=1)`` in that stack therefore sees a width axis of
extent 1, so the left and right kernel columns only ever multiply zero padding:
they are dead weights, and the stack computes exactly what the equivalent
``Conv1d(k=3, padding=1)`` computes with the centre column. This module is that
``Conv1d`` stack. It is numerically identical to the reference (verified at
stride 1 and stride 2) with a third of the parameters per convolution and none
of the width-axis bookkeeping.

See :class:`Conv1dResNetEncoder` for the remaining deliberate divergences.
"""

from __future__ import annotations

__all__ = ["Conv1dBasicBlock", "Conv1dBottleneckBlock", "Conv1dResNetEncoder"]

import torch
from torch import nn

from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.layers import NormalizationLayerType


def _norm_layer(
    *, num_channels: int, normalization_layer_type: NormalizationLayerType
) -> nn.Module:
    """Build the configured normalization layer for ``num_channels`` channels.

    Args:
        num_channels: Number of channels the layer normalizes.
        normalization_layer_type: ``CHANNEL`` for GroupNorm(1, C) (batch-size
            independent) or ``BATCH`` for BatchNorm1d.

    Returns:
        The instantiated normalization module.
    """
    # The reference hardcodes ``BatchNorm2d`` throughout. Two things differ
    # here; only the second is a divergence.
    #
    # ``BATCH`` instantiates ``BatchNorm1d`` rather than ``BatchNorm2d``. This
    # is a consequence of the 1-D port, not a change in behaviour: on the
    # reference's width-1 axis the two are the same operation. ``BatchNorm2d``
    # over ``(B, C, T, 1)`` accumulates its per-channel statistics over
    # ``B * T * 1`` elements and ``BatchNorm1d`` over ``(B, C, T)`` over
    # ``B * T`` — the same element set, hence the same mean, variance, affine
    # output, and running statistics. Asserted in both training and evaluation
    # mode in ``tests/unit/test_simclr.py``. The equality is mathematical
    # rather than bitwise: the two kernels reduce in different orders, so the
    # trailing float32 bits depend on the input's memory layout.
    #
    # DIVERGENCE: the default is ``CHANNEL`` (``GroupNorm(1, C)``) rather than
    # the reference's ``BATCH``. GroupNorm normalizes each sample from its own
    # statistics, so it remains well-defined at ``batch_size=1`` where
    # BatchNorm's batch statistics are degenerate, and it matches every other
    # standard-convolution encoder in this library. Pass ``BATCH`` to
    # reproduce the reference exactly.
    if normalization_layer_type == NormalizationLayerType.CHANNEL:
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    return nn.BatchNorm1d(num_channels)


class Conv1dBasicBlock(nn.Module):
    """Two-convolution residual block (``expansion = 1``).

    The 1-D counterpart of the reference's ``BasicBlock_CIFAR``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of channels produced by each convolution. The
            block's output width is ``out_channels * expansion``.
        conv_kernel_size: Kernel size of both convolutions.
        stride: Stride of the first convolution (and of the shortcut).
        normalization_layer_type: Normalization strategy for both branches.
    """

    expansion: int = 1

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        stride: int = 1,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        padding = conv_kernel_size // 2
        self._conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self._norm1 = _norm_layer(
            num_channels=out_channels, normalization_layer_type=normalization_layer_type
        )
        self._conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self._norm2 = _norm_layer(
            num_channels=out_channels, normalization_layer_type=normalization_layer_type
        )
        self._shortcut = _build_shortcut(
            in_channels=in_channels,
            out_channels=out_channels * self.expansion,
            stride=stride,
            normalization_layer_type=normalization_layer_type,
        )
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to a ``(B, C, T)`` feature map."""
        out = self._relu(self._norm1(self._conv1(x)))
        out = self._norm2(self._conv2(out))
        return self._relu(out + self._shortcut(x))


class Conv1dBottleneckBlock(nn.Module):
    """Channel-bottleneck residual block (``expansion = 4``).

    The 1-D counterpart of the reference's ``Bottleneck_CIFAR``: a 1-tap
    channel reduction, a ``conv_kernel_size``-tap convolution, then a 1-tap
    expansion back to ``out_channels * expansion``.

    Args:
        in_channels: Number of input channels.
        out_channels: Bottleneck width. The block's output width is
            ``out_channels * expansion``.
        conv_kernel_size: Kernel size of the middle convolution.
        stride: Stride of the middle convolution (and of the shortcut).
        normalization_layer_type: Normalization strategy for both branches.
    """

    expansion: int = 4

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        stride: int = 1,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        expanded = out_channels * self.expansion
        self._conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self._norm1 = _norm_layer(
            num_channels=out_channels, normalization_layer_type=normalization_layer_type
        )
        self._conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=conv_kernel_size // 2,
            bias=False,
        )
        self._norm2 = _norm_layer(
            num_channels=out_channels, normalization_layer_type=normalization_layer_type
        )
        self._conv3 = nn.Conv1d(out_channels, expanded, kernel_size=1, bias=False)
        self._norm3 = _norm_layer(
            num_channels=expanded, normalization_layer_type=normalization_layer_type
        )
        self._shortcut = _build_shortcut(
            in_channels=in_channels,
            out_channels=expanded,
            stride=stride,
            normalization_layer_type=normalization_layer_type,
        )
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to a ``(B, C, T)`` feature map."""
        out = self._relu(self._norm1(self._conv1(x)))
        out = self._relu(self._norm2(self._conv2(out)))
        out = self._norm3(self._conv3(out))
        return self._relu(out + self._shortcut(x))


def _build_shortcut(
    *,
    in_channels: int,
    out_channels: int,
    stride: int,
    normalization_layer_type: NormalizationLayerType,
) -> nn.Module:
    """Return the residual shortcut: identity, or a projection when shapes differ.

    Args:
        in_channels: Channels entering the block.
        out_channels: Channels leaving the block (already expansion-scaled).
        stride: Temporal stride the main branch applies.
        normalization_layer_type: Normalization strategy for the projection.

    Returns:
        ``nn.Identity`` when the shapes already match, otherwise a strided
        1-tap convolution followed by normalization.
    """
    if stride == 1 and in_channels == out_channels:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        _norm_layer(num_channels=out_channels, normalization_layer_type=normalization_layer_type),
    )


_BLOCKS: dict[ResidualBlockType, type[Conv1dBasicBlock | Conv1dBottleneckBlock]] = {
    ResidualBlockType.BASIC: Conv1dBasicBlock,
    ResidualBlockType.BOTTLENECK: Conv1dBottleneckBlock,
}


class Conv1dResNetEncoder(nn.Module):
    """ResNet backbone over the time axis, returning a pooled flat vector.

    Layout: a stem convolution, then one residual stage per entry of
    ``encoder_stage_depths``, then global average pooling over time. The output is
    ``(B, encoder_stage_channels[-1] * block.expansion)``.

    Length-agnostic by construction — every layer is a convolution, a
    normalization, or a pooling with a computed kernel, so the encoder accepts
    any sequence length without a resampling patch.

    Deliberate divergences from the reference (ULTS ``ResNet_CIFAR``):

    - **1-D convolutions instead of 2-D over a width-1 axis.** Numerically
      identical; see the module docstring.
    - **``in_channels`` comes from ``input_dim``, not from a dataset name.**
      The reference selects it with an ``if dataset_name == 'HAR': ...`` chain
      that has no ``else``, so any unlisted dataset raises
      ``UnboundLocalError`` before the first forward pass.
    - **The fourth stage uses its own block count.** The reference builds its
      stages with ``layers[0], layers[1], layers[2], layers[2]`` — the last
      index is a typo, so ``resnet50_cifar`` is handed ``[3, 4, 6, 3]`` and
      builds ``[3, 4, 6, 6]``. ``encoder_stage_depths`` is honoured entry by entry,
      which restores the ResNet-50 stage pattern the reference intended.
    - **No unused projection.** The reference constructs
      ``self.linear = nn.Linear(512 * expansion, num_features)`` and never
      calls it.
    - **Normalization is configurable and defaults to GroupNorm(1, C).** The
      reference hardcodes ``BatchNorm2d``. ``BATCH`` reproduces the reference
      behaviour; the default matches every other standard-convolution encoder
      in this library and stays well-defined at ``batch_size=1``.

    Args:
        input_dim: Number of input features (channels).
        conv_kernel_size: Kernel size of the stem and of every residual
            convolution that is not a 1-tap bottleneck projection.
        stem_conv_channels: Channel count produced by the stem convolution.
        encoder_stage_channels: Per-stage channel counts. One entry per residual
            stage.
        encoder_stage_depths: Number of residual blocks per stage. Must be the same
            length as ``encoder_stage_channels``.
        encoder_stage_strides: Temporal stride of the first block of each stage.
            Must be the same length as ``encoder_stage_channels``.
        residual_block_type: ``BASIC`` (``expansion = 1``) or ``BOTTLENECK``
            (``expansion = 4``).
        normalization_layer_type: ``CHANNEL`` for GroupNorm(1, C) or ``BATCH``
            for BatchNorm1d.
    """

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
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()
        if not encoder_stage_channels:
            msg = "encoder_stage_channels must contain at least one stage"
            raise ValueError(msg)
        if len(encoder_stage_depths) != len(encoder_stage_channels):
            msg = (
                f"encoder_stage_depths must have one entry per stage: got "
                f"{len(encoder_stage_depths)} for {len(encoder_stage_channels)} stages"
            )
            raise ValueError(msg)
        if len(encoder_stage_strides) != len(encoder_stage_channels):
            msg = (
                f"encoder_stage_strides must have one entry per stage: got "
                f"{len(encoder_stage_strides)} for {len(encoder_stage_channels)} stages"
            )
            raise ValueError(msg)

        block_cls = _BLOCKS[ResidualBlockType(residual_block_type)]
        self._representation_dim = encoder_stage_channels[-1] * block_cls.expansion

        self._stem = nn.Sequential(
            nn.Conv1d(
                input_dim,
                stem_conv_channels,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=conv_kernel_size // 2,
                bias=False,
            ),
            _norm_layer(
                num_channels=stem_conv_channels, normalization_layer_type=normalization_layer_type
            ),
            nn.ReLU(),
        )

        # DIVERGENCE: one stage per entry of ``encoder_stage_depths``, read
        # positionally. The reference builds its four stages from
        # ``layers[0]``, ``layers[1]``, ``layers[2]``, ``layers[2]``, repeating
        # the third index in place of the fourth. Its ResNet-50 configuration
        # is therefore constructed with stage depths ``[3, 4, 6, 6]`` rather
        # than the ``[3, 4, 6, 3]`` it is given, and the fourth stage's depth
        # cannot be configured at all. Reading the tuple positionally restores
        # the requested depths.
        stages: list[nn.Module] = []
        in_channels = stem_conv_channels
        for channels, num_blocks, stride in zip(
            encoder_stage_channels, encoder_stage_depths, encoder_stage_strides, strict=True
        ):
            blocks: list[nn.Module] = []
            for block_index in range(num_blocks):
                blocks.append(
                    block_cls(
                        in_channels=in_channels,
                        out_channels=channels,
                        conv_kernel_size=conv_kernel_size,
                        stride=stride if block_index == 0 else 1,
                        normalization_layer_type=normalization_layer_type,
                    )
                )
                in_channels = channels * block_cls.expansion
            stages.append(nn.Sequential(*blocks))
        self._stages = nn.Sequential(*stages)

        # Global average pooling over time. The reference reaches the same
        # value the long way round -- it tiles the width-1 axis to length T'
        # and then applies avg_pool2d with kernel T', which averages T'
        # identical columns. Verified equal to a mean over the time axis.
        self._pool = nn.AdaptiveAvgPool1d(1)

    @property
    def representation_dim(self) -> int:
        """Width of the pooled feature vector this encoder returns."""
        return self._representation_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into flat pooled representations.

        Args:
            x: Input batch of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Representations of shape ``(batch, representation_dim)``.
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self._stem(x)
        x = self._stages(x)
        return self._pool(x).flatten(1)
