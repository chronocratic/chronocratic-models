"""Conv1D residual blocks used by the SimCLR ResNet encoder.

The reference implementation (ULTS ``models/SimCLR/models.py``) builds **2-D**
blocks and feeds them ``(B, C, T, 1)`` — the time series is unsqueezed into an
image of width 1. Every ``Conv2d(k=3, padding=1)`` in that stack therefore sees
a width axis of extent 1, so the left and right kernel columns only ever
multiply zero padding: they are dead weights, and the stack computes exactly
what the equivalent ``Conv1d(k=3, padding=1)`` computes with the centre column.
These are that ``Conv1d`` form — numerically identical to the reference
(verified at stride 1 and stride 2, and for the normalization layer, in
``tests/unit/test_simclr.py``) with a third of the parameters per convolution
and none of the width-axis bookkeeping.

Kept beside the model rather than in a shared layers package: SimCLR is
currently their only consumer. Promote them if a second model needs them.
"""

from __future__ import annotations

__all__ = ["Conv1dBasicBlock", "Conv1dBottleneckBlock"]

import torch
from torch import nn

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
