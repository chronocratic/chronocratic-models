"""ResNet-style Conv1D encoder backbone for SimCLR.

Stacks the residual blocks from
:mod:`~chronocratic.models.convolutional.standard.simclr.layers` into strided
stages and returns the resulting feature map ``(B, representation_dim, T')``.

The encoder deliberately does not pool over time. Reducing the temporal axis
is the caller's decision, because SimCLR's two callers need different answers:
the contrastive loss needs the temporal axis preserved (pooling it away leaves
NT-Xent with no angular diversity and pins the loss at ``log(K)``), while
``encode()`` needs a fixed-width vector for downstream probes. See
:meth:`~chronocratic.models.convolutional.standard.simclr.model.SimCLR._encode_batch`,
which owns that reduction.

See :class:`Conv1dResNetEncoder` for the deliberate divergences from the
reference.
"""

from __future__ import annotations

__all__ = ["Conv1dResNetEncoder"]

import torch
from torch import nn

from chronocratic.models.convolutional.standard.simclr.layers import (
    _norm_layer,
    Conv1dBasicBlock,
    Conv1dBottleneckBlock,
)
from chronocratic.models.enums.blocks import ResidualBlockType
from chronocratic.models.enums.layers import NormalizationLayerType

_BLOCKS: dict[ResidualBlockType, type[Conv1dBasicBlock | Conv1dBottleneckBlock]] = {
    ResidualBlockType.BASIC: Conv1dBasicBlock,
    ResidualBlockType.BOTTLENECK: Conv1dBottleneckBlock,
}


class Conv1dResNetEncoder(nn.Module):
    """ResNet backbone over the time axis, returning a feature map.

    Layout: a stem convolution, then one residual stage per entry of
    ``encoder_stage_depths``. The output is
    ``(B, encoder_stage_channels[-1] * block.expansion, T')``, where ``T'`` is
    the input length after the configured strides. No pooling — the caller
    reduces the temporal axis as it requires.

    Length-agnostic by construction — every layer is a convolution or a
    normalization, so the encoder accepts any sequence length without a
    resampling patch. ``representation_dim`` describes the channel width and is
    independent of ``T'``.

    Deliberate divergences from the reference (ULTS ``ResNet_CIFAR``):

    - **No temporal pooling, matching the reference.** ULTS applies
      ``x.repeat(1, 1, 1, x.size(2))`` and then ``F.avg_pool2d(x, x.size(2))``
      to a tensor whose width axis is 1: it replicates that axis ``T`` times and
      averages the identical copies, which is the identity. It then flattens
      ``C x T``. The temporal axis is preserved here and reduced by the caller.
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

    @property
    def representation_dim(self) -> int:
        """Channel width of the feature map this encoder returns."""
        return self._representation_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into a convolutional feature map.

        Deliberately unpooled. Callers reduce the temporal axis themselves:
        :meth:`SimCLR._encode_batch` averages it for VECTOR and transposes it
        for SEQUENCE, while the contrastive loss path keeps it, because
        averaging it away leaves NT-Xent without the angular diversity it needs
        to produce a gradient.

        Args:
            x: Input batch of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Feature map of shape ``(batch, representation_dim, reduced_len)``,
            where ``reduced_len`` follows from ``encoder_stage_strides``.
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self._stem(x)
        return self._stages(x)
