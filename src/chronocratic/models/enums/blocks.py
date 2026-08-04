"""Residual block type enum.

Selects which residual block a ResNet-style encoder stacks. The two values
correspond to the two blocks defined by the original ResNet paper and carried
into every downstream port.

Mapping to depth conventions
    BASIC:      two 3-tap convolutions, ``expansion = 1`` (ResNet-18/34).
    BOTTLENECK: 1-tap / 3-tap / 1-tap, ``expansion = 4`` (ResNet-50/101/152).

``expansion`` multiplies the stage's channel count to give the block's output
width, so it also determines the encoder's representation width.
"""

from __future__ import annotations

from enum import StrEnum


class ResidualBlockType(StrEnum):
    """Type of residual block to stack in a ResNet-style encoder.

    Attributes:
        BASIC: Two convolutions per block with ``expansion = 1``. The
            ResNet-18/34 block.
        BOTTLENECK: Channel-reducing 1-tap, 3-tap, then channel-expanding
            1-tap convolution, with ``expansion = 4``. The ResNet-50/101/152
            block.
    """

    BASIC = "basic"
    BOTTLENECK = "bottleneck"
