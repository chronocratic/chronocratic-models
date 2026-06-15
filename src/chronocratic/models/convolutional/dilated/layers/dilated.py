__all__ = ['Conv1dDilatedEncoder']

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from chronocratic.models.convolutional.dilated.layers.same_pad import Conv1dSamePadMultiBlock


class Conv1dDilatedEncoder(nn.Module):
    """A sequence of Conv1dMultiBlock layers with exponentially increasing dilation.

    in_channels: Number of input channels.
    channels: List of output channels for each layer.
    kernel_size: Size of the kernel for the convolutions.
    activation_fn: Activation function to use in the Conv1dMultiBlock.
    num_blocks: Number of convolutional layers in each block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int,
        stride: int = 1,
        num_blocks: int = 2,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[
                Conv1dSamePadMultiBlock(
                    in_channels=channels[i - 1] if i > 0 else in_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    stride=stride,
                    num_blocks=num_blocks,
                    activation_fn=activation_fn,
                    is_final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)
