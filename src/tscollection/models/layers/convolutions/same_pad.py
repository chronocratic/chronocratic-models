__all__ = ['Conv1dSamePad', 'Conv1dSamePadMultiBlock']

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePad(nn.Module):
    """
    A 1D convolutional layer that ensures the output has the same length as the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.conv(x)
        if self.remove > 0:
            output = output[:, :, : -self.remove]
        return output


class Conv1dSamePadMultiBlock(nn.Module):
    """
    A block consisting of n 1D convolutional layers with user specified activation functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int = 1,
        num_blocks: int = 2,
        is_final: bool = False,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        super().__init__()

        self.activation_fn = activation_fn

        self.__initiate_blocks(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            num_blocks=num_blocks,
        )

        self.__initiate_projector(
            in_channels=in_channels, out_channels=out_channels, stride=stride, is_final=is_final
        )

    def __initiate_blocks(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int,
        num_blocks: int,
    ) -> None:
        self.blocks = nn.ModuleList()

        self.blocks.append(
            Conv1dSamePad(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
            )
        )

        for i in range(1, num_blocks):
            self.blocks.append(
                Conv1dSamePad(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                )
            )

    def __initiate_projector(
        self, in_channels: int, out_channels: int, stride: int, is_final: bool
    ) -> None:

        if stride == 1:
            if in_channels != out_channels or is_final:
                self.projector = nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1
                )
            else:
                self.projector = None

        else:
            self.projector = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x if self.projector is None else self.projector(x)
        for block in self.blocks:
            x = self.activation_fn(x)
            x = block(x)
        return x + residual
