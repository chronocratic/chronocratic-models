"""Reconstruction loss functions for the recurrent autoencoder.

Provides MSELoss and MAELoss as thin nn.Module wrappers, mirroring the
loss classes from https://github.com/PyLink88/Recurrent-Autoencoder.
"""

from __future__ import annotations

__all__ = [
    'MAELoss',
    'MSELoss',
]

import torch
from torch import nn


class MAELoss(nn.Module):
    """Mean absolute error loss with ``reduction='mean'``."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the mean absolute error between predictions and targets.

        Args:
            y_hat: Predicted values.
            y_true: Target values.

        Returns:
            Scalar MAE loss.
        """
        return self.loss(y_hat, y_true)


class MSELoss(nn.Module):
    """Mean squared error loss with ``reduction='mean'``."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error between predictions and targets.

        Args:
            y_hat: Predicted values.
            y_true: Target values.

        Returns:
            Scalar MSE loss.
        """
        return self.loss(y_hat, y_true)
