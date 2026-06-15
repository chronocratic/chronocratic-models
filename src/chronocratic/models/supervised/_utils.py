"""Loss helpers for downstream fine-tuning.

Uses ``nn.functional`` exclusively (no module-class instantiants) to avoid
creating temporary ``nn.Module`` objects at call time.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["classification_loss", "regression_loss"]


def classification_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with safe target flattening.

    ``cross_entropy`` expects 1-D integer targets of shape ``(B,)``.
    DataLoaders sometimes return ``(B, 1)`` float tensors, so we flatten
    with ``view(-1)`` instead of ``squeeze()`` to avoid the 0-D scalar
    edge case at batch_size=1.

    Args:
        predictions: Logits of shape ``(B, num_classes)``.
        targets: Class indices (may be float from DataLoader).

    Returns:
        Scalar cross-entropy loss.
    """
    return nn.functional.cross_entropy(predictions, targets.long().view(-1))


def regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss for regression tasks.

    Args:
        predictions: Model outputs of shape ``(B, num_outputs)``.
        targets: Ground truth values of shape ``(B, num_outputs)``.

    Returns:
        Scalar MSE loss.
    """
    return nn.functional.mse_loss(predictions, targets)
