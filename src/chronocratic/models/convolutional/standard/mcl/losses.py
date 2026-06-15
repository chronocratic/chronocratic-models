__all__ = ["MixUpLoss"]

import torch
from torch import nn


class MixUpLoss(torch.nn.Module):
    """MixUp contrastive loss used by MCL."""

    def __init__(self) -> None:
        super().__init__()

        self.tau = 0.5
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(
        self, z_aug: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size = z_aug.shape[0]
        device = z_aug.device

        z_1 = nn.functional.normalize(z_1)
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam * torch.eye(batch_size, device=device)
        labels_lam_1 = (1 - lam) * torch.eye(batch_size, device=device)

        labels = torch.cat((labels_lam_0, labels_lam_1), 1)

        logits = torch.cat((torch.mm(z_aug, z_1.T), torch.mm(z_aug, z_2.T)), 1)

        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """Compute soft-label cross-entropy as the mean of per-sample log-softmax dot products."""
        return torch.mean(torch.sum(-soft_targets * self.logsoftmax(logits), 1))
