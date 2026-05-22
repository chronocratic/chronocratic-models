__all__ = ['MixUpLoss']

import torch
from torch import nn


class MixUpLoss(torch.nn.Module):
    def __init__(self, device: torch.device, batch_size: int) -> None:
        super().__init__()

        self.tau = 0.5
        self.device = device
        self.batch_size = batch_size
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(
        self, z_aug: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor, lam: int
    ) -> torch.Tensor:
        """Forward pass."""
        z_1 = nn.functional.normalize(z_1)
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam * torch.eye(self.batch_size, device=self.device)
        labels_lam_1 = (1 - lam) * torch.eye(self.batch_size, device=self.device)

        labels = torch.cat((labels_lam_0, labels_lam_1), 1)

        logits = torch.cat((torch.mm(z_aug, z_1.T), torch.mm(z_aug, z_2.T)), 1)

        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits, soft_targets) -> torch.Tensor:
        """Compute soft-label cross-entropy as the mean of per-sample log-softmax dot products."""
        return torch.mean(torch.sum(-soft_targets * self.logsoftmax(logits), 1))
