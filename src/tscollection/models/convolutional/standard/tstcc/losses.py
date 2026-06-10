from __future__ import annotations

__all__ = ['NTXentLoss']

import torch
from torch import nn
from torch.nn import functional


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss (SimCLR / NT-Xent).

    Unlike the original TS-TCC implementation, the correlated-sample mask is
    computed lazily inside ``forward`` so variable-sized last batches work.
    """

    def __init__(self, temperature: float = 0.2, *, use_cosine_similarity: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        """Return the NT-Xent loss for paired projection embeddings.

        Args:
            zis: Projection embeddings with shape ``(batch, dim)``.
            zjs: Projection embeddings with shape ``(batch, dim)``.

        Returns:
            Scalar NT-Xent loss.
        """
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)  # (2*batch, dim)

        if self.use_cosine_similarity:
            similarity_matrix = functional.cosine_similarity(
                representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
            )
        else:
            similarity_matrix = torch.mm(representations, representations.T)

        mask = self._correlated_mask(batch_size, zis.device)
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=zis.device)
        return self.criterion(logits, labels) / (2 * batch_size)

    @staticmethod
    def _correlated_mask(batch_size: int, device: torch.device) -> torch.Tensor:
        n = 2 * batch_size
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        idx = torch.arange(batch_size, device=device)
        mask[idx, idx + batch_size] = False
        mask[idx + batch_size, idx] = False
        return mask
