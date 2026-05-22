__all__ = ['hierarchical_contrastive_loss']

import torch
import torch.nn.functional as F  # noqa: N812

from tscollection.models.losses import instance_contrastive_loss, temporal_contrastive_loss


def hierarchical_contrastive_loss(
    instance_1: torch.Tensor, instance_2: torch.Tensor, alpha: float = 0.5, temporal_unit: int = 0
) -> torch.Tensor:
    """
    Compute hierarchical contrastive loss combining instance and temporal losses.

    Parameters
    ----------
    instance_1 : torch.Tensor
        The first batch of sequences.
    instance_2 : torch.Tensor
        The second batch of sequences.
    alpha : float, optional
        The weight of the instance contrastive loss in the total loss (default is 0.5).
    temporal_unit : int, optional
        The temporal unit for hierarchical computation (default is 0).

    Returns:
    -------
    torch.Tensor
        A tensor representing the hierarchical contrastive loss.
    """
    loss = torch.tensor(0.0, device=instance_1.device)
    hierarchy_level = 0
    while instance_1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(instance_1, instance_2)
        if hierarchy_level >= temporal_unit:  # noqa: SIM102
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(instance_1, instance_2)
        hierarchy_level += 1
        instance_1 = F.max_pool1d(instance_1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        instance_2 = F.max_pool1d(instance_2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if instance_1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(instance_1, instance_2)
        hierarchy_level += 1
    return loss / hierarchy_level
