__all__ = [
    'hierarchical_contrastive_loss',
    'instance_contrastive_loss',
    'temporal_contrastive_loss',
]

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def _compute_contrastive_loss_logits(
    combined_instance: torch.Tensor, dimension_size: int, indexing_factor: np.ndarray
) -> torch.Tensor:
    """
    Compute contrastive loss logits between two sets of embeddings.

    Parameters
    ----------
    combined_instance : torch.Tensor
        A Tensor representing the concatenated set of instance embeddings.
    dimension_size : int
        The size of the dimension along which the embeddings are concatenated.
    indexing_factor : np.array
        The index used to compute the loss.

    Returns:
    -------
    torch.Tensor
        The computed contrastive loss.
    """
    similarity_scores = torch.matmul(
        combined_instance, combined_instance.transpose(1, 2)
    )  # T x 2B x 2B
    lower_triangular_logits = torch.tril(similarity_scores, diagonal=-1)[:, :, :-1]
    upper_triangular_logits = torch.triu(similarity_scores, diagonal=1)[:, :, 1:]
    logits = lower_triangular_logits + upper_triangular_logits
    logits = -F.log_softmax(logits, dim=-1)

    loss = (
        logits[:, indexing_factor, dimension_size + indexing_factor - 1].mean()
        + logits[:, dimension_size + indexing_factor, indexing_factor].mean()
    ) / 2
    return loss


def temporal_contrastive_loss(instance_1: torch.Tensor, instance_2: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal contrastive loss.

    Parameters
    ----------
    instance_1 : torch.Tensor
        The first batch of sequences.
    instance_2 : torch.Tensor
        The second batch of sequences.

    Returns:
    -------
    torch.Tensor
        A tensor representing the temporal contrastive loss.
    """
    sequence_length = instance_1.size(1)
    if sequence_length == 1:
        return instance_1.new_tensor(0.0)
    combined_instance = torch.cat([instance_1, instance_2], dim=1)  # B x 2T x C

    indexing_factor = torch.arange(sequence_length, device=instance_1.device).cpu().numpy()

    return _compute_contrastive_loss_logits(
        combined_instance=combined_instance,
        dimension_size=sequence_length,
        indexing_factor=indexing_factor,
    )


def instance_contrastive_loss(instance_1: torch.Tensor, instance_2: torch.Tensor) -> torch.Tensor:
    """
    Compute instance contrastive loss.

    Parameters
    ----------
    instance_1 : torch.Tensor
        The first batch of sequences.
    instance_2 : torch.Tensor
        The second batch of sequences.

    Returns:
    -------
    torch.Tensor
        A tensor representing the instance contrastive loss.
    """
    batch_size = instance_1.size(0)
    if batch_size == 1:
        return instance_1.new_tensor(0.0)
    combined_instance = torch.cat([instance_1, instance_2], dim=0)  # 2B x T x C
    combined_instance = combined_instance.transpose(0, 1)  # T x 2B x C
    indexing_factor = torch.arange(batch_size, device=instance_1.device).cpu().numpy()

    return _compute_contrastive_loss_logits(
        combined_instance=combined_instance,
        dimension_size=batch_size,
        indexing_factor=indexing_factor,
    )


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
