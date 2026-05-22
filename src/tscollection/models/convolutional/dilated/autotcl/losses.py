__all__ = ['l1_out_loss', 'local_info_nce_loss']

import random

import torch
import torch.nn.functional as F  # noqa: N812


def local_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,  # noqa: ARG001
    pooling: str = 'max',
    temperature: float = 1.0,
    k: int = 16,
) -> torch.Tensor:
    """
    Compute local InfoNCE loss for contrastive learning.

    Parameters
    ----------
    z1 : torch.Tensor
        The first set of embeddings.
    z2 : torch.Tensor
        The second set of embeddings.
    pooling : str, optional
        The pooling method to use ('max' or 'mean', default is 'max').
    temperature : float, optional
        The temperature parameter for scaling the logits (default is 1.0).
    k : int, optional
        The number of local crops (default is 16).

    Returns:
    -------
    torch.Tensor
        The computed local InfoNCE loss.
    """
    z1 = F.normalize(z1, dim=2)
    batch_size = z1.size(0)
    sequence_length = z1.size(1)
    embedding_dim = z1.size(2)
    crop_size = int(sequence_length / k)

    if crop_size < 1:
        return torch.tensor(0.0, device=z1.device)

    crop_length = crop_size * k
    start = random.randint(0, sequence_length - crop_length)  # noqa: S311
    crop_z1 = z1[:, start : start + crop_length, :].reshape(batch_size, k, crop_size, embedding_dim)

    if pooling == 'max':
        crop_z1 = crop_z1.reshape(batch_size * k, crop_size, embedding_dim)
        crop_z1_pooling = F.max_pool1d(
            crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size
        ).transpose(1, 2)
        crop_z1_pooling = crop_z1_pooling.reshape(batch_size, k, embedding_dim)
    elif pooling == 'mean':
        crop_z1_pooling = torch.mean(crop_z1, dim=2)
    else:
        msg = f'Invalid pooling method: {pooling}'
        raise ValueError(msg)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1, 2)  # noqa: N806
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k - 1, dtype=torch.float32)
    labels = torch.cat([labels, torch.zeros(1, k - 1)], 0)
    labels = torch.cat([torch.zeros(k, 1), labels], -1)

    pos_labels = labels.clone().to(z1.device)
    pos_labels[k - 1, k - 2] = 1.0

    neg_labels = labels.T + labels + torch.eye(k, device=z1.device)
    neg_labels[0, 2] = 1.0
    neg_labels[-1, -3] = 1.0

    similarity_matrix = similarity_matrices[0]

    positives = similarity_matrix[pos_labels.bool()].reshape(labels.shape[0], -1)
    negatives = similarity_matrix[~neg_labels.bool()].reshape(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()

    return loss


def l1_out_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pooling: str = 'max',
    temperature: float = 1.0,  ## noqa: ARG001
) -> torch.Tensor:
    """
    Compute L1out loss.

    Parameters
    ----------
    z1 : torch.Tensor
        The first set of embeddings.
    z2 : torch.Tensor
        The second set of embeddings.
    pooling : str, optional
        The pooling method to use ('max' or 'mean', default is 'max').
    temperature : float, optional
        The temperature parameter for scaling the logits (default is 1.0).

    Returns:
    -------
    torch.Tensor
        The computed L1out loss.
    """
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    batch_size = z1.size(0)
    features = torch.cat([z1, z2], dim=0).squeeze(1)
    features = F.normalize(features, dim=1)

    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    mmax = torch.unsqueeze(torch.max(logits, -1)[0], -1)
    exp_negatives = torch.exp(negatives - mmax)
    sum_negs = torch.sum(exp_negatives, -1) + 1e-6
    pos_exp = torch.exp(positives - mmax) + 1e-5
    logits = -torch.log(pos_exp / sum_negs)
    loss = logits.mean()

    return loss
