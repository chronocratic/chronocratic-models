__all__ = [
    'global_info_nce_loss',
    'hierarchical_contrastive_loss',
    'info_nce_loss',
    'instance_contrastive_loss',
    'l1_out_loss',
    'local_info_nce_loss',
    'maximum_mean_discrepancy_loss',
    'maximum_mean_discrepancy_with_gaussian_kernel_loss',
    'pairwise_distance_triplet_contrastive_loss',
    'similarity_loss',
    'sliding_local_info_nce_loss',
    'subsequence_info_nce_loss',
    'temporal_contrastive_loss',
    'triplet_contrastive_loss',
]

import random
from typing import cast

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
        A Tensor representing the concatenated set of instance embeddings
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


def triplet_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    negative_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Calculate the triplet contrastive loss given anchor, positive, and negative representations.

    Args:
        anchor (torch.Tensor): Anchor representations of shape (batch_size, embedding_dim).
        positive (torch.Tensor): Positive representations of shape (batch_size, embedding_dim).
        negative (torch.Tensor): Negative representations of shape
            (batch_size, num_negatives, embedding_dim).
        negative_penalty (float): Penalty coefficient for the negative samples (default: 1.0).

    Returns:
        torch.Tensor: Triplet contrastive loss.
    """
    # Positive loss: -logsigmoid of dot product between anchor and positive representations
    pos_dot_product = torch.sum(anchor * positive, dim=1)
    pos_loss = -torch.mean(F.logsigmoid(pos_dot_product))

    # Negative loss: -logsigmoid of negative dot product between anchor and negative representations
    neg_dot_product = torch.sum(anchor.unsqueeze(1) * negative, dim=2)
    neg_loss = -torch.mean(F.logsigmoid(-neg_dot_product))

    # Total loss
    total_loss = pos_loss + negative_penalty * neg_loss

    return total_loss


def pairwise_distance_triplet_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
    p: float = 2.0,
) -> torch.Tensor:
    """Compute triplet loss using pairwise L_p distance with a margin."""
    positive_distance = F.pairwise_distance(anchor, positive, p=p)
    negative_distance = F.pairwise_distance(anchor, negative, p=p)

    loss = F.relu(positive_distance - negative_distance + margin)
    return loss.mean()


def info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = 'max', temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute InfoNCE loss for contrastive learning.

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
        The computed InfoNCE loss.
    """
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    else:
        raise ValueError(f'Invalid pooling method: {pooling}')  # noqa: TRY003, EM102

    z1t = torch.nn.functional.normalize(z1, dim=2)
    z2t = torch.nn.functional.normalize(z2, dim=2)

    similarity_matrix = torch.matmul(z1t.squeeze(1), z2t.squeeze(1).T)

    mask = torch.eye(z1.shape[0], dtype=torch.bool).to(z1.device)

    positives = similarity_matrix[mask].view(mask.shape[0], -1)
    negatives = similarity_matrix[~mask].view(mask.shape[0], mask.shape[1] - 1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()

    return loss


def local_info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = 'max', temperature: float = 1.0, k: int = 16  # noqa: ARG001
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


def subsequence_info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = 'max', temperature: float = 1.0, k: int = 16
) -> torch.Tensor:
    """
    Compute subsequence InfoNCE loss for contrastive learning.

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
        The computed subsequence InfoNCE loss.
    """
    batch_size = z1.size(0)
    sequence_length = z1.size(1)
    embedding_dim = z1.size(2)
    crop_size = int(sequence_length / k)
    crop_length = crop_size * k

    start = random.randint(0, sequence_length - crop_length)  # noqa: S311
    crop_z1 = z1[:, start : start + crop_length, :].view(batch_size, k, crop_size, embedding_dim)
    crop_z2 = z2[:, start : start + crop_length, :].view(batch_size, k, crop_size, embedding_dim)

    crop_z1 = crop_z1.view(batch_size * k, crop_size, embedding_dim)
    crop_z2 = crop_z2.view(batch_size * k, crop_size, embedding_dim)

    if pooling == 'max':
        crop_z1_pooling = F.max_pool1d(
            crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size
        ).transpose(1, 2)
        crop_z2_pooling = F.max_pool1d(
            crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size
        ).transpose(1, 2)
    elif pooling == 'mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1, 1), 1)
        crop_z2_pooling = torch.unsqueeze(torch.mean(z2, 1), 1)
    else:
        raise ValueError(f'Invalid pooling method: {pooling}')  # noqa: TRY003, EM102

    return info_nce_loss(crop_z1_pooling, crop_z2_pooling, cast('str', temperature))


def sliding_local_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,  # noqa: ARG001
    pooling: str = 'max',
    temperature: float = 1.0,
    k: int = 16,
    sliding: int = 16,
    negative_number: int = 16,
) -> torch.Tensor:
    """
    Compute sliding local InfoNCE loss for contrastive learning.

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
    sliding : int, optional
        The sliding window size (default is 16).
    negative_number : int, optional
        The number of negative samples (default is 16).

    Returns:
    -------
    torch.Tensor
        The computed sliding local InfoNCE loss.
    """
    batch_size = z1.size(0)  # noqa: F841
    sequence_length = z1.size(1)
    embedding_dim = z1.size(2)  # noqa: F841
    crop_length = int(sequence_length / k)

    anchors = []
    positives = []
    negatives = []

    start = random.randint(0, crop_length)  # noqa: S311
    while start < sequence_length - crop_length:
        anchors.append(z1[:, start : start + crop_length, :])
        pos_start = random.randint(  # noqa: S311
            max(start - crop_length, 0), min(start + crop_length, sequence_length - crop_length)
        )
        positives.append(z1[:, pos_start : pos_start + crop_length, :])

        neg = []
        while len(neg) < negative_number:
            neg_start = random.randint(0, sequence_length - crop_length)  # noqa: S311
            if (
                max(pos_start - crop_length, 0)
                <= start
                <= min(pos_start + crop_length, sequence_length - crop_length)
            ):
                continue
            neg.append(z1[:, neg_start : neg_start + crop_length, :])
        negatives.append(torch.stack(neg, 1))

        start += sliding

    anchors_array = torch.stack(anchors, 0)
    positives_array = torch.stack(positives, 0)
    negatives_array = torch.stack(negatives, 0)

    if pooling == 'max':
        anchors_array_pooling = torch.max(anchors_array, 2)[0]
        positives_array_pooling = torch.max(positives_array, 2)[0]
        negatives_array_pooling = torch.max(negatives_array, 3)[0]
    elif pooling == 'mean':
        anchors_array_pooling = torch.mean(anchors_array, 2)
        positives_array_pooling = torch.mean(positives_array, 2)
        negatives_array_pooling = torch.mean(negatives_array, 3)
    else:
        raise ValueError(f'Invalid pooling method: {pooling}')  # noqa: TRY003, EM102

    anchors_array_pooling = anchors_array_pooling.view(
        anchors_array_pooling.shape[0] * anchors_array_pooling.shape[1],
        1,
        anchors_array_pooling.shape[2],
    )
    positives_array_pooling = positives_array_pooling.view(
        positives_array_pooling.shape[0] * positives_array_pooling.shape[1],
        1,
        positives_array_pooling.shape[2],
    )
    negatives_array_pooling = negatives_array_pooling.view(
        negatives_array_pooling.shape[0] * negatives_array_pooling.shape[1],
        negative_number,
        negatives_array_pooling.shape[-1],
    )

    apn = torch.cat([anchors_array_pooling, positives_array_pooling, negatives_array_pooling], 1)
    apn_T = apn.transpose(1, 2)  # noqa: N806

    similarity_matrices = torch.bmm(apn, apn_T)[:, 1:, :]
    logits = similarity_matrices / temperature
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


def global_info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = 'max', temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute global InfoNCE loss.

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
        The computed global InfoNCE loss.
    """
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    else:
        msg = f'Invalid pooling method: {pooling}'
        raise ValueError(msg)
    return info_nce_loss(z1, z2, cast('str', temperature))


def similarity_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = 'max', temperature: float = 1.0  # noqa: ARG001
) -> torch.Tensor:
    """
    Compute similarity loss.

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
        The computed similarity loss.
    """
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    else:
        raise ValueError(f'Invalid pooling method: {pooling}')  # noqa: TRY003, EM102

    z1_normalized = F.normalize(z1, dim=2)
    z2_normalized = F.normalize(z2, dim=2)

    similarity_matrix = torch.matmul(z1_normalized.squeeze(1), z2_normalized.squeeze(1).T)

    mask = torch.eye(z1.shape[0], dtype=torch.bool)
    positives = similarity_matrix[mask].view(mask.shape[0], -1)
    loss = positives[:, 0].mean()

    return loss


def maximum_mean_discrepancy_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) loss.

    Parameters
    ----------
    source : torch.Tensor
        Source embeddings.
    target : torch.Tensor
        Target embeddings.

    Returns:
    -------
    torch.Tensor
        The computed MMD loss.
    """
    source_mean = torch.mean(source, dim=0)
    target_mean = torch.mean(target, dim=0)

    loss = torch.sum(torch.square(source_mean - target_mean))

    return loss


def _compute_gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the Gaussian kernel between source and target embeddings.

    Parameters
    ----------
    source : torch.Tensor
        Source embeddings.
    target : torch.Tensor
        Target embeddings.
    kernel_mul : float, optional
        Kernel multiplier (default is 2.0).
    kernel_num : int, optional
        Number of kernels (default is 5).
    fix_sigma : float, optional
        Fixed sigma value for the Gaussian kernel (default is None).
    epsilon : float, optional
    Small value to ensure numerical stability (default is 1e-6).

    Returns:
    -------
    torch.Tensor
        The computed Gaussian kernel.
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples**2 - n_samples)
        bandwidth += epsilon

    if bandwidth <= 0:
        msg = f'Error in function {__name__}: bandwidth is invalid; got {bandwidth}'
        raise ValueError(msg)

    bandwidth /= kernel_mul ** (kernel_num // 2)

    if bandwidth <= 0:
        msg = (
            f'Error in function {__name__}: bandwidth after adjustment is invalid; got {bandwidth}'
        )
        raise ValueError(msg)

    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return torch.sum(torch.stack(kernel_val), dim=0)


def maximum_mean_discrepancy_with_gaussian_kernel_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = 1.0,
    pooling: str = 'max',
) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) loss with Gaussian kernel.

    Parameters
    ----------
    source : torch.Tensor
        Source embeddings.
    target : torch.Tensor
        Target embeddings.
    kernel_mul : float, optional
        Kernel multiplier (default is 2.0).
    kernel_num : int, optional
        Number of kernels (default is 5).
    fix_sigma : float, optional
        Fixed sigma value for the Gaussian kernel (default is None).
    pooling : str, optional
        The pooling method to use ('max' or 'mean', default is 'max').

    Returns:
    -------
    torch.Tensor
        The computed MMD loss.
    """
    if pooling == 'max':
        source = F.max_pool1d(
            source.transpose(1, 2).contiguous(), kernel_size=source.size(1)
        ).transpose(1, 2)
        target = F.max_pool1d(
            target.transpose(1, 2).contiguous(), kernel_size=target.size(1)
        ).transpose(1, 2)
    elif pooling == 'mean':
        source = torch.unsqueeze(torch.mean(source, 1), 1)
        target = torch.unsqueeze(torch.mean(target, 1), 1)
    else:
        raise ValueError(f'Invalid pooling method: {pooling}')  # noqa: TRY003, EM102

    batch_size = int(source.size()[0])
    kernels = _compute_gaussian_kernel(
        source.squeeze(1),
        target.squeeze(1),
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]

    loss = torch.mean(xx + yy - xy - yx)
    return loss
