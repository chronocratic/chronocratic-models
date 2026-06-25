__all__ = [
    "info_nce_loss",
    "l1_out_loss",
    "local_info_nce_loss",
    "maximum_mean_discrepancy_with_gaussian_kernel_loss",
]

import torch
import torch.nn.functional as F  # noqa: N812

_MIN_BATCH_SIZE = 2


def local_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,  # noqa: ARG001
    pooling: str = "max",
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
    start = int(torch.randint(0, sequence_length - crop_length + 1, (1,), device=z1.device).item())
    crop_z1 = z1[:, start : start + crop_length, :].reshape(batch_size, k, crop_size, embedding_dim)

    if pooling == "max":
        crop_z1 = crop_z1.reshape(batch_size * k, crop_size, embedding_dim)
        crop_z1_pooling = F.max_pool1d(
            crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size
        ).transpose(1, 2)
        crop_z1_pooling = crop_z1_pooling.reshape(batch_size, k, embedding_dim)
    elif pooling == "mean":
        crop_z1_pooling = torch.mean(crop_z1, dim=2)
    else:
        msg = f"Invalid pooling method: {pooling}"
        raise ValueError(msg)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1, 2)  # noqa: N806
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k - 1, dtype=torch.float32, device=z1.device)
    labels = torch.cat([labels, torch.zeros(1, k - 1, device=z1.device)], 0)
    labels = torch.cat([torch.zeros(k, 1, device=z1.device), labels], -1)

    pos_labels = labels.clone()
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
    pooling: str = "max",
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
    if pooling == "max":
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == "mean":
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    batch_size = z1.size(0)
    features = torch.cat([z1, z2], dim=0).squeeze(1)
    features = F.normalize(features, dim=1)

    labels = torch.cat([torch.arange(batch_size, device=z1.device) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=z1.device)
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


def info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, pooling: str = "max", temperature: float = 1.0
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
    if pooling == "max":
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == "mean":
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    else:
        msg = f"Invalid pooling method: {pooling}"
        raise ValueError(msg)

    if z1.shape[0] != z2.shape[0]:
        msg = f"Batch size mismatch: z1 has {z1.shape[0]} samples, z2 has {z2.shape[0]}"
        raise ValueError(msg)
    if z1.shape[0] < _MIN_BATCH_SIZE:
        return z1.new_tensor(0.0)

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


def _compute_gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
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
        msg = f"Error in function {__name__}: bandwidth is invalid; got {bandwidth}"
        raise ValueError(msg)

    bandwidth /= kernel_mul ** (kernel_num // 2)

    if bandwidth <= 0:
        msg = (
            f"Error in function {__name__}: bandwidth after adjustment is invalid; got {bandwidth}"
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
    fix_sigma: float | None = None,
    pooling: str = "max",
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
    if pooling == "max":
        source = F.max_pool1d(
            source.transpose(1, 2).contiguous(), kernel_size=source.size(1)
        ).transpose(1, 2)
        target = F.max_pool1d(
            target.transpose(1, 2).contiguous(), kernel_size=target.size(1)
        ).transpose(1, 2)
    elif pooling == "mean":
        source = torch.unsqueeze(torch.mean(source, 1), 1)
        target = torch.unsqueeze(torch.mean(target, 1), 1)
    else:
        msg = f"Invalid pooling method: {pooling}"
        raise ValueError(msg)

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
