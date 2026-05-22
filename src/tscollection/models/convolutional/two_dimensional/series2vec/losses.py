import torch
from torch import nn
import torch.nn.functional as F


def _lower_triangular_pair_indices(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.tril_indices(batch_size, batch_size, offset=-1, device=device)


def _distance_normalizer(distance: torch.Tensor) -> torch.Tensor:
    """Normalize distances to ``[0, 1]`` without changing device placement."""
    if distance.numel() <= 1:
        return torch.zeros_like(distance)

    min_val = torch.min(distance)
    max_val = torch.max(distance)
    denominator = max_val - min_val
    if torch.isclose(denominator, torch.zeros_like(denominator)):
        return torch.zeros_like(distance)
    return (distance - min_val) / denominator


def pairwise_soft_dtw_distances(soft_dtw: nn.Module, time_series: torch.Tensor) -> torch.Tensor:
    """Compute lower-triangular pairwise SoftDTW distances on the input device."""
    if time_series.size(0) < 2:
        return time_series.new_empty(0)

    pair_indices = _lower_triangular_pair_indices(
        batch_size=time_series.size(0), device=time_series.device
    )
    first_series = time_series[pair_indices[0]]
    second_series = time_series[pair_indices[1]]
    return soft_dtw(first_series, second_series)


def pairwise_euclidean_distances(time_series: torch.Tensor) -> torch.Tensor:
    """Compute lower-triangular pairwise Euclidean distances on the input device."""
    if time_series.size(0) < 2:
        return time_series.new_empty(0)

    pair_indices = _lower_triangular_pair_indices(
        batch_size=time_series.size(0), device=time_series.device
    )
    first_series = time_series[pair_indices[0]].reshape(pair_indices.size(1), -1)
    second_series = time_series[pair_indices[1]].reshape(pair_indices.size(1), -1)
    return torch.norm(first_series - second_series, dim=1)


def pretraining_loss(
    temporal_distances: torch.Tensor,
    frequency_distances: torch.Tensor,
    target_temporal_distances: torch.Tensor,
    target_frequency_distances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Series2Vec temporal + frequency pretraining loss."""
    lower_triangular_mask = torch.tril(
        torch.ones_like(temporal_distances, dtype=torch.bool), diagonal=-1
    )

    temporal_distances = _distance_normalizer(
        torch.masked_select(temporal_distances, lower_triangular_mask)
    )
    if temporal_distances.numel() == 0:
        zero_loss = temporal_distances.new_tensor(0.0)
        return zero_loss, zero_loss, zero_loss

    frequency_distances = _distance_normalizer(
        torch.masked_select(frequency_distances, lower_triangular_mask)
    )
    target_temporal_distances = _distance_normalizer(target_temporal_distances)
    target_frequency_distances = _distance_normalizer(target_frequency_distances)

    temporal_loss = F.smooth_l1_loss(temporal_distances, target_temporal_distances)
    frequency_loss = F.smooth_l1_loss(frequency_distances, target_frequency_distances)
    return temporal_loss + frequency_loss, temporal_loss, frequency_loss
