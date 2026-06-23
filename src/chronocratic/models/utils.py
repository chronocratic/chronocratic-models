__all__ = [
    "apply_slicing",
    "concat_last_step_features",
    "extract_features_from_batch",
    "full_series_pooling",
    "integer_pooling",
    "multiscale_pooling",
    "pool_feature_map",
    "process_sample_length",
    "process_sliding_window",
]


from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812


def extract_features_from_batch(batch: torch.Tensor | tuple | list) -> torch.Tensor:
    """
    Extracts the features (inputs) from a batch.

    Parameters
    ----------
    `batch` : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        The input batch which may contain only features
        or a tuple with features and other elements (e.g., labels).

    Returns:
    -------
    torch.Tensor
        The extracted features.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, tuple | list):
        return batch[0]
    msg = f"Unsupported batch format; {type(batch)}"
    raise ValueError(msg)


def process_sample_length(
    sample: torch.Tensor, max_sample_length: int | None = None
) -> torch.Tensor:
    """Randomly crop the sample to `max_sample_length` along the time axis; no-op if unset."""
    if max_sample_length is not None and sample.size(1) > max_sample_length:
        device = sample.device
        rng = np.random.default_rng()
        window_offset = rng.integers(sample.size(1) - max_sample_length + 1)
        sample = sample[:, window_offset : window_offset + max_sample_length]

        sample = sample.to(device)

    return sample


def apply_slicing(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """
    Apply slicing to the tensor if slicing is provided.

    Args:
        tensor (torch.Tensor): The input tensor.
        slicing (slice | None): The slicing to apply.

    Returns:
        torch.Tensor: The sliced tensor.
    """
    if slicing is not None:
        tensor = tensor[:, slicing]
    return tensor


def full_series_pooling(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """
    Apply full series pooling to the tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        slicing (slice | None): The slicing to apply.

    Returns:
        torch.Tensor: The pooled tensor.
    """
    tensor = apply_slicing(tensor=tensor, slicing=slicing)
    pooled_tensor = F.max_pool1d(tensor.transpose(1, 2), kernel_size=tensor.size(1)).transpose(1, 2)
    return pooled_tensor


def multiscale_pooling(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """
    Apply multiscale pooling to the tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        slicing (slice | None): The slicing to apply.

    Returns:
        torch.Tensor: The pooled tensor with multiscale pooling applied.
    """
    all_representations = []
    scale_factor = 0
    while (1 << scale_factor) + 1 < tensor.size(1):
        pooled_output = F.max_pool1d(
            tensor.transpose(1, 2),
            kernel_size=(1 << (scale_factor + 1)) + 1,
            stride=1,
            padding=1 << scale_factor,
        ).transpose(1, 2)
        pooled_output = apply_slicing(tensor=pooled_output, slicing=slicing)
        all_representations.append(pooled_output)
        scale_factor += 1
    multiscale_pooled_tensor = torch.cat(all_representations, dim=-1)
    return multiscale_pooled_tensor


def integer_pooling(
    tensor: torch.Tensor, encoding_window: int, slicing: slice | None = None
) -> torch.Tensor:
    """
    Apply integer-based pooling to the tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        encoding_window (int): The kernel size for max pooling.
        slicing (slice | None): The slicing to apply.

    Returns:
        torch.Tensor: The pooled tensor with integer-based pooling applied.
    """
    pooled_tensor = F.max_pool1d(
        tensor.transpose(1, 2), kernel_size=encoding_window, stride=1, padding=encoding_window // 2
    ).transpose(1, 2)
    if encoding_window % 2 == 0:
        pooled_tensor = pooled_tensor[:, :-1]
    pooled_tensor = apply_slicing(tensor=pooled_tensor, slicing=slicing)
    return pooled_tensor


def process_sliding_window(
    input_tensor: torch.Tensor, left_index: int, right_index: int, time_series_length: int
) -> torch.Tensor:
    """
    Process a sliding window of the input tensor with padding.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        left_index (int): The left index for slicing.
        right_index (int): The right index for slicing.
        time_series_length (int): The total length of the time series.

    Returns:
        torch.Tensor: The padded sliding window tensor.
    """
    return pad_tensor_with_nan(
        tensor=input_tensor[:, max(left_index, 0) : min(right_index, time_series_length)],
        left_pad=-left_index if left_index < 0 else 0,
        right_pad=right_index - time_series_length if right_index > time_series_length else 0,
        axis=1,
    )


def concat_last_step_features(
    trend_embeddings: torch.Tensor, seasonality_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Extract last time-step features from out_t and out_s, concatenate them, and return the result.

    Args:
        trend_embeddings (torch.Tensor):
            First output tensor from the network, shape (batch_size, sequence_length, feature_dim).
        seasonality_embeddings (torch.Tensor):
            Second output tensor from the network, shape (batch_size, sequence_length, feature_dim).

    Returns:
        torch.Tensor: The concatenated tensor of shape (batch_size, 1, concatenated_feature_dim).
    """
    # Extract features from the last time step
    last_step_out_trend = trend_embeddings[:, -1, :]  # Shape: (batch_size, feature_dim)
    last_step_out_seasonality = seasonality_embeddings[:, -1, :]  # Shape: (batch_size, feature_dim)

    # Concatenate along the feature dimension
    concatenated_features = torch.cat(
        [last_step_out_trend, last_step_out_seasonality], dim=-1
    )  # Shape: (batch_size, total_feature_dim)

    # Rearrange to add an extra dimension
    concatenated_features = rearrange(concatenated_features, "b d -> b () d")

    return concatenated_features


def pool_feature_map(features: torch.Tensor) -> torch.Tensor:
    """Global-average-pool an encoder feature map over the time dimension.

    Shared by TSTCC (via ``_postprocess``) and the supervised adapter
    (``tstcc_representations``) to avoid duplicating the pooling logic.

    Args:
        features: Feature map of shape ``(B, C, L)``.

    Returns:
        Pooled tensor of shape ``(B, C)``.

    Raises:
        ValueError: If features is not 3-dimensional.
    """
    if features.ndim != 3:
        msg = f"pool_feature_map expects (B, C, L), got {features.ndim}D tensor"
        raise ValueError(msg)
    return features.mean(dim=-1)


def pad_tensor_with_nan(
    tensor: torch.Tensor, left_pad: int = 0, right_pad: int = 0, axis: int = 0
) -> torch.Tensor:
    """
    Pad a tensor with NaN values along a specified axis.

    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        left_pad (int): The number of NaN values to add to the left side of the specified axis.
        right_pad (int): The number of NaN values to add to the right side of the specified axis.
        axis (int): The axis along which to pad the tensor.

    Returns:
        torch.Tensor: The padded tensor.
    """
    if left_pad > 0:
        left_padding_shape = list(tensor.shape)
        left_padding_shape[axis] = left_pad
        left_padding = torch.full(left_padding_shape, np.nan)
        tensor = torch.cat((left_padding, tensor), dim=axis)

    if right_pad > 0:
        right_padding_shape = list(tensor.shape)
        right_padding_shape[axis] = right_pad
        right_padding = torch.full(right_padding_shape, np.nan)
        tensor = torch.cat((tensor, right_padding), dim=axis)

    return tensor
