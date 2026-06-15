__all__ = ['MaskMode', 'generate_mask', 'generate_not_nan_mask']

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch

_rng = np.random.default_rng()


class MaskMode(Enum):
    """Masking strategies applied to time-series encoder inputs.

    Attributes:
        BINOMIAL: Each time step is independently masked with fixed probability.
        CONTINUOUS: Contiguous segments of random length are masked.
        ALL_TRUE: No masking; all time steps are kept.
        ALL_FALSE: All time steps are masked.
        MASK_LAST: All time steps are kept except the final one.
    """

    BINOMIAL = 'binomial'
    CONTINUOUS = 'continuous'
    ALL_TRUE = 'all_true'
    ALL_FALSE = 'all_false'
    MASK_LAST = 'mask_last'


def generate_continuous_mask(
    batch_size: int, seq_length: int, n_segments: int = 5, segment_length: float = 0.1
) -> torch.Tensor:
    """
    Generate a continuous mask.

    Args:
        batch_size (int): The batch size.
        seq_length (int): The sequence length.
        n_segments (int or float): Number of segments to mask.
        If float, interpreted as a fraction of seq_length.
        segment_length (int or float): Length of each segment to mask.
        If float, interpreted as a fraction of seq_length.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, seq_length).
    """
    mask = torch.full((batch_size, seq_length), True, dtype=torch.bool)  ## noqa: FBT003

    if isinstance(n_segments, float):
        n_segments = int(n_segments * seq_length)
    n_segments = max(min(n_segments, seq_length // 2), 1)

    if isinstance(segment_length, float):
        segment_length = int(segment_length * seq_length)
    segment_length = max(segment_length, 1)

    for i in range(batch_size):
        for _ in range(n_segments):
            start = _rng.integers(seq_length - segment_length + 1)
            mask[i, start : start + segment_length] = False

    return mask


def generate_binomial_mask(
    batch_size: int, seq_length: int, probability: float = 0.5
) -> torch.Tensor:
    """
    Generate a binomial mask.

    Args:
        batch_size (int): The batch size.
        seq_length (int): The sequence length.
        probability (float): Probability of masking each element.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, seq_length).
    """
    samples = _rng.binomial(1, probability, size=(batch_size, seq_length))
    return torch.from_numpy(samples).to(torch.bool)


def generate_all_true_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """
    Generate a mask where all elements are True.

    Args:
        batch_size (int): The batch size.
        seq_length (int): The sequence length.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, seq_length).
    """
    return torch.full((batch_size, seq_length), True, dtype=torch.bool)  ## noqa: FBT003


def generate_all_false_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """
    Generate a mask where all elements are False.

    Args:
        batch_size (int): The batch size.
        seq_length (int): The sequence length.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, seq_length).
    """
    return torch.full((batch_size, seq_length), False, dtype=torch.bool)  # noqa: FBT003


def generate_mask_last_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """
    Generate a mask where all elements are True except for the last element.

    Args:
        batch_size (int): The batch size.
        seq_length (int): The sequence length.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, seq_length).
    """
    mask = torch.full((batch_size, seq_length), True, dtype=torch.bool)  # noqa: FBT003
    mask[:, -1] = False
    return mask


def get_mask_function(mask_mode: MaskMode | str) -> Callable[..., torch.Tensor]:
    """Return the mask-generation callable for a given mask mode.

    Accepts either a ``MaskMode`` enum value or its string equivalent
    (e.g. ``'binomial'``). Raises ``ValueError`` for unrecognised strings
    or enum values that have no registered function.

    Args:
        mask_mode: The masking strategy to look up.

    Returns:
        A callable with signature ``(batch_size, seq_length, **kwargs) -> torch.Tensor``
        that produces a boolean mask of shape ``(batch_size, seq_length)``.

    Raises:
        ValueError: If ``mask_mode`` is an unrecognised string or has no
            registered implementation.
    """
    if isinstance(mask_mode, str):
        try:
            mask_mode = MaskMode(mask_mode)
        except ValueError:
            msg = f'Unknown mask type string: {mask_mode}'
            raise ValueError(msg)  # noqa: B904

    mask_functions = {
        MaskMode.BINOMIAL: generate_binomial_mask,
        MaskMode.CONTINUOUS: generate_continuous_mask,
        MaskMode.ALL_TRUE: generate_all_true_mask,
        MaskMode.ALL_FALSE: generate_all_false_mask,
        MaskMode.MASK_LAST: generate_mask_last_mask,
    }

    if mask_mode not in mask_functions:
        msg = f'Mask mode functionality not defined: {mask_mode}'
        raise ValueError(msg)

    return mask_functions[mask_mode]


def generate_mask(x: torch.Tensor, mask_mode: MaskMode) -> torch.Tensor:
    """Generate a boolean mask matching the batch and time dimensions of ``x``.

    Args:
        x: Input tensor of shape ``(batch, time, ...)``. Only the first two
            dimensions are used to determine mask shape.
        mask_mode: Masking strategy to apply.

    Returns:
        Boolean mask tensor of shape ``(batch, time)`` on the same device as ``x``.
    """
    mask_function = get_mask_function(mask_mode)
    return mask_function(x.size(0), x.size(1)).to(x.device)


def generate_not_nan_mask(x: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask marking time steps that contain no NaN values.

    Args:
        x: Input tensor of shape ``(batch, time, channels)``.

    Returns:
        Boolean tensor of shape ``(batch, time)`` where ``True`` indicates
        that all channels at that time step are finite.
    """
    return ~x.isnan().any(dim=-1)
