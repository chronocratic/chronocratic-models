from __future__ import annotations

__all__ = ['DataTransform']

import numpy as np
import torch


def DataTransform(
    sample: np.ndarray,
    jitter_scale_ratio: float,
    jitter_ratio: float,
    max_seg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply weak and strong augmentations to a batch of time series.

    Args:
        sample: ``(batch, channels, seq_len)`` numpy array
        jitter_scale_ratio: scale std for the weak (scaling) augmentation
        jitter_ratio: jitter std for the strong augmentation
        max_seg: maximum number of segments for the permutation augmentation

    Returns:
        weak_aug, strong_aug: ``(batch, channels, seq_len)`` float tensors
    """
    weak_aug = _scaling(sample, jitter_scale_ratio)
    strong_aug = _jitter(_permutation(sample, max_segments=max_seg), jitter_ratio)
    return weak_aug, strong_aug


def _jitter(x: np.ndarray, sigma: float = 0.8) -> torch.Tensor:
    return torch.from_numpy(x + np.random.normal(loc=0.0, scale=sigma, size=x.shape))


def _scaling(x: np.ndarray, sigma: float = 1.1) -> torch.Tensor:
    factor = np.random.normal(loc=2.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    scaled = np.concatenate(
        [np.multiply(x[:, i, :], factor)[:, np.newaxis, :] for i in range(x.shape[1])],
        axis=1,
    )
    return torch.from_numpy(scaled)


def _permutation(x: np.ndarray, max_segments: int = 5) -> np.ndarray:
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0],))
    result = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            result[i] = pat[:, warp]  # apply permutation across all channels
        else:
            result[i] = pat
    return result
