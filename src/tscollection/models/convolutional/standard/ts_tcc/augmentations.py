from __future__ import annotations

__all__ = ['DataTransform', 'TSTCCAugmentationParameters', 'TSTCCWeakStrongAugmentation']

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tscollection.models.augmentation.base import AugmentationMethod, TrainingViews


@dataclass
class TSTCCAugmentationParameters:
    """Parameters for the TS-TCC weak/strong augmentation pair.

    Args:
        jitter_scale_ratio: Std of the Gaussian scaling factor used by
            the weak (scaling) augmentation.
        jitter_ratio: Std of the additive Gaussian noise used by the
            strong (jitter-after-permutation) augmentation.
        max_seg: Upper bound on the number of segments sampled by the
            permutation step of the strong augmentation.
    """

    jitter_scale_ratio: float = 1.1
    jitter_ratio: float = 0.8
    max_seg: int = 5


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


class TSTCCWeakStrongAugmentation(AugmentationMethod):
    """Weak + strong augmentation pair used by TS-TCC.

    Returns two views: a weak (scaling) view and a strong
    (permutation + jitter) view. Operates on tensors of shape
    ``(batch, channels, seq_len)`` to match the TCC encoder.
    """

    def __init__(self, params: TSTCCAugmentationParameters | None = None) -> None:
        """Initialize the weak/strong augmentation pair.

        Args:
            params: Configuration controlling jitter and permutation
                magnitudes. When ``None``, uses the dataclass defaults.
        """
        self._params = params if params is not None else TSTCCAugmentationParameters()

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> TrainingViews:
        """Return weak and strong augmented views of ``data``.

        Args:
            data: Input tensor of shape ``(batch, channels, seq_len)``.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            TrainingViews with two tensors: ``(weak, strong)``.
        """
        device = data.device
        sample = data.detach().cpu().numpy()
        weak, strong = DataTransform(
            sample,
            jitter_scale_ratio=self._params.jitter_scale_ratio,
            jitter_ratio=self._params.jitter_ratio,
            max_seg=self._params.max_seg,
        )
        return TrainingViews(
            views=(weak.to(device).float(), strong.to(device).float()),
            metadata={},
        )
