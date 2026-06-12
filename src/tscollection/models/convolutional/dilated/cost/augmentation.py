"""CoST augmentation: random jitter/scale/shift.

Contains the ``CosTRandomFunctionAugmentation`` class and its
``CosTRandomFunctionAugmentationParameters`` dataclass, moved from the shared
``augmentation/strategies.py`` and ``augmentation/config.py`` for per-model
self-containment.

Implements the :class:`~tscollection.models.augmentation.base.Augmentation`
Protocol (``__call__: Tensor -> Tensor``) for use with producer combinators.
"""

__all__ = [
    'CosTRandomFunctionAugmentation',
    'CosTRandomFunctionAugmentationParameters',
]

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tscollection.models.augmentation.base import (
    Augmentation,
)


@dataclass
class CosTRandomFunctionAugmentationParameters:
    """Parameters for :class:`CosTRandomFunctionAugmentation`.

    Controls the stochastic jitter/scale/shift transforms used by CoST.

    Args:
        sigma: Noise scale for jitter, magnitude for scale, offset
            for shift.
        p: Probability of applying each individual transform
            (default ``0.5``).
    """

    sigma: float = 0.1
    p: float = 0.5


class CosTRandomFunctionAugmentation:
    """Stochastic jitter/scale/shift augmentation used by CoST.

    Implements the :class:`~tscollection.models.augmentation.base.Augmentation`
    Protocol (``__call__: Tensor -> Tensor``) for use with producer combinators
    like :class:`~tscollection.models.augmentation.producers.IndependentPair`.
    """

    def __init__(
        self,
        params: CosTRandomFunctionAugmentationParameters | dict[str, Any] | None = None,
        *,
        sigma: float | None = None,
    ) -> None:
        """Initialize the random-function augmentation.

        Args:
            params: Configuration controlling noise scale and apply
                probability. Accepts either a
                ``CosTRandomFunctionAugmentationParameters`` dataclass or a
                dict with ``sigma`` (required) and ``p`` (optional, default
                ``0.5``) keys. When ``None``, uses dataclass defaults
                (sigma=0.1, p=0.5).
            sigma: Convenience keyword argument to set sigma directly
                when not using the ``params`` argument.

        Raises:
            ValueError: If both ``params`` and ``sigma`` are provided.
        """
        if params is not None and sigma is not None:
            msg = "Cannot specify both 'params' and 'sigma'. Use one or the other."
            raise ValueError(msg)
        if params is None and sigma is not None:
            params = {'sigma': sigma}
        if params is None:
            self._params = CosTRandomFunctionAugmentationParameters()
        elif isinstance(params, CosTRandomFunctionAugmentationParameters):
            self._params = params
        else:
            # Backward-compat shim for dict-based params (factories)
            if 'sigma' not in params:
                msg = (
                    "CosTRandomFunctionAugmentation requires 'sigma' in params. "
                    'Pass CosTRandomFunctionAugmentationParameters or a dict '
                    "with 'sigma' (required) and 'p' (optional)."
                )
                raise ValueError(msg)
            self._params = CosTRandomFunctionAugmentationParameters(
                sigma=params['sigma'], p=params.get('p', 0.5)
            )
        self._sigma = self._params.sigma
        self._p = self._params.p

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with std ``sigma`` with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x + (torch.randn(x.shape, device=x.device) * self._sigma)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by a Gaussian factor around 1 with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x * (torch.randn(x.size(-1), device=x.device) * self._sigma + 1)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Add a per-channel Gaussian offset with probability ``p``."""
        if np.random.random() > self._p:  # noqa: NPY002
            return x
        return x + (torch.randn(x.size(-1), device=x.device) * self._sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply jitter/scale/shift and return the augmented tensor.

        Implements the :class:`~tscollection.models.augmentation.base.Augmentation`
        Protocol: ``__call__(Tensor) -> Tensor``.

        Args:
            x: Input time-series tensor of shape ``(batch, time, channels)``.

        Returns:
            Augmented tensor with the same shape as ``x``.
        """
        return self._jitter(self._shift(self._scale(x)))

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> torch.Tensor:
        """Return ``data`` after stochastically applying scale, shift, and jitter.

        Backward-compatible interface. Returns the augmented tensor directly.

        Args:
            data: Input time-series tensor.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            Augmented tensor with the same shape as ``data``.
        """
        return self(data)
