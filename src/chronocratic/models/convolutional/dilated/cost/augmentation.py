"""CoST augmentation: random jitter/scale/shift.

Contains the ``CosTRandomFunctionAugmentation`` class and its
``CosTRandomFunctionAugmentationParameters`` dataclass, moved from the shared
``augmentation/strategies.py`` and ``augmentation/config.py`` for per-model
self-containment.

Implements the :class:`~chronocratic.models.augmentation.base.Augmentation`
Protocol (``__call__: Tensor -> Tensor``) for use with producer combinators.
"""

__all__ = ["CosTRandomFunctionAugmentation", "CosTRandomFunctionAugmentationParameters"]

from dataclasses import dataclass
from typing import Any

import torch

from chronocratic.models.augmentation.base import Augmentation


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

    sigma: float = 0.5
    p: float = 0.5


class CosTRandomFunctionAugmentation(Augmentation):
    """Stochastic jitter/scale/shift augmentation used by CoST.

    Implements the :class:`~chronocratic.models.augmentation.base.Augmentation`
    Protocol (``__call__: Tensor -> Tensor``) for use with producer combinators
    like :class:`~chronocratic.models.augmentation.producers.IndependentPairProducer`.
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
                (sigma=0.5, p=0.5).
            sigma: Convenience keyword argument to set sigma directly
                when not using the ``params`` argument.

        Raises:
            ValueError: If both ``params`` and ``sigma`` are provided.
        """
        if params is not None and sigma is not None:
            msg = "Cannot specify both 'params' and 'sigma'. Use one or the other."
            raise ValueError(msg)
        if params is None and sigma is not None:
            params = {"sigma": sigma}
        if params is None:
            self._params = CosTRandomFunctionAugmentationParameters()
        elif isinstance(params, CosTRandomFunctionAugmentationParameters):
            self._params = params
        else:
            self._params = CosTRandomFunctionAugmentationParameters(
                sigma=params["sigma"], p=params.get("p", 0.5)
            )
        self._sigma = self._params.sigma
        self._p = self._params.p

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with std ``sigma`` with probability ``p``."""
        if torch.rand(1).item() > self._p:  # device-ok: CPU scalar probability
            return x
        return x + (torch.randn(x.shape, device=x.device) * self._sigma)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply each channel by a Gaussian factor around 1 with probability ``p``.

        Expects input of shape ``(batch, time, channels)``.
        """
        if torch.rand(1).item() > self._p:  # device-ok: CPU scalar probability
            return x
        channels = x.size(-1)
        return x * (torch.randn(channels, device=x.device) * self._sigma + 1)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Add a per-channel Gaussian offset with probability ``p``.

        Expects input of shape ``(batch, time, channels)``.
        """
        if torch.rand(1).item() > self._p:  # device-ok: CPU scalar probability
            return x
        channels = x.size(-1)
        return x + (torch.randn(channels, device=x.device) * self._sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply jitter/scale/shift and return the augmented tensor.

        Implements the :class:`~chronocratic.models.augmentation.base.Augmentation`
        Protocol: ``__call__(Tensor) -> Tensor``.

        Args:
            x: Input time series tensor of shape ``(batch, time, channels)``.

        Returns:
            Augmented tensor with the same shape as ``x``.
        """
        return self._jitter(self._shift(self._scale(x)))
