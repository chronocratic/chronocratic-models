"""Shared augmentation producer combinators.

Each class wraps one or more :class:`Augmentation` primitives and assembles
a typed :class:`ViewSet` result. These are generic combinators — they import
nothing model-specific.

Exported symbols:
    - ``SingleViewProducer``: wraps one Augmentation, returns SingleView.
    - ``IndependentPair``: applies one Augmentation twice, returns ViewPair.
    - ``RolePair``: applies two Augmentations, returns ViewPair.
    - ``FullOverlapPair``: applies one Augmentation twice, returns AlignedPair
      with overlap_length == input time dimension.
"""

from __future__ import annotations

__all__ = [
    'FullOverlapPair',
    'IndependentPair',
    'RolePair',
    'SingleViewProducer',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tscollection.models.augmentation.base import (
        AlignedPair,
        Augmentation,
        SingleView,
        ViewPair,
    )

from tscollection.models.augmentation.base import (
    AlignedPair,
    SingleView,
    ViewPair,
)

# --------------------------------------------------------------------------- #
# SingleViewProducer
# --------------------------------------------------------------------------- #


class SingleViewProducer:
    """Wrap one :class:`Augmentation` and return a :class:`SingleView`.

    Satisfies ``AugmentationProducer[SingleView]`` structurally.

    Args:
        aug: The augmentation primitive to apply.
    """

    def __init__(self, *, aug: Augmentation) -> None:
        self._aug = aug

    def produce(self, x: torch.Tensor) -> SingleView:
        """Produce a single augmented view from ``x``.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            SingleView containing the augmented tensor.
        """
        return SingleView(view=self._aug(x))


# --------------------------------------------------------------------------- #
# IndependentPair
# --------------------------------------------------------------------------- #


class IndependentPair:
    """Apply one :class:`Augmentation` twice and return a :class:`ViewPair`.

    Each call to ``aug`` produces an independent (stochastic) draw, so the
    two views differ even though they use the same primitive.

    Satisfies ``AugmentationProducer[ViewPair]`` structurally.

    Args:
        aug: The augmentation primitive to apply (called twice independently).
    """

    def __init__(self, *, aug: Augmentation) -> None:
        self._aug = aug

    def produce(self, x: torch.Tensor) -> ViewPair:
        """Produce two independent augmented views from ``x``.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            ViewPair with two independently transformed views.
        """
        return ViewPair(first=self._aug(x), second=self._aug(x))


# --------------------------------------------------------------------------- #
# RolePair
# --------------------------------------------------------------------------- #


class RolePair:
    """Apply two different :class:`Augmentation`s and return a :class:`ViewPair`.

    Useful when each view has a distinct role (e.g. weak/strong in TS-TCC).

    Satisfies ``AugmentationProducer[ViewPair]`` structurally.

    Args:
        first: Augmentation for the first view.
        second: Augmentation for the second view.
    """

    def __init__(self, *, first: Augmentation, second: Augmentation) -> None:
        self._first = first
        self._second = second

    def produce(self, x: torch.Tensor) -> ViewPair:
        """Produce two augmented views using different primitives.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            ViewPair with first and second views transformed by their
            respective primitives.
        """
        return ViewPair(first=self._first(x), second=self._second(x))


# --------------------------------------------------------------------------- #
# FullOverlapPair
# --------------------------------------------------------------------------- #


class FullOverlapPair:
    """Apply one :class:`Augmentation` twice and return an :class:`AlignedPair`.

    Sets ``overlap_length`` to the full time dimension of the input, indicating
    that the two views are completely aligned (no cropping offset).

    Satisfies ``AugmentationProducer[AlignedPair]`` structurally. Because
    ``AlignedPair`` is-a ``ViewPair``, this also satisfies
    ``AugmentationProducer[ViewPair]`` via covariance.

    Args:
        aug: The augmentation primitive to apply (called twice independently).
        time_dim: The time dimension index in the input tensor.
            Defaults to 1 for (batch, time, channels). Use -1 for (batch, channels, time).
    """

    def __init__(self, *, aug: Augmentation, time_dim: int = 1) -> None:
        self._aug = aug
        self._time_dim = time_dim

    def produce(self, x: torch.Tensor) -> AlignedPair:
        """Produce two aligned augmented views with full overlap.

        Args:
            x: Input tensor.

        Returns:
            AlignedPair with overlap_length equal to the time dimension of x.
        """
        return AlignedPair(
            first=self._aug(x),
            second=self._aug(x),
            overlap_length=x.size(self._time_dim),
        )
