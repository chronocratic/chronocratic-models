"""Combinators that build augmentations out of other augmentations.

- :class:`ComposeAugmentation` chains augmentations sequentially on a
  single view (analogous to ``torchvision.transforms.Compose``).
- :class:`PairedAugmentation` runs two augmentations in parallel on the
  same input and returns both as a two-view :class:`TrainingViews`,
  matching the contract of contrastive models like TS-TCC.
"""

from __future__ import annotations

__all__ = ['ComposeAugmentation', 'PairedAugmentation']

from typing import Any

import torch

from tscollection.models.augmentation.base import AugmentationMethod, TrainingViews


class ComposeAugmentation(AugmentationMethod):
    """Apply a sequence of augmentations one after another.

    Each augmentation's first view is fed as input to the next. The
    final output is a single-view :class:`TrainingViews`.
    """

    def __init__(self, augmentations: list[AugmentationMethod]) -> None:
        """Initialize the composition.

        Args:
            augmentations: Augmentations to apply in order.
        """
        self._augmentations = augmentations

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401
    ) -> TrainingViews:
        """Apply each augmentation in order and return the final view."""
        current = data
        for augmentation in self._augmentations:
            current = augmentation.augment(current, **kwargs).views[0]
        return TrainingViews(views=(current,), metadata={})


class PairedAugmentation(AugmentationMethod):
    """Run two augmentations on the same input and return both views.

    Used by contrastive setups (e.g. TS-TCC's weak/strong pair) where
    the model needs two differently augmented views of each sample.
    """

    def __init__(self, first: AugmentationMethod, second: AugmentationMethod) -> None:
        """Initialize the paired augmentation.

        Args:
            first: Augmentation producing the first view.
            second: Augmentation producing the second view.
        """
        self._first = first
        self._second = second

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401
    ) -> TrainingViews:
        """Return ``(first.augment(data), second.augment(data))`` as two views."""
        view_a = self._first.augment(data, **kwargs).views[0]
        view_b = self._second.augment(data, **kwargs).views[0]
        return TrainingViews(views=(view_a, view_b), metadata={})
