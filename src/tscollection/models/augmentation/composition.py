"""Abstract two-view augmentation contract.

Defines :class:`PairedAugmentation` — the abstract base type for any
augmentation that produces two views of the input, used by contrastive
setups like TS-TCC. Concrete pairs live alongside the models that use
them (e.g. ``ts_tcc/augmentations.py``).
"""

from __future__ import annotations

__all__ = ['PairedAugmentation']

from abc import ABC, abstractmethod
from typing import Any

import torch

from tscollection.models.augmentation.base import AugmentationMethod, TrainingViews


class PairedAugmentation(AugmentationMethod, ABC):
    """Abstract augmentation that produces two views from a single input.

    Subclasses implement :meth:`first` and :meth:`second` to return the
    :class:`AugmentationMethod` used for each view. The base :meth:`augment`
    runs both on the same input and bundles their first views into a
    two-view :class:`TrainingViews`.

    The slot names ``first`` / ``second`` are intentionally role-agnostic;
    subclasses may expose additional aliases (``weak`` / ``strong``,
    ``query`` / ``key``, …) that name the roles they assign.
    """

    @property
    @abstractmethod
    def first(self) -> AugmentationMethod:
        """Augmentation producing the first view."""

    @property
    @abstractmethod
    def second(self) -> AugmentationMethod:
        """Augmentation producing the second view."""

    def augment(
        self,
        data: torch.Tensor,
        **kwargs: Any,  # noqa: ANN401
    ) -> TrainingViews:
        """Apply ``first`` and ``second`` to ``data`` and return both views."""
        view_a = self.first.augment(data, **kwargs).views[0]
        view_b = self.second.augment(data, **kwargs).views[0]
        return TrainingViews(views=(view_a, view_b), metadata={})
