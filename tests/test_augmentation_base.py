"""Tests for augmentation/base.py extracted ABC hierarchy.

Verifies that TrainingViews, AugmentationMethod, AugmentationTrainingStrategy,
and TrainableAugmentation are correctly defined in the base module.
"""

import pytest
import torch
from torch import nn

from tscollection.models.augmentation.base import (
    AugmentationMethod,
    AugmentationTrainingStrategy,
    TrainableAugmentation,
    TrainingViews,
)


# --------------------------------------------------------------------------- #
# TrainingViews
# --------------------------------------------------------------------------- #


class TestTrainingViewsFromBase:
    """TrainingViews dataclass is importable from base and has correct fields."""

    def test_views_field_is_tuple(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        views = TrainingViews(views=(t1, t2), metadata={'k': 'v'})
        assert isinstance(views.views, tuple)
        assert len(views.views) == 2

    def test_metadata_field_is_dict(self) -> None:
        views = TrainingViews(views=(torch.randn(1, 5, 2),), metadata={'crop_length': 10})
        assert isinstance(views.metadata, dict)
        assert views.metadata['crop_length'] == 10


# --------------------------------------------------------------------------- #
# AugmentationMethod ABC
# --------------------------------------------------------------------------- #


class TestAugmentationMethodFromBase:
    """AugmentationMethod is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AugmentationMethod()  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy ABC
# --------------------------------------------------------------------------- #


class TestAugmentationTrainingStrategyFromBase:
    """AugmentationTrainingStrategy is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AugmentationTrainingStrategy()  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# TrainableAugmentation ABC
# --------------------------------------------------------------------------- #


class TestTrainableAugmentationFromBase:
    """TrainableAugmentation is abstract and inherits nn.Module."""

    def test_has_abstract_methods(self) -> None:
        """TrainableAugmentation declares augment() as abstract."""
        assert 'augment' in TrainableAugmentation.__abstractmethods__

    def test_inherits_from_nn_module(self) -> None:
        assert issubclass(TrainableAugmentation, nn.Module)

    def test_inherits_from_augmentation_method(self) -> None:
        assert issubclass(TrainableAugmentation, AugmentationMethod)
