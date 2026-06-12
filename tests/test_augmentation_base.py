"""Tests for augmentation/base.py extracted ABC hierarchy.

Verifies that AugmentationTrainingStrategy are correctly defined in the base module.
Also tests new contract types: Augmentation Protocol, AugmentationProducer[ViewSet],
TrainableAugmentationProducer, and ViewSet dataclasses (SingleView, ViewPair, AlignedPair).

Legacy symbols (TrainingViews, AugmentationMethod, TrainableAugmentation) were
removed — their tests have been deleted.
"""

import pytest
import torch
from torch import nn

from tscollection.models.augmentation.base import (
    AlignedPair,
    Augmentation,
    AugmentationProducer,
    AugmentationTrainingStrategy,
    SingleView,
    TrainableAugmentationProducer,
    ViewPair,
)

# TrainingViews, AugmentationMethod, TrainableAugmentation deleted.


# --------------------------------------------------------------------------- #
# AugmentationTrainingStrategy ABC
# --------------------------------------------------------------------------- #


class TestAugmentationTrainingStrategyFromBase:
    """AugmentationTrainingStrategy is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AugmentationTrainingStrategy()  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# New contract — ViewSet dataclasses
# --------------------------------------------------------------------------- #


class TestSingleViewBase:
    """SingleView dataclass is correctly defined in base.py."""

    def test_single_view_fields(self) -> None:
        t = torch.randn(2, 10, 4)
        sv = SingleView(view=t)
        assert sv.view is t

    def test_single_view_is_frozen(self) -> None:
        sv = SingleView(view=torch.randn(2, 10, 4))
        with pytest.raises(Exception):
            sv.view = torch.randn(3, 5, 2)  # type: ignore[attr-defined]


class TestViewPairBase:
    """ViewPair dataclass is correctly defined in base.py."""

    def test_view_pair_fields(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        vp = ViewPair(first=t1, second=t2)
        assert vp.first is t1
        assert vp.second is t2

    def test_view_pair_is_frozen(self) -> None:
        vp = ViewPair(first=torch.randn(2, 10, 4), second=torch.randn(2, 10, 4))
        with pytest.raises(Exception):
            vp.first = torch.randn(3, 5, 2)  # type: ignore[attr-defined]


class TestAlignedPairBase:
    """AlignedPair extends ViewPair in base.py."""

    def test_aligned_pair_fields(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        ap = AlignedPair(first=t1, second=t2, overlap_length=8)
        assert ap.first is t1
        assert ap.second is t2
        assert ap.overlap_length == 8

    def test_aligned_pair_is_view_pair(self) -> None:
        t1 = torch.randn(2, 10, 4)
        t2 = torch.randn(2, 10, 4)
        ap = AlignedPair(first=t1, second=t2, overlap_length=8)
        assert isinstance(ap, ViewPair)

    def test_aligned_pair_is_frozen(self) -> None:
        ap = AlignedPair(
            first=torch.randn(2, 10, 4),
            second=torch.randn(2, 10, 4),
            overlap_length=5,
        )
        with pytest.raises(Exception):
            ap.overlap_length = 10  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# New contract — Augmentation Protocol
# --------------------------------------------------------------------------- #


class TestAugmentationProtocolBase:
    """Augmentation Protocol is correctly defined."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Structural typing: any class with __call__(Tensor) -> Tensor satisfies it."""

        class CallableAug:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x

        assert isinstance(CallableAug(), Augmentation)


# --------------------------------------------------------------------------- #
# New contract — TrainableAugmentationProducer
# --------------------------------------------------------------------------- #


class TestTrainableAugmentationProducerBase:
    """TrainableAugmentationProducer is a nominal ABC + nn.Module."""

    def test_is_abc(self) -> None:
        """TrainableAugmentationProducer cannot be instantiated directly."""
        from tscollection.models.augmentation import RIPTrainingStrategy

        with pytest.raises(TypeError):
            TrainableAugmentationProducer(training_strategy=RIPTrainingStrategy())

    def test_inherits_from_nn_module(self) -> None:
        assert issubclass(TrainableAugmentationProducer, nn.Module)

    def test_has_abstract_produce(self) -> None:
        assert 'produce' in TrainableAugmentationProducer.__abstractmethods__

    def test_has_abstract_train_step(self) -> None:
        assert 'train_step' in TrainableAugmentationProducer.__abstractmethods__
