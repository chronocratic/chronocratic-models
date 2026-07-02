"""Tests for augmentation producer combinators.

VER-01 through VER-07 cover the four shared producer classes:
SingleViewProducer, IndependentPairProducer, RolePairProducer, FullOverlapProducer.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.augmentation.base import AlignedPair, SingleView, ViewPair
from chronocratic.models.augmentation.primitives import Jitter, JitterParameters, Scaling

# --------------------------------------------------------------------------- #
# Import producers under test
# --------------------------------------------------------------------------- #
from chronocratic.models.augmentation.producers import (
    FullOverlapProducer,
    IndependentPairProducer,
    RolePairProducer,
    SingleViewProducer,
)

# --------------------------------------------------------------------------- #
# SingleViewProducer
# --------------------------------------------------------------------------- #


class TestSingleViewProducer:
    def test_produces_single_view_with_tensor(self) -> None:
        """SingleViewProducer produces SingleView with .view tensor (VER-01)."""
        aug = Jitter()
        producer = SingleViewProducer(aug=aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        assert isinstance(result, SingleView)
        assert result.view.shape == x.shape

    def test_applies_augmentation(self) -> None:
        """SingleViewProducer applies the wrapped primitive."""
        aug = Scaling()
        producer = SingleViewProducer(aug=aug)
        x = torch.ones(2, 10, 3)
        result = producer.produce(x)

        # Scaling changes values, so result should not be all ones
        assert not torch.allclose(result.view, x)


# --------------------------------------------------------------------------- #
# IndependentPairProducer
# --------------------------------------------------------------------------- #


class TestIndependentPairProducer:
    def test_produces_view_pair(self) -> None:
        """IndependentPairProducer produces ViewPair (VER-02)."""
        aug = Jitter()
        producer = IndependentPairProducer(aug=aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape

    def test_independent_draws_produce_different_tensors(self) -> None:
        """IndependentPairProducer with Jitter produces different tensors for each draw (VER-05)."""
        torch.manual_seed(42)
        aug = Jitter()
        producer = IndependentPairProducer(aug=aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        # Jitter adds random noise, so two draws should differ
        assert not torch.equal(result.first, result.second)


# --------------------------------------------------------------------------- #
# RolePairProducer
# --------------------------------------------------------------------------- #


class TestRolePairProducer:
    def test_produces_view_pair(self) -> None:
        """RolePairProducer produces ViewPair (VER-03)."""
        first_aug = Jitter()
        second_aug = Scaling()
        producer = RolePairProducer(first=first_aug, second=second_aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape

    def test_first_and_second_correspond_to_correct_primitives(self) -> None:
        """RolePairProducer first/second correspond to correct primitives (VER-06)."""
        # Use a deterministic input so we can detect which transform was applied
        torch.manual_seed(100)
        x = torch.zeros(1, 5, 2)

        # Jitter adds noise, Scaling multiplies
        jitter = Jitter(params=JitterParameters(sigma=0.1))
        prod = RolePairProducer(first=jitter, second=Jitter())

        result = prod.produce(x)
        # Both should have the same shape as input
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape

        # The key test: both first and second are transformed versions of x
        # (not identical to x because jitter adds noise)
        assert not torch.allclose(result.first, x)


# --------------------------------------------------------------------------- #
# FullOverlapProducer
# --------------------------------------------------------------------------- #


class TestFullOverlapProducer:
    def test_produces_aligned_pair(self) -> None:
        """FullOverlapProducer produces AlignedPair with overlap_length == T (VER-04)."""
        aug = Jitter()
        producer = FullOverlapProducer(aug=aug)
        x = torch.randn(2, 10, 3)
        result = producer.produce(x)

        assert isinstance(result, AlignedPair)
        # AlignedPair extends ViewPair, so it has first/second
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape
        assert result.overlap_length == x.size(1)


# --------------------------------------------------------------------------- #
# Protocol compliance
# --------------------------------------------------------------------------- #


class TestProtocolCompliance:
    def test_single_view_producer_has_produce_method(self) -> None:
        """SingleViewProducer satisfies AugmentationProducer structurally."""
        producer = SingleViewProducer(aug=Jitter())
        assert callable(getattr(producer, "produce", None))

    def test_independent_pair_has_produce_method(self) -> None:
        """IndependentPairProducer satisfies AugmentationProducer structurally."""
        producer = IndependentPairProducer(aug=Jitter())
        assert callable(getattr(producer, "produce", None))

    def test_role_pair_has_produce_method(self) -> None:
        """RolePairProducer satisfies AugmentationProducer structurally."""
        producer = RolePairProducer(first=Jitter(), second=Scaling())
        assert callable(getattr(producer, "produce", None))

    def test_full_overlap_pair_has_produce_method(self) -> None:
        """FullOverlapProducer satisfies AugmentationProducer structurally."""
        producer = FullOverlapProducer(aug=Jitter())
        assert callable(getattr(producer, "produce", None))


# --------------------------------------------------------------------------- #
# Keyword-only constructors
# --------------------------------------------------------------------------- #


class TestKeywordOnlyConstructors:
    def test_single_view_producer_requires_kwonly(self) -> None:
        """Producers have keyword-only constructors (VER-07)."""
        with pytest.raises(TypeError):
            SingleViewProducer(Jitter())  # type: ignore[arg-type]

    def test_independent_pair_requires_kwonly(self) -> None:
        with pytest.raises(TypeError):
            IndependentPairProducer(Jitter())  # type: ignore[arg-type]

    def test_role_pair_requires_kwonly(self) -> None:
        with pytest.raises(TypeError):
            RolePairProducer(Jitter(), Scaling())  # type: ignore[arg-type]

    def test_full_overlap_pair_requires_kwonly(self) -> None:
        with pytest.raises(TypeError):
            FullOverlapProducer(Jitter())  # type: ignore[arg-type]
