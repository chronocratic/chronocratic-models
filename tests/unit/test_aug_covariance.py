"""Type covariance verification tests.

Verifies that AugmentationProducer[V] covariance allows producers returning
more specific view set types (e.g., AlignedPair) to be used in slots expecting
broader types (e.g., ViewPair), confirming Liskov substitution compliance.
"""

from __future__ import annotations

import torch

from chronocratic.models.augmentation import AlignedPair, AugmentationProducer, SingleView, ViewPair
from chronocratic.models.augmentation.primitives import Jitter, JitterParameters
from chronocratic.models.augmentation.producers import (
    FullOverlapProducer,
    IndependentPairProducer,
    RolePairProducer,
    SingleViewProducer,
)


class TestViewSetHierarchy:
    """Test ViewSet dataclass inheritance relationships."""

    def test_aligned_pair_is_viewpair(self) -> None:
        """AlignedPair is a subclass of ViewPair."""
        assert issubclass(AlignedPair, ViewPair)

    def test_aligned_pair_has_viewpair_fields(self) -> None:
        """AlignedPair instances have first and second fields like ViewPair."""
        pair = AlignedPair(
            first=torch.randn(2, 10, 3), second=torch.randn(2, 10, 3), overlap_length=10
        )
        assert isinstance(pair, ViewPair)
        assert pair.first.shape == (2, 10, 3)
        assert pair.second.shape == (2, 10, 3)

    def test_single_view_is_not_viewpair(self) -> None:
        """SingleView is not a subclass of ViewPair."""
        assert not issubclass(SingleView, ViewPair)


class TestProducerCovariance:
    """Test AugmentationProducer[V] covariance at runtime."""

    def test_full_overlap_pair_fits_viewpair_slot(self) -> None:
        """FullOverlapProducer produces AlignedPair, which fits ViewPair consumer."""

        def consumer(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 50, 3))

        producer = FullOverlapProducer(aug=Jitter(JitterParameters(sigma=0.1)))
        result = consumer(producer)
        assert isinstance(result, ViewPair)

    def test_independent_pair_fits_viewpair_slot(self) -> None:
        """IndependentPairProducer produces ViewPair directly."""

        def consumer(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 50, 3))

        producer = IndependentPairProducer(aug=Jitter(JitterParameters(sigma=0.1)))
        result = consumer(producer)
        assert isinstance(result, ViewPair)

    def test_role_pair_fits_viewpair_slot(self) -> None:
        """RolePairProducer produces ViewPair directly."""

        def consumer(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 50, 3))

        producer = RolePairProducer(
            first=Jitter(JitterParameters(sigma=0.05)), second=Jitter(JitterParameters(sigma=0.1))
        )
        result = consumer(producer)
        assert isinstance(result, ViewPair)

    def test_single_view_producer_fits_single_view_slot(self) -> None:
        """SingleViewProducer produces SingleView."""

        def consumer(p: AugmentationProducer[SingleView]) -> SingleView:
            return p.produce(torch.randn(2, 50, 3))

        producer = SingleViewProducer(aug=Jitter(JitterParameters(sigma=0.1)))
        result = consumer(producer)
        assert isinstance(result, SingleView)
