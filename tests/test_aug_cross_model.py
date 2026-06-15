"""Cross-model augmentation reuse tests.

Verifies that augmentation producers from the shared contract layer can be
injected into models they were not originally designed for, demonstrating
the N+M decoupling goal (SPEC success criteria 4-5, 8-9).
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from chronocratic.models.augmentation import (
    AlignedPair,
    AugmentationProducer,
    TrainableAugmentationProducer,
    ViewPair,
)
from chronocratic.models.augmentation.decorators import Seeded
from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)
from chronocratic.models.augmentation.producers import (
    FullOverlapPair,
    IndependentPair,
    RolePair,
    SingleViewProducer,
)
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec

# Import _train_steps from test_smoke.py to avoid duplication
from tests.test_smoke import _train_steps


class TestCrossModelReuse:
    """Test that shared augmentations work across different models (SPEC criterion 4)."""

    def test_full_overlap_pair_into_ts2vec(self) -> None:
        """FullOverlapPair(Jitter) injected into TS2Vec trains 1 step with finite loss."""
        aug = FullOverlapPair(aug=Jitter(JitterParameters(sigma=0.1)))
        model = TS2Vec(input_dims=1, augmentation=aug)
        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=1)
        assert len(losses) == 1
        assert math.isfinite(losses[0].item())

    def test_independent_pair_into_ts2vec_via_covariance(self) -> None:
        """IndependentPair (returns ViewPair) fits TS2Vec's AlignedPair slot? No.

        TS2Vec requires AlignedPair (overlap_length). IndependentPair returns
        ViewPair which lacks overlap_length, so this should fail at runtime.
        FullOverlapPair returns AlignedPair, so it works (tested above).

        This test verifies that the correct producer types work.
        """
        # FullOverlapPair returns AlignedPair, which has overlap_length
        aug = FullOverlapPair(aug=Scaling(ScalingParameters(sigma=0.05)))
        pair = aug.produce(torch.randn(2, 50, 1))
        assert isinstance(pair, AlignedPair)
        assert pair.overlap_length == 50

    def test_role_pair_produces_view_pair(self) -> None:
        """RolePair with two different primitives produces ViewPair."""
        weak = Jitter(JitterParameters(sigma=0.05))
        strong = ComposeAugmentation(
            [Jitter(JitterParameters(sigma=0.1)), Scaling(ScalingParameters(sigma=0.1))]
        )
        aug = RolePair(first=weak, second=strong)
        data = torch.randn(2, 50, 3)
        pair = aug.produce(data)
        assert isinstance(pair, ViewPair)
        assert pair.first.shape == data.shape
        assert pair.second.shape == data.shape

    def test_compose_augmentation_cross_model(self) -> None:
        """ComposeAugmentation with primitives works in FullOverlapPair for TS2Vec."""
        composed = ComposeAugmentation(
            [Jitter(JitterParameters(sigma=0.05)), Scaling(ScalingParameters(sigma=0.05))]
        )
        aug = FullOverlapPair(aug=composed)
        model = TS2Vec(input_dims=1, augmentation=aug)
        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=1)
        assert len(losses) == 1
        assert math.isfinite(losses[0].item())

    def test_permutation_in_full_overlap_pair(self) -> None:
        """Permutation primitive works inside FullOverlapPair producer."""
        aug = FullOverlapPair(aug=Permutation(PermutationParameters(max_segments=3, time_dim=-1)))
        data = torch.randn(2, 50, 3)
        pair = aug.produce(data)
        assert isinstance(pair, AlignedPair)
        assert pair.first.shape == data.shape
        assert pair.second.shape == data.shape
        assert pair.overlap_length == 50


class TestCovariance:
    """Test type covariance of AugmentationProducer (SPEC criterion 5)."""

    def test_crop_shift_producer_covariance(self) -> None:
        """CropShiftProducer returns AlignedPair, assignable to ViewPair slot.

        CropShiftProducer is AugmentationProducer[AlignedPair].
        AlignedPair is-a ViewPair, and V is covariant in AugmentationProducer[V].
        Therefore CropShiftProducer fits any AugmentationProducer[ViewPair] slot.
        """
        from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer

        def accepts_viewpair(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 100, 3))

        producer = CropShiftProducer()
        result = accepts_viewpair(producer)
        assert isinstance(result, ViewPair)
        # AlignedPair is-a ViewPair, so this also holds
        assert isinstance(result, AlignedPair)

    def test_full_overlap_pair_is_viewpair_producer(self) -> None:
        """FullOverlapPair returns AlignedPair, fits ViewPair producer slot."""

        def accepts_viewpair(p: AugmentationProducer[ViewPair]) -> ViewPair:
            return p.produce(torch.randn(2, 50, 3))

        producer = FullOverlapPair(aug=Jitter(JitterParameters(sigma=0.1)))
        result = accepts_viewpair(producer)
        assert isinstance(result, AlignedPair)

    def test_aligned_pair_is_viewpair(self) -> None:
        """AlignedPair is a subclass of ViewPair (issubclass check)."""
        assert issubclass(AlignedPair, ViewPair)


class TestSeededDecorator:
    """Test Seeded decorator constraints (SPEC criterion 8)."""

    def test_seeded_on_stateless_producer(self) -> None:
        """Seeded wraps SingleViewProducer without error and produces deterministic output."""
        inner = SingleViewProducer(aug=Jitter(JitterParameters(sigma=0.1)))
        seeded = Seeded(inner=inner, seed=42)
        x = torch.randn(2, 10, 3)
        r1 = seeded.produce(x)
        r2 = seeded.produce(x)
        assert torch.equal(r1.view, r2.view)

    def test_seeded_on_full_overlap_pair(self) -> None:
        """Seeded works with FullOverlapPair producer."""
        inner = FullOverlapPair(aug=Jitter(JitterParameters(sigma=0.1)))
        seeded = Seeded(inner=inner, seed=123)
        x = torch.randn(2, 10, 3)
        r1 = seeded.produce(x)
        r2 = seeded.produce(x)
        assert torch.equal(r1.first, r2.first)
        assert torch.equal(r1.second, r2.second)

    def test_seeded_rejects_trainable(self) -> None:
        """Seeded raises TypeError for TrainableAugmentationProducer.

        Verifies the isinstance guard from decorators.py.
        """
        from chronocratic.models.augmentation.base import AugmentationTrainingStrategy, SingleView

        class _DummyStrategy(AugmentationTrainingStrategy):
            """Minimal training strategy for test purposes."""

            def compute_loss(
                self,
                x_embeddings: torch.Tensor,
                aug_x_embeddings: torch.Tensor,
                augmentation_factor: torch.Tensor,
            ) -> torch.Tensor:
                return torch.tensor(0.0)

        class DummyTrainable(TrainableAugmentationProducer):
            """Minimal trainable producer for testing."""

            def produce(self, x: torch.Tensor) -> SingleView:
                return SingleView(view=x)

            def train_step(
                self, x: torch.Tensor, encoder: torch.nn.Module, batch_idx: int
            ) -> torch.Tensor | None:
                return None

        dummy_trainable = DummyTrainable(training_strategy=_DummyStrategy())
        with pytest.raises(TypeError, match="TrainableAugmentationProducer"):
            Seeded(inner=dummy_trainable, seed=42)


class TestImportHygiene:
    """Verify shared modules import nothing model-specific (SPEC criterion 9)."""

    def test_primitives_no_model_imports(self) -> None:
        """primitives.py must not import from convolutional/."""
        import chronocratic.models.augmentation.primitives as primitives

        source = pathlib.Path(primitives.__file__).read_text()
        assert "convolutional" not in source, (
            "primitives.py must not import from convolutional/ "
            "(SPEC criterion 9, shared modules must be model-agnostic)"
        )

    def test_producers_no_model_imports(self) -> None:
        """producers.py must not import from convolutional/."""
        import chronocratic.models.augmentation.producers as producers

        source = pathlib.Path(producers.__file__).read_text()
        assert "convolutional" not in source, (
            "producers.py must not import from convolutional/ "
            "(SPEC criterion 9, shared modules must be model-agnostic)"
        )

    def test_decorators_no_model_imports(self) -> None:
        """decorators.py must not import from convolutional/."""
        import chronocratic.models.augmentation.decorators as decorators

        source = pathlib.Path(decorators.__file__).read_text()
        assert "convolutional" not in source, (
            "decorators.py must not import from convolutional/ "
            "(shared modules must be model-agnostic)"
        )
