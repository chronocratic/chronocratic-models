"""TDD tests for TS2Vec producer integration.

Verifies CropShiftProducer returns AlignedPair, TS2Vec accepts
AugmentationProducer[AlignedPair], and training runs with finite loss.
"""

from collections.abc import Callable
from copy import deepcopy

import torch

from chronocratic.models.augmentation.base import AlignedPair
from chronocratic.models.augmentation.decorators import Seeded
from chronocratic.models.augmentation.primitives import Jitter
from chronocratic.models.augmentation.producers import FullOverlapPair
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import (
    CropShiftProducer,
)
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec

# --------------------------------------------------------------------------- #
# CropShiftProducer unit tests
# --------------------------------------------------------------------------- #


class TestCropShiftProducer:
    """Tests for CropShiftProducer produce() output."""

    def test_produces_aligned_pair(self) -> None:
        """CropShiftProducer.produce() returns AlignedPair."""
        producer = CropShiftProducer()
        data = torch.randn(4, 100, 3)
        result = producer.produce(data)

        assert isinstance(result, AlignedPair)

    def test_overlap_length_in_valid_range(self) -> None:
        """0 < overlap_length <= T for CropShiftProducer output."""
        producer = CropShiftProducer()
        seq_len = 100
        data = torch.randn(2, seq_len, 1)
        pair = producer.produce(data)

        assert 0 < pair.overlap_length <= seq_len

    def test_first_second_shapes(self) -> None:
        """first and second tensors have correct batch size and channels."""
        producer = CropShiftProducer()
        batch, channels = 4, 3
        data = torch.randn(batch, 100, channels)
        pair = producer.produce(data)

        assert pair.first.shape[0] == batch
        assert pair.first.shape[-1] == channels
        assert pair.second.shape[0] == batch
        assert pair.second.shape[-1] == channels

    def test_per_sample_crop_offsets(self) -> None:
        """CropShiftProducer preserves per-sample crop offsets (views differ)."""
        producer = CropShiftProducer()
        # Use identical rows so any difference must come from cropping
        row = torch.randn(100, 3)
        data = row.unsqueeze(0).repeat(4, 1, 1)
        pair = producer.produce(data)

        # At least two samples should differ (different offsets)
        # This is probabilistic but very likely with 4 samples
        all_equal = True
        for i in range(1, pair.first.shape[0]):
            if not torch.equal(pair.first[0], pair.first[i]):
                all_equal = False
                break
        assert not all_equal, 'Expected per-sample crop offsets to produce different views'


# --------------------------------------------------------------------------- #
# TS2Vec constructor acceptance tests
# --------------------------------------------------------------------------- #


class TestTS2VecConstructor:
    """TS2Vec accepts AugmentationProducer[AlignedPair]."""

    def test_accepts_crop_shift_producer(self) -> None:
        """TS2Vec constructor accepts CropShiftProducer."""
        producer = CropShiftProducer()
        model = TS2Vec(input_dims=1, augmentation=producer)

        assert model._augmentation is producer  # noqa: SLF001

    def test_accepts_full_overlap_pair_jitter(self) -> None:
        """TS2Vec constructor accepts FullOverlapPair(Jitter(...))."""
        jitter = Jitter()
        producer = FullOverlapPair(aug=jitter)
        model = TS2Vec(input_dims=1, augmentation=producer)

        assert model._augmentation is producer  # noqa: SLF001

    def test_default_is_crop_shift_producer(self) -> None:
        """TS2Vec default augmentation is CropShiftProducer."""
        model = TS2Vec(input_dims=1)

        assert isinstance(model._augmentation, CropShiftProducer)  # noqa: SLF001


# --------------------------------------------------------------------------- #
# TS2Vec training smoke tests with producers
# --------------------------------------------------------------------------- #


class TestTS2VecTraining:
    """TS2Vec trains with producer augmentations."""

    def test_trains_5_steps_with_crop_shift_producer(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
        finite_losses: Callable[..., None],
    ) -> None:
        """TS2Vec trains 5 steps with CropShiftProducer (finite loss)."""
        model = TS2Vec(
            input_dims=1,
            augmentation=CropShiftProducer(),
        )

        losses = train_steps(
            model=model,
            batch_size=4,
            seq_length=100,
            input_dims=1,
            num_steps=5,
        )

        finite_losses(losses, expected_min=5)

    def test_trains_5_steps_with_full_overlap_pair_jitter(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
        finite_losses: Callable[..., None],
    ) -> None:
        """TS2Vec trains 5 steps with FullOverlapPair(Jitter(...)) (finite loss)."""
        jitter = Jitter()
        producer = FullOverlapPair(aug=jitter)
        model = TS2Vec(input_dims=1, augmentation=producer)

        losses = train_steps(
            model=model,
            batch_size=4,
            seq_length=100,
            input_dims=1,
            num_steps=5,
        )

        finite_losses(losses, expected_min=5)


# --------------------------------------------------------------------------- #
# Determinism test (SC-7)
# --------------------------------------------------------------------------- #


class TestTS2VecDeterminism:
    """Seeded TS2Vec produces deterministic output (SC-7)."""

    def test_seeded_producer_identical_embeddings(self) -> None:
        """Two TS2Vec models with same seed produce identical embeddings."""
        seed = 42
        producer1 = Seeded(inner=CropShiftProducer(), seed=seed)
        producer2 = Seeded(inner=CropShiftProducer(), seed=seed)

        model1 = TS2Vec(input_dims=1, augmentation=producer1)
        model2 = TS2Vec(input_dims=1, augmentation=producer2)

        # Copy weights so encoders are identical
        model2.load_state_dict(deepcopy(model1.state_dict()))

        model1.eval()
        model2.eval()

        data = torch.randn(2, 50, 1)

        with torch.no_grad():
            emb1_a, emb2_a = model1._encode_augmented_views(data)  # noqa: SLF001
            emb1_b, emb2_b = model2._encode_augmented_views(data)  # noqa: SLF001

        torch.testing.assert_close(emb1_a, emb1_b)
        torch.testing.assert_close(emb2_a, emb2_b)
