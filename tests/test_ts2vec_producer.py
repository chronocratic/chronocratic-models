"""TDD tests for TS2Vec producer integration.

Verifies CropShiftProducer returns AlignedPair, TS2Vec accepts
AugmentationProducer[AlignedPair], and training runs with finite loss.
"""

import math
from copy import deepcopy

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from tscollection.models.augmentation.base import AlignedPair
from tscollection.models.augmentation.decorators import Seeded
from tscollection.models.augmentation.primitives import Jitter
from tscollection.models.augmentation.producers import FullOverlapPair


def _train_steps(
    model: pl.LightningModule,
    batch_size: int,
    seq_length: int,
    input_dims: int,
    num_steps: int,
) -> list[torch.Tensor]:
    """Run ``num_steps`` training steps via a minimal Lightning Trainer."""
    data = torch.randn(batch_size * num_steps, seq_length, input_dims)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model._test_losses = []  # type: ignore[attr-defined]

    original_training_step = model.training_step

    def _wrapped_training_step(
        batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        loss = original_training_step(batch, batch_idx)
        if loss is not None:
            model._test_losses.append(loss.detach())  # type: ignore[attr-defined]
        return loss

    model.training_step = _wrapped_training_step  # type: ignore[method-assign]

    trainer = pl.Trainer(
        accelerator='cpu',
        max_steps=num_steps,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    return model._test_losses  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# CropShiftProducer unit tests
# --------------------------------------------------------------------------- #


class TestCropShiftProducer:
    """Tests for CropShiftProducer produce() output."""

    def test_produces_aligned_pair(self) -> None:
        """CropShiftProducer.produce() returns AlignedPair."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        producer = CropShiftProducer()
        data = torch.randn(4, 100, 3)
        result = producer.produce(data)

        assert isinstance(result, AlignedPair)

    def test_overlap_length_in_valid_range(self) -> None:
        """0 < overlap_length <= T for CropShiftProducer output."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

        producer = CropShiftProducer()
        T = 100
        data = torch.randn(2, T, 1)
        pair = producer.produce(data)

        assert 0 < pair.overlap_length <= T

    def test_first_second_shapes(self) -> None:
        """first and second tensors have correct batch size and channels."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

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
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )

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
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

        producer = CropShiftProducer()
        model = TS2Vec(input_dims=1, augmentation=producer)

        assert model._augmentation is producer

    def test_accepts_full_overlap_pair_jitter(self) -> None:
        """TS2Vec constructor accepts FullOverlapPair(Jitter(...))."""
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

        jitter = Jitter()
        producer = FullOverlapPair(aug=jitter)
        model = TS2Vec(input_dims=1, augmentation=producer)

        assert model._augmentation is producer

    def test_default_is_crop_shift_producer(self) -> None:
        """TS2Vec default augmentation is CropShiftProducer."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

        model = TS2Vec(input_dims=1)

        assert isinstance(model._augmentation, CropShiftProducer)


# --------------------------------------------------------------------------- #
# TS2Vec training smoke tests with producers
# --------------------------------------------------------------------------- #


class TestTS2VecTraining:
    """TS2Vec trains with producer augmentations."""

    def test_trains_5_steps_with_crop_shift_producer(self) -> None:
        """TS2Vec trains 5 steps with CropShiftProducer (finite loss)."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

        model = TS2Vec(
            input_dims=1,
            augmentation=CropShiftProducer(),
        )

        losses = _train_steps(
            model=model,
            batch_size=4,
            seq_length=100,
            input_dims=1,
            num_steps=5,
        )

        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, 'Loss must be a scalar tensor'
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )

    def test_trains_5_steps_with_full_overlap_pair_jitter(self) -> None:
        """TS2Vec trains 5 steps with FullOverlapPair(Jitter(...)) (finite loss)."""
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

        jitter = Jitter()
        producer = FullOverlapPair(aug=jitter)
        model = TS2Vec(input_dims=1, augmentation=producer)

        losses = _train_steps(
            model=model,
            batch_size=4,
            seq_length=100,
            input_dims=1,
            num_steps=5,
        )

        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, 'Loss must be a scalar tensor'
            assert math.isfinite(loss.item()), (
                f'Loss at step {step_idx} is not finite: {loss.item()}'
            )


# --------------------------------------------------------------------------- #
# Determinism test (SC-7)
# --------------------------------------------------------------------------- #


class TestTS2VecDeterminism:
    """Seeded TS2Vec produces deterministic output (SC-7)."""

    def test_seeded_producer_identical_embeddings(self) -> None:
        """Two TS2Vec models with same seed produce identical embeddings."""
        from tscollection.models.convolutional.dilated.ts2vec.augmentation import (
            CropShiftProducer,
        )
        from tscollection.models.convolutional.dilated.ts2vec.model import TS2Vec

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
            emb1_a, emb2_a = model1._encode_augmented_views(data)
            emb1_b, emb2_b = model2._encode_augmented_views(data)

        torch.testing.assert_close(emb1_a, emb1_b)
        torch.testing.assert_close(emb2_a, emb2_b)
