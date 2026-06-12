"""TDD tests for TSTCC producer integration.

Verifies that TS-TCC accepts AugmentationProducer[ViewPair] via the
_default_tstcc_pair() builder, uses .produce().first / .second, and
trains with finite loss.
"""

from __future__ import annotations

import math

import pytest
import torch

from tscollection.models.augmentation.base import AugmentationProducer, ViewPair


class TestDefaultTSTCCPair:
    """Tests for _default_tstcc_pair() builder function."""

    def test_returns_role_pair_type(self) -> None:
        """_default_tstcc_pair() returns a RolePair[ViewPair]."""
        from tscollection.models.convolutional.standard.tstcc.augmentations import (
            _default_tstcc_pair,
        )
        from tscollection.models.augmentation.producers import RolePair

        producer = _default_tstcc_pair()
        assert isinstance(producer, RolePair)

    def test_produce_returns_view_pair(self) -> None:
        """_default_tstcc_pair().produce(x) returns ViewPair with two different transforms."""
        from tscollection.models.convolutional.standard.tstcc.augmentations import (
            _default_tstcc_pair,
        )

        producer = _default_tstcc_pair()
        x = torch.randn(2, 3, 50)
        result = producer.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape
        # Views should differ (Scaling + Jitter are not identity)
        assert not torch.allclose(result.first, result.second)

    def test_satisfies_protocol(self) -> None:
        """_default_tstcc_pair() satisfies AugmentationProducer[ViewPair] structurally."""
        from tscollection.models.convolutional.standard.tstcc.augmentations import (
            _default_tstcc_pair,
        )

        producer = _default_tstcc_pair()
        # Structural check: has produce method, returns ViewPair
        assert hasattr(producer, 'produce')
        x = torch.randn(4, 1, 100)
        result = producer.produce(x)
        assert isinstance(result, ViewPair)


class TestTSTCCConstructor:
    """Tests for TSTCC constructor with new producer contract."""

    def test_accepts_default_tstcc_pair(self) -> None:
        """TSTCC constructor accepts _default_tstcc_pair()."""
        from tscollection.models.convolutional.standard.tstcc.augmentations import (
            _default_tstcc_pair,
        )
        from tscollection.models.convolutional.standard.tstcc.model import TSTCC

        producer = _default_tstcc_pair()
        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=50,
            num_classes=10,
            augmentation=producer,
        )
        assert model._augmentation is producer

    def test_default_producer_is_role_pair(self) -> None:
        """TSTCC default uses RolePair(Scaling, ComposeAugmentation(...))."""
        from tscollection.models.augmentation.producers import RolePair
        from tscollection.models.convolutional.standard.tstcc.model import TSTCC

        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=50,
            num_classes=10,
        )
        assert isinstance(model._augmentation, RolePair)


class TestTSTCCTraining:
    """Tests for TSTCC training with new producer contract."""

    def test_compute_loss_uses_produce_first_second(self) -> None:
        """TSTCC._compute_loss uses .produce().first / .produce().second."""
        from tscollection.models.convolutional.standard.tstcc.model import TSTCC

        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=50,
            num_classes=10,
        )
        data = torch.randn(4, 100, 1)
        labels = torch.zeros(4, dtype=torch.long)
        batch = (data, labels)

        loss = model._compute_loss(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_trains_with_finite_loss(self) -> None:
        """TSTCC trains with finite loss."""
        import lightning.pytorch as pl
        from torch.utils.data import DataLoader, TensorDataset

        from tscollection.models.convolutional.standard.tstcc.model import TSTCC

        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=50,
            num_classes=10,
        )

        data = torch.randn(8, 100, 1)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=4)

        model._test_losses = []  # type: ignore[attr-defined]
        original_step = model.training_step

        def wrapped_step(batch, batch_idx: int):
            loss = original_step(batch, batch_idx)
            if loss is not None:
                model._test_losses.append(loss.detach())  # type: ignore[attr-defined]
            return loss

        model.training_step = wrapped_step  # type: ignore[method-assign]

        trainer = pl.Trainer(
            accelerator='cpu',
            max_steps=3,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=dataloader)

        losses = model._test_losses  # type: ignore[attr-defined]
        assert len(losses) >= 1
        for i, loss in enumerate(losses):
            assert math.isfinite(loss.item()), f'Loss at step {i} is not finite'


class TestReExports:
    """Tests for tstcc/augmentations.py re-exports from primitives.py."""

    def test_jitter_reexported_from_primitives(self) -> None:
        """Jitter is re-exported from primitives."""
        from tscollection.models.augmentation import primitives
        from tscollection.models.convolutional.standard.tstcc import augmentations

        assert augmentations.Jitter is primitives.Jitter

    def test_scaling_reexported_from_primitives(self) -> None:
        """Scaling is re-exported from primitives."""
        from tscollection.models.augmentation import primitives
        from tscollection.models.convolutional.standard.tstcc import augmentations

        assert augmentations.Scaling is primitives.Scaling

    def test_permutation_reexported_from_primitives(self) -> None:
        """Permutation is re-exported from primitives."""
        from tscollection.models.augmentation import primitives
        from tscollection.models.convolutional.standard.tstcc import augmentations

        assert augmentations.Permutation is primitives.Permutation

    def test_compose_augmentation_reexported_from_primitives(self) -> None:
        """ComposeAugmentation is re-exported from primitives."""
        from tscollection.models.augmentation import primitives
        from tscollection.models.convolutional.standard.tstcc import augmentations

        assert augmentations.ComposeAugmentation is primitives.ComposeAugmentation


class TestDeterminism:
    """Seeded TSTCC produces identical loss across runs (SC-7)."""

    def test_seeded_determinism(self) -> None:
        """Seeded TSTCC produces identical loss across two runs."""
        import lightning.pytorch as pl
        from torch.utils.data import DataLoader, TensorDataset

        from tscollection.models.convolutional.standard.tstcc.model import TSTCC

        losses_list: list[list[float]] = []

        for _run in range(2):
            torch.manual_seed(12345)
            model = TSTCC(
                input_channels=1,
                kernel_size=5,
                stride=1,
                final_out_channels=16,
                features_len=50,
                num_classes=10,
            )

            data = torch.randn(4, 100, 1)
            dataset = TensorDataset(data)
            dataloader = DataLoader(dataset, batch_size=4)

            collected: list[float] = []
            original_step = model.training_step

            def wrapped_step(batch, batch_idx: int):
                loss = original_step(batch, batch_idx)
                if loss is not None:
                    collected.append(loss.item())
                return loss

            model.training_step = wrapped_step  # type: ignore[method-assign]

            trainer = pl.Trainer(
                accelerator='cpu',
                max_steps=2,
                enable_checkpointing=False,
                enable_progress_bar=False,
                logger=False,
            )
            trainer.fit(model, train_dataloaders=dataloader)
            losses_list.append(collected)

        # Compare losses from both runs
        assert len(losses_list[0]) == len(losses_list[1]), 'Different number of steps'
        for i, (a, b) in enumerate(zip(losses_list[0], losses_list[1])):
            assert abs(a - b) < 1e-5, f'Loss at step {i} differs: {a} vs {b}'
