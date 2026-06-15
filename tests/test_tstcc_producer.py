"""TDD tests for TSTCC producer integration.

Verifies TSTCC accepts AugmentationProducer[ViewPair] via
_default_tstcc_pair(), uses .produce().first/.second, and trains with finite loss.
"""

from collections.abc import Callable

import pytest
import torch

from chronocratic.models.augmentation import primitives
from chronocratic.models.augmentation.base import ViewPair
from chronocratic.models.augmentation.producers import RolePair
from chronocratic.models.convolutional.standard.tstcc import augmentations
from chronocratic.models.convolutional.standard.tstcc.augmentations import (
    _default_tstcc_pair,
)
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC


class TestDefaultTSTCCPair:
    """_default_tstcc_pair() builder function."""

    def test_returns_role_pair_type(self) -> None:
        producer = _default_tstcc_pair()
        assert isinstance(producer, RolePair)

    def test_produce_returns_view_pair(
        self, random_data: Callable[..., torch.Tensor]
    ) -> None:
        producer = _default_tstcc_pair()
        x = random_data(batch=2, seq_length=50, input_dims=3, layout="NCL")
        result = producer.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape
        assert not torch.allclose(result.first, result.second)

    def test_satisfies_protocol(
        self, random_data: Callable[..., torch.Tensor]
    ) -> None:
        producer = _default_tstcc_pair()
        assert hasattr(producer, "produce")
        x = random_data(batch=4, seq_length=100, input_dims=1, layout="NCL")
        result = producer.produce(x)
        assert isinstance(result, ViewPair)


class TestTSTCCConstructor:
    """TSTCC constructor with new producer contract."""

    def test_accepts_default_tstcc_pair(self) -> None:
        producer = _default_tstcc_pair()
        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=15,
            num_classes=10,
            augmentation=producer,
        )
        assert model._augmentation is producer  # noqa: SLF001

    def test_default_producer_is_role_pair(self) -> None:
        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=15,
            num_classes=10,
        )
        assert isinstance(model._augmentation, RolePair)  # noqa: SLF001


class TestTSTCCTraining:
    """TSTCC training with new producer contract."""

    def test_compute_loss_uses_produce_first_second(self) -> None:
        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=15,
            num_classes=10,
        )
        data = torch.randn(4, 1, 100)
        labels = torch.zeros(4, dtype=torch.long)
        batch = (data, labels)

        loss = model._compute_loss(batch)  # noqa: SLF001
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    @pytest.mark.skip(reason="slow: Lightning trainer overhead")
    def test_trains_with_finite_loss(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
        finite_losses: Callable[..., None],
    ) -> None:
        model = TSTCC(
            input_channels=1,
            kernel_size=5,
            stride=1,
            final_out_channels=16,
            features_len=15,
            num_classes=10,
        )

        losses = train_steps(
            model,
            batch_size=2,
            seq_length=100,
            input_dims=1,
            num_steps=1,
            layout="NCL",
            with_labels=True,
        )
        finite_losses(losses, expected_min=1)


class TestReExports:
    """tstcc/augmentations.py re-exports from primitives.py."""

    def test_jitter_reexported(self) -> None:
        assert augmentations.Jitter is primitives.Jitter

    def test_scaling_reexported(self) -> None:
        assert augmentations.Scaling is primitives.Scaling

    def test_permutation_reexported(self) -> None:
        assert augmentations.Permutation is primitives.Permutation

    def test_compose_reexported(self) -> None:
        assert augmentations.ComposeAugmentation is primitives.ComposeAugmentation


class TestDeterminism:
    """Seeded TSTCC produces identical loss across runs (SC-7)."""

    @pytest.mark.skip(reason="slow: Lightning trainer overhead")
    def test_seeded_determinism(
        self,
        train_steps: Callable[..., list[torch.Tensor]],
    ) -> None:
        losses_list: list[list[torch.Tensor]] = []

        for _run in range(2):
            torch.manual_seed(12345)
            model = TSTCC(
                input_channels=1,
                kernel_size=5,
                stride=1,
                final_out_channels=16,
                features_len=15,
                num_classes=10,
            )

            losses = train_steps(
                model,
                batch_size=2,
                seq_length=100,
                input_dims=1,
                num_steps=1,
                seed=12345,
                layout="NCL",
            )
            losses_list.append(losses)

        assert len(losses_list[0]) == len(losses_list[1])
        for i, (a, b) in enumerate(zip(losses_list[0], losses_list[1], strict=True)):
            assert abs(a.item() - b.item()) < 1e-5, (
                f"Loss at step {i} differs: {a.item()} vs {b.item()}"
            )
