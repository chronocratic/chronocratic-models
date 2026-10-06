"""TDD tests for TSTCC producer integration and its model-local augmentations.

Verifies TSTCC accepts AugmentationProducer[ViewPair] via
_default_tstcc_pair(), uses .produce().first/.second, and trains with finite loss.
Also covers TSTCCScaling, the pair builders' composition, and instance
normalization in training and encoding.
"""

from collections.abc import Callable

import pytest
import torch

from chronocratic.models.augmentation.base import Augmentation, ViewPair
from chronocratic.models.augmentation.primitives import ComposeAugmentation, Jitter, Permutation
from chronocratic.models.augmentation.producers import RolePairProducer
from chronocratic.models.convolutional.standard.tstcc import model as tstcc_model
from chronocratic.models.convolutional.standard.tstcc.augmentations import (
    _default_tstcc_pair,
    TSTCCScaling,
    TSTCCScalingParameters,
)
from chronocratic.models.convolutional.standard.tstcc.model import _instance_normalize, TSTCC


class TestDefaultTSTCCPair:
    """_default_tstcc_pair() builder function."""

    def test_returns_role_pair_type(self) -> None:
        producer = _default_tstcc_pair()
        assert isinstance(producer, RolePairProducer)

    def test_produce_returns_view_pair(self, random_data: Callable[..., torch.Tensor]) -> None:
        producer = _default_tstcc_pair()
        x = random_data(batch=2, seq_length=50, input_dim=3, layout="NTC")
        result = producer.produce(x)

        assert isinstance(result, ViewPair)
        assert result.first.shape == x.shape
        assert result.second.shape == x.shape
        assert not torch.allclose(result.first, result.second)

    def test_satisfies_protocol(self, random_data: Callable[..., torch.Tensor]) -> None:
        producer = _default_tstcc_pair()
        assert hasattr(producer, "produce")
        x = random_data(batch=4, seq_length=100, input_dim=1, layout="NTC")
        result = producer.produce(x)
        assert isinstance(result, ViewPair)


class TestTSTCCConstructor:
    """TSTCC constructor with new producer contract."""

    def test_accepts_default_tstcc_pair(self) -> None:
        producer = _default_tstcc_pair()
        model = TSTCC(
            input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16, augmentation=producer
        )
        assert model._augmentation is producer

    def test_default_producer_is_role_pair(self) -> None:
        model = TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16)
        assert isinstance(model._augmentation, RolePairProducer)


class TestTSTCCTraining:
    """TSTCC training with new producer contract."""

    def test_compute_loss_uses_produce_first_second(self) -> None:
        model = TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16)
        data = torch.randn(4, 100, 1)  # (B, T, C)
        labels = torch.zeros(4, dtype=torch.long)
        batch = (data, labels)

        loss, batch_size = model._compute_loss(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert batch_size == 4

    @pytest.mark.skip(reason="slow: Lightning trainer overhead")
    def test_trains_with_finite_loss(
        self, train_steps: Callable[..., list[torch.Tensor]], finite_losses: Callable[..., None]
    ) -> None:
        model = TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16)

        losses = train_steps(
            model,
            batch_size=2,
            seq_length=100,
            input_dim=1,
            num_steps=1,
            layout="NTC",
            with_labels=True,
        )
        finite_losses(losses, expected_min=1)


class TestDeterminism:
    """Seeded TSTCC produces identical loss across runs (SC-7)."""

    @pytest.mark.skip(reason="slow: Lightning trainer overhead")
    def test_seeded_determinism(self, train_steps: Callable[..., list[torch.Tensor]]) -> None:
        losses_list: list[list[torch.Tensor]] = []

        for _run in range(2):
            torch.manual_seed(12345)
            model = TSTCC(input_dim=1, conv_kernel_size=5, stride=1, representation_dim=16)

            losses = train_steps(
                model,
                batch_size=2,
                seq_length=100,
                input_dim=1,
                num_steps=1,
                seed=12345,
                layout="NTC",
            )
            losses_list.append(losses)

        assert len(losses_list[0]) == len(losses_list[1])
        for i, (a, b) in enumerate(zip(losses_list[0], losses_list[1], strict=True)):
            assert abs(a.item() - b.item()) < 1e-5, (
                f"Loss at step {i} differs: {a.item()} vs {b.item()}"
            )


def _strong_view_parts(producer: object) -> list[object]:
    """Return the strong view's primitives, flattening a ComposeAugmentation."""
    assert isinstance(producer, RolePairProducer)
    strong = producer._second
    if isinstance(strong, ComposeAugmentation):
        return list(strong._augmentations)
    return [strong]


class TestTSTCCScaling:
    """TSTCCScaling: upstream per-timestep scaling, broadcast across channels."""

    def test_satisfies_augmentation_protocol(self) -> None:
        assert isinstance(TSTCCScaling(), Augmentation)

    def test_keeps_shape(self) -> None:
        x = torch.ones(4, 50, 3)
        assert TSTCCScaling()(x).shape == x.shape

    def test_factor_varies_across_timesteps(self) -> None:
        torch.manual_seed(0)
        factor = TSTCCScaling()(torch.ones(2, 50, 1))
        assert factor[0, :, 0].std().item() > 0.1

    def test_factor_broadcast_across_channels(self) -> None:
        torch.manual_seed(0)
        factor = TSTCCScaling()(torch.ones(2, 50, 3))
        assert torch.equal(factor[..., 0], factor[..., 1])
        assert torch.equal(factor[..., 0], factor[..., 2])

    def test_factor_distribution_follows_params(self) -> None:
        torch.manual_seed(0)
        params = TSTCCScalingParameters(sigma=0.5, mean=3.0)
        factor = TSTCCScaling(params)(torch.ones(64, 256, 1))
        assert factor.mean().item() == pytest.approx(3.0, abs=0.02)
        assert factor.std().item() == pytest.approx(0.5, abs=0.02)


class TestPairComposition:
    """_default_tstcc_pair() composition."""

    def test_default_weak_view_is_tstcc_scaling(self) -> None:
        producer = _default_tstcc_pair()
        assert isinstance(producer, RolePairProducer)
        assert isinstance(producer._first, TSTCCScaling)

    def test_default_has_no_permutation(self) -> None:
        parts = _strong_view_parts(_default_tstcc_pair())
        assert len(parts) == 1
        assert isinstance(parts[0], Jitter)
        assert parts[0]._params.sigma == pytest.approx(0.8)

    def test_permutation_present_when_max_segments_above_one(self) -> None:
        parts = _strong_view_parts(_default_tstcc_pair(max_segments=3, jitter_sigma=0.2))
        assert [type(part) for part in parts] == [Permutation, Jitter]
        assert parts[0]._params.max_segments == 3
        assert parts[0]._params.time_dim == 1
        assert parts[1]._params.sigma == pytest.approx(0.2)

    def test_rejects_max_segments_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_segments"):
            _default_tstcc_pair(max_segments=0)

    def test_max_segments_8_is_upstream_har_recipe(self) -> None:
        producer = _default_tstcc_pair(max_segments=8)
        assert isinstance(producer, RolePairProducer)
        weak = producer._first
        assert isinstance(weak, TSTCCScaling)
        assert weak._params == TSTCCScalingParameters(sigma=1.1, mean=2.0)
        parts = _strong_view_parts(producer)
        assert [type(part) for part in parts] == [Permutation, Jitter]
        assert parts[0]._params.max_segments == 8
        assert parts[1]._params.sigma == pytest.approx(0.8)


class TestInstanceNormalize:
    """_instance_normalize and its use in training and encoding."""

    def test_zero_mean_unit_std_per_series_and_channel(self) -> None:
        torch.manual_seed(0)
        x = torch.rand(4, 100, 3) * 5.0 + 2.0
        normalized = _instance_normalize(x)
        assert torch.allclose(normalized.mean(dim=1), torch.zeros(4, 3), atol=1e-5)
        assert torch.allclose(normalized.std(dim=1, unbiased=False), torch.ones(4, 3), atol=1e-4)

    def test_constant_series_gives_zeros(self) -> None:
        normalized = _instance_normalize(torch.full((2, 50, 1), 0.7))
        assert not torch.isnan(normalized).any()
        assert torch.equal(normalized, torch.zeros_like(normalized))

    @pytest.mark.parametrize("instance_normalize", [True, False])
    def test_compute_loss_normalizes_only_when_enabled(
        self, *, instance_normalize: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[torch.Tensor] = []

        def recording_normalize(x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return _instance_normalize(x)

        monkeypatch.setattr(tstcc_model, "_instance_normalize", recording_normalize)
        model = TSTCC(
            input_dim=1,
            conv_kernel_size=5,
            representation_dim=16,
            instance_normalize=instance_normalize,
        )
        model._compute_loss((torch.rand(4, 100, 1), torch.zeros(4)))
        assert len(calls) == int(instance_normalize)

    @pytest.mark.parametrize("instance_normalize", [True, False])
    def test_encode_batch_normalizes_only_when_enabled(self, *, instance_normalize: bool) -> None:
        torch.manual_seed(0)
        model = TSTCC(
            input_dim=1,
            conv_kernel_size=5,
            representation_dim=16,
            instance_normalize=instance_normalize,
        )
        model.eval()
        x = torch.rand(4, 100, 1)
        with torch.no_grad():
            encoded = model._encode_batch(model.encoder, x * 10.0 + 3.0)
            reference_input = _instance_normalize(x) if instance_normalize else x * 10.0 + 3.0
            reference = model.encoder(reference_input).mean(dim=-1)
        assert torch.allclose(encoded, reference, atol=1e-5)

    def test_default_is_enabled(self) -> None:
        model = TSTCC(input_dim=1, conv_kernel_size=5, representation_dim=16)
        assert model._instance_normalize is True
        assert model.hparams["instance_normalize"] is True
