"""Tests for augmentation primitives.

Covers Jitter, Scaling, TimestepScaling, Permutation and ComposeAugmentation.
Verifies that shared, model-agnostic primitives:
- Return same-shape tensors (no TrainingViews wrapping).
- Accept optional parameter dataclasses with correct defaults.
- Chain correctly via ComposeAugmentation.
- Satisfy the Augmentation Protocol (callable, Tensor -> Tensor).
"""

from dataclasses import is_dataclass

import pytest
import torch

from chronocratic.models.augmentation.base import Augmentation
from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
    TimestepScaling,
    TimestepScalingParameters,
)


class TestJitter:
    """Jitter adds Gaussian noise, returns same-shape Tensor."""

    def test_default_params_produce_same_shape_output(self) -> None:
        jitter = Jitter()
        data = torch.randn(4, 3, 100)
        result = jitter(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape

    def test_default_params_add_noise(self) -> None:
        jitter = Jitter()
        torch.manual_seed(42)
        data = torch.zeros(4, 3, 100)
        result = jitter(data)
        assert not torch.equal(result, data)

    def test_p_equals_zero_returns_unchanged(self) -> None:
        jitter = Jitter(params=JitterParameters(p=0.0))
        data = torch.randn(4, 3, 100)
        result = jitter(data)
        torch.testing.assert_close(result, data)


class TestScaling:
    """Scaling multiplies by per-channel factor, returns same-shape Tensor."""

    def test_default_params_produce_same_shape_output(self) -> None:
        scaling = Scaling()
        data = torch.randn(4, 3, 100)
        result = scaling(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape

    def test_per_sample_draws_independent_factor(self) -> None:
        scaling = Scaling(params=ScalingParameters(per_sample=True))
        data = torch.ones(4, 2, 10)
        result = scaling(data)
        # Each batch element should have a different scale applied
        # (with high probability given sigma=0.1, mean=1.0)
        ratios = result[:, :, 0] / data[:, :, 0]
        assert not torch.allclose(ratios, ratios[0:1])


class TestTimestepScaling:
    """TimestepScaling: per-timestep factor, broadcast across channels."""

    def test_satisfies_augmentation_protocol(self) -> None:
        assert isinstance(TimestepScaling(), Augmentation)

    def test_keeps_shape(self) -> None:
        x = torch.ones(4, 50, 3)
        assert TimestepScaling()(x).shape == x.shape

    def test_factor_varies_across_timesteps(self) -> None:
        torch.manual_seed(0)
        factor = TimestepScaling()(torch.ones(2, 50, 1))
        assert factor[0, :, 0].std().item() > 0.1

    def test_factor_broadcast_across_channels(self) -> None:
        torch.manual_seed(0)
        factor = TimestepScaling()(torch.ones(2, 50, 3))
        assert torch.equal(factor[..., 0], factor[..., 1])
        assert torch.equal(factor[..., 0], factor[..., 2])

    def test_factor_distribution_follows_params(self) -> None:
        torch.manual_seed(0)
        params = TimestepScalingParameters(sigma=0.5, mean=3.0)
        factor = TimestepScaling(params)(torch.ones(64, 256, 1))
        assert factor.mean().item() == pytest.approx(3.0, abs=0.02)
        assert factor.std().item() == pytest.approx(0.5, abs=0.02)

    def test_defaults_are_tstcc_reference(self) -> None:
        assert TimestepScalingParameters() == TimestepScalingParameters(sigma=1.1, mean=2.0)


class TestPermutation:
    """Permutation permutes time segments, returns same-shape Tensor."""

    def test_produces_same_shape_output(self) -> None:
        perm = Permutation()
        data = torch.randn(4, 3, 100)
        result = perm(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape

    def test_segments_reordered(self) -> None:
        perm = Permutation(params=PermutationParameters(max_segments=5, time_dim=-1))
        data = torch.arange(100, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Run with multiple seeds to find one that actually permutes
        reordered = False
        for seed in range(10):
            torch.manual_seed(seed)
            result = perm(data)
            if not torch.equal(result, data):
                reordered = True
                break
        assert reordered, "Permutation should reorder segments for at least one seed"


class TestComposeAugmentation:
    """ComposeAugmentation chains primitives sequentially."""

    def test_chains_jitter_and_scaling(self) -> None:
        jitter = Jitter(params=JitterParameters(sigma=0.1))
        scaling = Scaling(params=ScalingParameters(sigma=0.05))
        compose = ComposeAugmentation([jitter, scaling])
        data = torch.randn(4, 3, 100)
        result = compose(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == data.shape


class TestCallableProtocol:
    """All primitives implement __call__ (satisfy Augmentation Protocol)."""

    def test_jitter_is_callable(self) -> None:
        jitter = Jitter()
        data = torch.randn(2, 1, 50)
        result = jitter(data)
        assert isinstance(result, torch.Tensor)

    def test_scaling_is_callable(self) -> None:
        scaling = Scaling()
        data = torch.randn(2, 1, 50)
        result = scaling(data)
        assert isinstance(result, torch.Tensor)

    def test_permutation_is_callable(self) -> None:
        perm = Permutation()
        data = torch.randn(2, 1, 50)
        result = perm(data)
        assert isinstance(result, torch.Tensor)

    def test_compose_is_callable(self) -> None:
        compose = ComposeAugmentation([Jitter(), Scaling()])
        data = torch.randn(2, 1, 50)
        result = compose(data)
        assert isinstance(result, torch.Tensor)


class TestParameterDataclasses:
    """Parameter dataclasses have correct defaults."""

    def test_jitter_parameters_defaults(self) -> None:
        params = JitterParameters()
        assert params.sigma == 0.1
        assert params.p == 1.0

    def test_scaling_parameters_defaults(self) -> None:
        params = ScalingParameters()
        assert params.sigma == 0.1
        assert params.mean == 1.0
        assert params.p == 1.0
        assert params.per_sample is False
        assert params.channel_dim == 1

    def test_permutation_parameters_defaults(self) -> None:
        params = PermutationParameters()
        assert params.max_segments == 5
        assert params.time_dim == -1

    def test_all_are_dataclasses(self) -> None:
        assert is_dataclass(JitterParameters)
        assert is_dataclass(ScalingParameters)
        assert is_dataclass(PermutationParameters)
