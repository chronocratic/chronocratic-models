"""Tests for PerTimestepScaling, shared by TS-TCC and SimCLR."""

import pytest
import torch

from chronocratic.models.augmentation.base import Augmentation
from chronocratic.models.convolutional.standard.augmentation import (
    PerTimestepScaling,
    PerTimestepScalingParameters,
)


class TestPerTimestepScaling:
    """PerTimestepScaling: one factor per timestep, shared across channels."""

    def test_satisfies_augmentation_protocol(self) -> None:
        assert isinstance(PerTimestepScaling(), Augmentation)

    def test_keeps_shape(self) -> None:
        x = torch.ones(4, 50, 3)
        assert PerTimestepScaling()(x).shape == x.shape

    def test_factor_varies_across_timesteps(self) -> None:
        torch.manual_seed(0)
        factor = PerTimestepScaling()(torch.ones(2, 50, 1))
        assert factor[0, :, 0].std().item() > 0.1

    def test_factor_broadcast_across_channels(self) -> None:
        torch.manual_seed(0)
        factor = PerTimestepScaling()(torch.ones(2, 50, 3))
        assert torch.equal(factor[..., 0], factor[..., 1])
        assert torch.equal(factor[..., 0], factor[..., 2])

    def test_factor_distribution_follows_params(self) -> None:
        torch.manual_seed(0)
        params = PerTimestepScalingParameters(sigma=0.5, mean=3.0)
        factor = PerTimestepScaling(params)(torch.ones(64, 256, 1))
        assert factor.mean().item() == pytest.approx(3.0, abs=0.02)
        assert factor.std().item() == pytest.approx(0.5, abs=0.02)

    def test_defaults_are_tstcc_reference(self) -> None:
        assert PerTimestepScalingParameters() == PerTimestepScalingParameters(sigma=1.1, mean=2.0)
