"""NaN-padded input defense for BasicEncodingMixin encode() paths.

Verifies that all BasicEncodingMixin models produce finite representations
when encode(), encode_batch(), forward(), or predict() receive
variable-length series padded with trailing NaN timesteps.

Tests will fail RED until zero_fill_padding guards are wired into each
model's _encode_batch (and TimeVAE's forward/predict).
"""

from __future__ import annotations

__all__ = ["test_timevae_encode_nan_padded_vector"]

import numpy as np
import pytest
import torch

from chronocratic.models.generative.timevae.model import TimeVAE


# --------------------------------------------------------------------------- #
# Fixtures — tiny models for fast testing
# --------------------------------------------------------------------------- #


@pytest.fixture
def timevae() -> TimeVAE:
    """Small TimeVAE for fast testing."""
    return TimeVAE(
        sequence_length=32,
        input_dim=3,
        latent_dim=8,
        hidden_layer_sizes=(16, 32),
        conv_kernel_size=3,
        conv_stride=2,
    )


# --------------------------------------------------------------------------- #
# NaN-padded input test helper
# --------------------------------------------------------------------------- #


def _make_nan_padded(shape: tuple[int, int, int]) -> torch.Tensor:
    """Create a tensor with NaN padding on the last few timesteps."""
    x = torch.randn(*shape)
    # Pad last 3 timesteps with NaN
    x[:, -3:, :] = float("nan")
    return x


# --------------------------------------------------------------------------- #
# TimeVAE — encode() NaN tests
# --------------------------------------------------------------------------- #


def test_timevae_encode_nan_padded_vector(timevae: TimeVAE) -> None:
    """TimeVAE encode() on NaN-padded input returns finite VECTOR."""
    data = _make_nan_padded((4, 32, 3))
    reps = timevae.encode(data, batch_size=4)
    assert torch.isfinite(reps).all(), "TimeVAE encode() VECTOR contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TimeVAE — encode_batch() NaN tests
# --------------------------------------------------------------------------- #


def test_timevae_encode_batch_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE encode_batch() on NaN-padded input returns finite output."""
    batch_x = _make_nan_padded((4, 32, 3))
    reps = timevae.encode_batch(batch_x)
    assert torch.isfinite(reps).all(), "TimeVAE encode_batch() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TimeVAE — forward() NaN tests
# --------------------------------------------------------------------------- #


def test_timevae_forward_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE forward() on NaN-padded input returns finite reconstruction."""
    x = _make_nan_padded((4, 32, 3))
    out = timevae(x)
    assert torch.isfinite(out).all(), "TimeVAE forward() contains NaN/Inf"


# --------------------------------------------------------------------------- #
# TimeVAE — predict() NaN tests
# --------------------------------------------------------------------------- #


def test_timevae_predict_nan_padded(timevae: TimeVAE) -> None:
    """TimeVAE predict() on NaN-padded NumPy input returns finite reconstruction."""
    x = _make_nan_padded((4, 32, 3))
    x_np = x.numpy()
    out = timevae.predict(x_np)
    assert np.isfinite(out).all(), "TimeVAE predict() contains NaN/Inf"
