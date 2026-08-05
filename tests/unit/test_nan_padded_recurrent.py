"""NaN-padded batch defense for RecurrentAE and TimeNet.

Verifies that both reconstruction models produce finite loss when
trained on variable-length UEA series padded with trailing NaN timesteps.

Tests will fail RED until NaN defense (zero_fill_padding + masked_reconstruction_loss)
is wired into the model training_step and validation_step.
"""

from __future__ import annotations

import pytest
import torch

from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder
from chronocratic.models.recurrent.timenet.model import TimeNet

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def recurrentae() -> RecurrentAutoEncoder:
    """Small RecurrentAE for fast testing."""
    return RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0)


@pytest.fixture
def timenet() -> TimeNet:
    """Small TimeNet for fast testing."""
    return TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)


# --------------------------------------------------------------------------- #
# RecurrentAE — NaN-padded tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", ["mse", "mae"])
def test_recurrentae_nan_padded_training(loss: str) -> None:
    """NaN-padded batch -> training_step produces finite loss."""
    model = RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0, loss=loss)
    x = torch.randn(4, 32, 3)
    x[:, -3:, :] = float("nan")  # trailing NaN padding
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"training_step loss is not finite: {loss_tensor}"


@pytest.mark.parametrize("loss", ["mse", "mae"])
def test_recurrentae_nan_padded_validation(loss: str) -> None:
    """NaN-padded batch -> validation_step produces finite loss."""
    model = RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0, loss=loss)
    x = torch.randn(4, 32, 3)
    x[:, -3:, :] = float("nan")  # trailing NaN padding
    batch = (x,)
    loss_tensor = model.validation_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"validation_step loss is not finite: {loss_tensor}"


def test_recurrentae_all_nan_batch() -> None:
    """Fully-NaN batch -> finite loss, no crash."""
    model = RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0)
    x = torch.full((4, 32, 3), float("nan"))
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"all-NaN loss is not finite: {loss_tensor}"


def test_recurrentae_no_nan() -> None:
    """Clean batch -> loss equals unmasked MSE/L1 (no regression)."""
    model = RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0, loss="mse")
    x = torch.randn(4, 32, 3)
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor)
    assert loss_tensor.item() > 0.0


# --------------------------------------------------------------------------- #
# TimeNet — NaN-padded tests
# --------------------------------------------------------------------------- #


def test_timenet_nan_padded_validation() -> None:
    """NaN-padded batch -> validation_step produces finite loss."""
    model = TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)
    x = torch.randn(4, 32, 3)
    x[:, -3:, :] = float("nan")  # trailing NaN padding
    batch = (x,)
    loss_tensor = model.validation_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"validation_step loss is not finite: {loss_tensor}"


def test_timenet_nan_padded_training() -> None:
    """NaN-padded batch -> training_step produces finite loss."""
    model = TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)
    x = torch.randn(4, 32, 3)
    x[:, -3:, :] = float("nan")  # trailing NaN padding
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"training_step loss is not finite: {loss_tensor}"


def test_timenet_all_nan_batch() -> None:
    """Fully-NaN batch -> finite loss, no crash."""
    model = TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)
    x = torch.full((4, 32, 3), float("nan"))
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor), f"all-NaN loss is not finite: {loss_tensor}"


def test_timenet_no_nan() -> None:
    """Clean batch -> loss equals unmasked MSE (no regression)."""
    model = TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)
    x = torch.randn(4, 32, 3)
    batch = (x,)
    loss_tensor = model.training_step(batch, 0)
    assert torch.isfinite(loss_tensor)
    assert loss_tensor.item() > 0.0
