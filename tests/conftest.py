"""Shared fixtures for producer integration tests."""
from collections.abc import Callable
import math

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Shared training helper
# --------------------------------------------------------------------------- #


def _run_train_steps(
    model: pl.LightningModule,
    batch_size: int = 4,
    seq_length: int = 100,
    input_dims: int = 1,
    num_steps: int = 5,
    seed: int | None = None,
    layout: str = "NLC",  # "NLC"=(B,T,D), "NCL"=(B,C,T)
    *,
    with_labels: bool = False,
) -> list[torch.Tensor]:
    """Run *num_steps* training steps and return collected losses.

    Shared implementation for all per-model producer tests so each test
    file only constructs the model under test.

    Args:
        with_labels: If True, produce (data, labels) tuple batches.
            Needed for TSTCC which calls extract_features_from_batch(batch).
    """
    if seed is not None:
        torch.manual_seed(seed)

    if layout == "NLC":
        data = torch.randn(batch_size * num_steps, seq_length, input_dims)
    else:  # NCL
        data = torch.randn(batch_size * num_steps, input_dims, seq_length)

    if with_labels:
        labels = torch.zeros(batch_size * num_steps, dtype=torch.long)
        dataset = TensorDataset(data, labels)
    else:
        dataset = TensorDataset(data)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    collected: list[torch.Tensor] = []
    original_step = model.training_step

    def patched_step(
        batch: torch.Tensor | tuple[torch.Tensor, torch.LongTensor],
        batch_idx: int,
    ) -> torch.Tensor | None:
        loss = original_step(batch, batch_idx)
        if loss is not None:
            collected.append(loss.clone().detach())
        return loss

    model.training_step = patched_step  # type: ignore[method-assign]

    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=num_steps,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    return collected


# --------------------------------------------------------------------------- #
# Reusable pytest fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def train_steps() -> Callable[..., list[torch.Tensor]]:
    """Return the _run_train_steps helper so tests can pass a model.

    Usage in tests:
        losses = train_steps(model, batch_size=4, num_steps=5)
        losses = train_steps(model, batch_size=4, num_steps=5, with_labels=True)
    """
    return _run_train_steps


@pytest.fixture
def random_data() -> Callable[..., torch.Tensor]:
    """Factory for random time-series tensors.

    Usage:
        data = random_data(batch=4, seq_length=100, input_dims=1)
        data_ncl = random_data(batch=4, seq_length=100, input_dims=1, layout="NCL")
    """
    def _factory(
        batch: int = 4,
        seq_length: int = 100,
        input_dims: int = 1,
        layout: str = "NLC",
    ) -> torch.Tensor:
        if layout == "NLC":
            return torch.randn(batch, seq_length, input_dims)
        return torch.randn(batch, input_dims, seq_length)

    return _factory


# --------------------------------------------------------------------------- #
# Shared assertion helpers
# --------------------------------------------------------------------------- #


def assert_finite_losses(
    losses: list[torch.Tensor], expected_min: int = 1
) -> None:
    """Assert all losses are finite scalars."""
    assert len(losses) >= expected_min
    for i, loss in enumerate(losses):
        assert loss is not None
        assert loss.ndim == 0, "Loss must be a scalar tensor"
        assert math.isfinite(loss.item()), (
            f"Loss at step {i} is not finite: {loss.item()}"
        )


@pytest.fixture
def finite_losses() -> Callable[..., None]:
    """Return the assert_finite_losses helper.

    Usage in tests:
        finite_losses(losses, expected_min=5)
    """
    return assert_finite_losses
