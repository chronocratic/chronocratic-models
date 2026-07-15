"""NaN-padded training smoke tests for all model families.

Verifies that every model (except TST, which is deferred pending batch-contract
redesign) produces a finite training loss when fed batches with trailing NaN
timesteps — the pattern produced by variable-length UEA/UCR datasets.

Models that call ``self.log()`` without ``self.optimizers()`` are tested via
``training_step`` directly.  Models that require ``self.optimizers()`` (TSTCC,
TS2Vec, CoST, AutoTCL) are tested via their internal loss methods or wrapped
in a Lightning Trainer.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models import (
    AutoTCL,
    CoST,
    MCL,
    RecurrentAutoEncoder,
    Series2Vec,
    TimeNet,
    TimeVAE,
    TS2Vec,
    TSTCC,
)

# --------------------------------------------------------------------------- #
# Helpers — build NaN-padded tensors
# --------------------------------------------------------------------------- #


def _nan_padded(batch: int = 4, seq_len: int = 32, input_dim: int = 3) -> torch.Tensor:
    """Random tensor with the last 3 timesteps set to NaN."""
    x = torch.randn(batch, seq_len, input_dim)
    x[:, -3:, :] = float("nan")
    return x


# --------------------------------------------------------------------------- #
# Reconstruction models — training_step (no self.optimizers)
# --------------------------------------------------------------------------- #


def test_nan_padded_training_step_recurrentae() -> None:
    """RecurrentAE: NaN-padded batch -> finite training_step loss."""
    model = RecurrentAutoEncoder(input_dim=3, layers=(16,), dropout=0.0)
    x = _nan_padded()
    batch = (x,)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"RecurrentAE loss not finite: {loss}"


def test_nan_padded_training_step_timenet() -> None:
    """TimeNet: NaN-padded batch -> finite training_step loss."""
    model = TimeNet(input_dim=3, hidden_dim=16, depth=1, dropout_rate=0.0)
    x = _nan_padded()
    batch = (x,)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"TimeNet loss not finite: {loss}"


def test_nan_padded_training_step_timevae() -> None:
    """TimeVAE: NaN-padded batch -> finite training_step loss."""
    model = TimeVAE(sequence_length=32, input_dim=3, latent_dim=8, hidden_layer_sizes=(16, 32))
    x = _nan_padded()
    batch = (x,)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"TimeVAE loss not finite: {loss}"


# --------------------------------------------------------------------------- #
# Contrastive models — _step / _calculate_loss (no trainer needed)
# --------------------------------------------------------------------------- #


def test_nan_padded_training_step_series2vec() -> None:
    """Series2Vec: NaN-padded batch -> finite training_step loss."""
    model = Series2Vec(
        input_dim=3,
        embedding_dim=8,
        representation_dim=16,
        temporal_kernel_size=4,
        num_heads=2,
        feedforward_dim=32,
    )
    x = _nan_padded()
    batch = (x,)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"Series2Vec loss not finite: {loss}"


def test_nan_padded_training_step_tstcc() -> None:
    """TSTCC: NaN-padded batch (data, labels) -> finite loss via _compute_loss."""
    model = TSTCC(
        input_dim=3,
        conv_kernel_size=5,
        stride=2,
        representation_dim=16,
        encoder_channels=(8, 16),
        encoder_inner_kernels=(5, 5),
        temporal_contrast_hidden_dim=32,
        temporal_contrast_timesteps=2,
    )
    x = _nan_padded()
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    batch = (x, labels)
    loss = model._compute_loss(batch)
    assert torch.isfinite(loss), f"TSTCC loss not finite: {loss}"


def test_nan_padded_training_step_mcl() -> None:
    """MCL: NaN-padded batch -> finite contrastive loss (zero-fill before mixup)."""
    model = MCL(
        input_dim=3,
        representation_dim=16,
        encoder_channels=(16, 32, 16),
        encoder_kernels=(5, 3, 3),
        encoder_dilations=(1, 2, 4),
        projection_dim=16,
    )
    x = _nan_padded()
    batch = (x,)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"MCL loss not finite: {loss}"


# --------------------------------------------------------------------------- #
# Dilated models — wrap in Trainer (require self.optimizers)
# --------------------------------------------------------------------------- #


def test_nan_padded_training_step_ts2vec() -> None:
    """TS2Vec: NaN-padded batch -> finite training loss via Trainer."""
    import lightning.pytorch as pl

    model = TS2Vec(input_dim=3, hidden_dim=16, representation_dim=32, depth=2)
    x = _nan_padded()
    dataloader = DataLoader(TensorDataset(x), batch_size=x.shape[0])

    losses: list[torch.Tensor] = []
    original_step = model.training_step

    def patched_step(*args, **kwargs):
        loss = original_step(*args, **kwargs)
        if loss is not None:
            losses.append(loss.clone().detach())
        return loss

    model.training_step = patched_step  # type: ignore[method-assign]
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    assert losses, "No losses collected"
    assert torch.isfinite(losses[0]), f"TS2Vec loss not finite: {losses[0]}"


def test_nan_padded_training_step_cost() -> None:
    """CoST: NaN-padded batch -> finite training loss via Trainer."""
    import lightning.pytorch as pl

    model = CoST(
        input_dim=3,
        sequence_length=32,
        hidden_dim=16,
        representation_dim=32,
        depth=2,
        max_train_length=32,
        queue_size=8,
    )
    x = _nan_padded()
    dataloader = DataLoader(TensorDataset(x), batch_size=x.shape[0])

    losses: list[torch.Tensor] = []
    original_step = model.training_step

    def patched_step(*args, **kwargs):
        loss = original_step(*args, **kwargs)
        if loss is not None:
            losses.append(loss.clone().detach())
        return loss

    model.training_step = patched_step  # type: ignore[method-assign]
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    assert losses, "No losses collected"
    assert torch.isfinite(losses[0]), f"CoST loss not finite: {losses[0]}"


def test_nan_padded_training_step_autotcl() -> None:
    """AutoTCL: NaN-padded batch -> finite training loss via Trainer."""
    import lightning.pytorch as pl

    model = AutoTCL(input_dim=3, hidden_dim=16, representation_dim=32, depth=2, max_train_length=32)
    x = _nan_padded()
    dataloader = DataLoader(TensorDataset(x), batch_size=x.shape[0])

    losses: list[torch.Tensor] = []
    original_step = model.training_step

    def patched_step(*args, **kwargs):
        loss = original_step(*args, **kwargs)
        if loss is not None:
            losses.append(loss.clone().detach())
        return loss

    model.training_step = patched_step  # type: ignore[method-assign]
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    assert losses, "No losses collected"
    assert torch.isfinite(losses[0]), f"AutoTCL loss not finite: {losses[0]}"
