"""Tests that the learn checks themselves can fail (so a broken model cannot pass silently)."""

import lightning.pytorch as pl
import pytest
import torch
from torch import nn

from tests.pipeline.checks import (
    gradients_flow,
    linear_probe_accuracy,
    loss_dropped,
    probe_beats_chance,
    recon_beats_baseline,
)
from tests.pipeline.datamodules import SyntheticClassificationDataModule
from tests.pipeline.harness import fit_model, FitRecord


def _record(*, losses: tuple[float, ...], grads: dict[str, bool] | None = None) -> FitRecord:
    return FitRecord(
        epoch_losses=losses,
        grad_seen=grads if grads is not None else {"enc": True},
        nonfinite_grad=False,
    )


def test_loss_dropped_passes_on_clear_drop() -> None:
    loss_dropped(record=_record(losses=(2.0, 1.5, 1.0)), max_ratio=0.9)


def test_loss_dropped_fails_on_flat_loss() -> None:
    with pytest.raises(AssertionError, match="did not drop"):
        loss_dropped(record=_record(losses=(2.0, 2.0, 1.99)), max_ratio=0.9)


def test_loss_dropped_fails_on_nan() -> None:
    with pytest.raises(AssertionError, match="non-finite"):
        loss_dropped(record=_record(losses=(2.0, float("nan"))), max_ratio=0.9)


def test_loss_dropped_handles_negative_losses() -> None:
    loss_dropped(record=_record(losses=(-1.0, -2.0)), max_ratio=0.9)


def test_gradients_flow_fails_on_dead_group() -> None:
    with pytest.raises(AssertionError, match="head"):
        gradients_flow(record=_record(losses=(1.0, 0.5), grads={"enc": True, "head": False}))


def test_gradients_flow_ignores_frozen_groups() -> None:
    record = _record(losses=(1.0, 0.5), grads={"enc": True, "ema": False})
    gradients_flow(record=record, frozen_groups=frozenset({"ema"}))


def test_gradients_flow_fails_on_nonfinite_gradient() -> None:
    record = FitRecord(epoch_losses=(1.0, 0.5), grad_seen={"enc": True}, nonfinite_grad=True)
    with pytest.raises(AssertionError, match="non-finite"):
        gradients_flow(record=record)


def test_probe_accuracy_is_high_on_separable_and_chance_on_constant_features() -> None:
    labels = torch.arange(90) % 3
    separable = torch.nn.functional.one_hot(labels, num_classes=3).float() + 0.05 * torch.randn(
        90, 3
    )
    kwargs = {"train_y": labels[:60], "test_y": labels[60:], "n_classes": 3}
    good = linear_probe_accuracy(train_x=separable[:60], test_x=separable[60:], **kwargs)
    collapsed = linear_probe_accuracy(train_x=torch.ones(60, 3), test_x=torch.ones(30, 3), **kwargs)
    assert good > 0.95
    assert collapsed <= 0.4  # constant features cannot beat 1/3 by much


class _ToyEncoder(pl.LightningModule):
    """Encoder with ``encode``: amplitude spectrum, or a constant when ``collapse``.

    Classes differ by frequency, so the spectrum separates them.
    """

    def __init__(self, *, collapse: bool) -> None:
        super().__init__()
        self.collapse = collapse

    def encode(self, data: torch.Tensor, batch_size: int, num_workers: int = 0) -> torch.Tensor:
        del batch_size, num_workers
        features = torch.fft.rfft(data.nan_to_num(), dim=1).abs().flatten(start_dim=1)
        return torch.zeros_like(features) if self.collapse else features


def test_probe_beats_chance_accepts_informative_encoder() -> None:
    dm = SyntheticClassificationDataModule(seq_len=32)
    dm.setup(stage="fit")
    probe_beats_chance(
        model=_ToyEncoder(collapse=False), datamodule=dm, encode_kwargs={}, margin=0.1
    )


def test_probe_beats_chance_rejects_collapsed_encoder() -> None:
    dm = SyntheticClassificationDataModule(seq_len=32)
    dm.setup(stage="fit")
    with pytest.raises(AssertionError, match="chance"):
        probe_beats_chance(
            model=_ToyEncoder(collapse=True), datamodule=dm, encode_kwargs={}, margin=0.1
        )


class _DetachedHeadAutoEncoder(pl.LightningModule):
    """``head`` output is detached from the loss, so its gradient stays ``None``."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        output = self.layer(batch) + self.head(batch).detach()
        return ((output - batch) ** 2).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.layer.parameters(), lr=0.1)


def test_fit_model_marks_group_without_gradient_as_dead() -> None:
    dm = SyntheticClassificationDataModule(seq_len=16, n_channels=1, loader_mode="sample")
    record = fit_model(model=_DetachedHeadAutoEncoder(), datamodule=dm, epochs=1)
    assert record.grad_seen == {"layer": True, "head": False}
    with pytest.raises(AssertionError, match="head"):
        gradients_flow(record=record)


class _ToyAutoEncoder(pl.LightningModule):
    def __init__(self, *, identity: bool) -> None:
        super().__init__()
        self.identity = identity
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        return ((self.layer(batch) - batch) ** 2).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.1)


def test_recon_beats_baseline_distinguishes_identity_from_zero_output() -> None:
    dm = SyntheticClassificationDataModule(seq_len=16, n_channels=1)
    dm.setup(stage="fit")
    good = _ToyAutoEncoder(identity=True)
    recon_beats_baseline(model=good, datamodule=dm, reconstruct=lambda _model, x: x, max_ratio=0.8)
    with pytest.raises(AssertionError, match="baseline"):
        recon_beats_baseline(
            model=good,
            datamodule=dm,
            reconstruct=lambda _model, x: torch.zeros_like(x) + 5.0,
            max_ratio=0.8,
        )


def test_recon_beats_baseline_rejects_output_that_only_shrinks_toward_mean() -> None:
    dm = SyntheticClassificationDataModule(seq_len=16, n_channels=1)
    dm.setup(stage="fit")
    with pytest.raises(AssertionError, match="baseline"):
        recon_beats_baseline(
            model=_ToyAutoEncoder(identity=True),
            datamodule=dm,
            reconstruct=lambda _model, x: 0.05 * x,
            max_ratio=0.8,
        )


def test_fit_model_records_decreasing_loss_and_gradients() -> None:
    dm = SyntheticClassificationDataModule(seq_len=16, n_channels=1, loader_mode="sample")
    model = _ToyAutoEncoder(identity=True)
    record = fit_model(model=model, datamodule=dm, epochs=4)
    assert len(record.epoch_losses) == 4
    assert record.epoch_losses[-1] < record.epoch_losses[0]
    assert record.grad_seen == {"layer": True}
    assert not record.nonfinite_grad
