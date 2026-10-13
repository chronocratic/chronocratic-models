"""Single Trainer.fit harness that records losses and gradient health."""

from collections import defaultdict
from dataclasses import dataclass

import lightning.pytorch as pl
import torch

__all__ = ["FitRecord", "fit_model"]


@dataclass(frozen=True)
class FitRecord:
    """What a short training run observed.

    Attributes:
        epoch_losses: Mean training loss per epoch.
        grad_seen: Top-level submodule name -> a finite, nonzero gradient was seen.
            Every group holding a trainable parameter is listed, so a group cut off
            from the loss (``grad is None`` throughout) shows up as ``False``.
        nonfinite_grad: Any NaN/Inf gradient was seen.
    """

    epoch_losses: tuple[float, ...]
    grad_seen: dict[str, bool]
    nonfinite_grad: bool


class _Recorder(pl.Callback):
    """Collects step losses per epoch and per-group gradient health."""

    def __init__(self, *, loss_key: str | None) -> None:
        self._loss_key = loss_key
        self._step_losses: dict[int, list[float]] = defaultdict(list)
        self._metric_losses: list[float] = []
        self._grad_seen: dict[str, bool] = {}
        self._nonfinite = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,  # noqa: ARG002
        outputs: object,
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor):
            self._step_losses[trainer.current_epoch].append(float(loss.detach()))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        if self._loss_key is not None and self._loss_key in trainer.callback_metrics:
            self._metric_losses.append(float(trainer.callback_metrics[self._loss_key]))

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        for name, parameter in pl_module.named_parameters():
            if parameter.requires_grad:
                self._grad_seen.setdefault(name.split(".")[0], False)

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is None:
                continue
            group = name.split(".")[0]
            if not torch.isfinite(parameter.grad).all():
                self._nonfinite = True
            elif parameter.grad.abs().sum() > 0:
                self._grad_seen[group] = True

    def record(self) -> FitRecord:
        if self._step_losses:
            losses = tuple(
                sum(values) / len(values) for _, values in sorted(self._step_losses.items())
            )
        else:
            losses = tuple(self._metric_losses)
        return FitRecord(
            epoch_losses=losses, grad_seen=dict(self._grad_seen), nonfinite_grad=self._nonfinite
        )


def fit_model(
    *,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    epochs: int,
    loss_key: str | None = None,
) -> FitRecord:
    """Fit ``model`` on ``datamodule`` for ``epochs`` epochs on CPU.

    Args:
        model: Module to train (already seeded/built by the caller).
        datamodule: Data source.
        epochs: Number of epochs.
        loss_key: Logged-metric name used when ``training_step`` returns ``None``.

    Returns:
        A :class:`FitRecord` of the run.
    """
    recorder = _Recorder(loss_key=loss_key)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
        callbacks=[recorder],
    )
    trainer.fit(model, datamodule=datamodule)
    return recorder.record()
