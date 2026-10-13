"""Pure "did it learn" checks. Each raises AssertionError with a diagnostic message."""

from collections.abc import Callable, Mapping
import math

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from tests.pipeline.datamodules import SyntheticClassificationDataModule
from tests.pipeline.harness import FitRecord

__all__ = [
    "encode_features",
    "gradients_flow",
    "linear_probe_accuracy",
    "loss_dropped",
    "probe_beats_chance",
    "recon_beats_baseline",
    "supervised_accuracy",
]


def loss_dropped(*, record: FitRecord, max_ratio: float) -> None:
    """Assert the late-epoch mean loss fell clearly below the early-epoch mean.

    Early/late windows are the first/last quarter of epochs (at least one), which
    tolerates the epoch-to-epoch noise of queue-based contrastive losses.

    Args:
        record: Result of :func:`fit_model`.
        max_ratio: Required ``last <= max_ratio * first`` (sign-safe: uses ``abs(first)``).
    """
    losses = record.epoch_losses
    assert len(losses) >= 2, f"need >= 2 epochs of loss, got {losses}"
    assert all(math.isfinite(value) for value in losses), f"non-finite loss in {losses}"
    window = max(1, len(losses) // 4)
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    assert first - last >= (1 - max_ratio) * abs(first), (
        f"loss did not drop enough: early mean {first:.4f}, late mean {last:.4f}, "
        f"required ratio <= {max_ratio}"
    )


def gradients_flow(*, record: FitRecord, frozen_groups: frozenset[str] = frozenset()) -> None:
    """Assert gradients were finite and every trainable parameter group received a nonzero one.

    Args:
        record: Result of :func:`fit_model`.
        frozen_groups: Top-level groups that are trainable but updated outside backprop
            (e.g. an EMA/SWA copy), so they are expected never to receive a gradient.
    """
    assert not record.nonfinite_grad, "non-finite gradient observed"
    dead = sorted(
        group for group, seen in record.grad_seen.items() if not seen and group not in frozen_groups
    )
    assert not dead, f"no nonzero gradient ever reached parameter groups: {dead}"


def linear_probe_accuracy(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_classes: int,
    seed: int = 0,
) -> float:
    """Fit a softmax-regression probe on standardized features; return test accuracy."""
    train_x = torch.nan_to_num(train_x)
    test_x = torch.nan_to_num(test_x)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    generator = torch.Generator().manual_seed(seed)
    probe = nn.Linear(train_x.shape[1], n_classes)
    with torch.no_grad():
        probe.weight.copy_(0.01 * torch.randn(probe.weight.shape, generator=generator))
        probe.bias.zero_()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(300):
        optimizer.zero_grad()
        F.cross_entropy(probe(train_x), train_y).backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(test_x).argmax(dim=1) == test_y).float().mean())


def encode_features(
    *, model: pl.LightningModule, x: torch.Tensor, encode_kwargs: Mapping[str, object]
) -> torch.Tensor:
    """Run ``model.encode`` and return finite 2-D features ``(N, D)``."""
    features = model.encode(x, batch_size=64, num_workers=0, **encode_kwargs)  # type: ignore[operator]
    features = torch.as_tensor(features).detach().float().flatten(start_dim=1)
    return torch.nan_to_num(features)


def probe_beats_chance(
    *,
    model: pl.LightningModule,
    datamodule: SyntheticClassificationDataModule,
    encode_kwargs: Mapping[str, object],
    margin: float,
) -> None:
    """Assert a linear probe on ``encode()`` features beats chance by ``margin``.

    Catches a collapsed encoder (constant or label-blind features), which a
    falling contrastive loss alone does not rule out.
    """
    train_x, train_y = datamodule.train_arrays()
    test_x, test_y = datamodule.test_arrays()
    accuracy = linear_probe_accuracy(
        train_x=encode_features(model=model, x=train_x, encode_kwargs=encode_kwargs),
        train_y=train_y,
        test_x=encode_features(model=model, x=test_x, encode_kwargs=encode_kwargs),
        test_y=test_y,
        n_classes=datamodule.num_classes,
    )
    chance = 1 / datamodule.num_classes
    assert accuracy > chance + margin, f"probe at chance: {accuracy:.3f} <= {chance:.3f} + {margin}"


def recon_beats_baseline(
    *,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    reconstruct: Callable[[pl.LightningModule, torch.Tensor], torch.Tensor],
    max_ratio: float,
) -> None:
    """Assert reconstruction MSE on test data clearly beats predicting the training mean.

    Args:
        model: Trained generative module.
        datamodule: Data source; its train split gives the mean baseline.
        reconstruct: Maps ``(model, x)`` to the reconstruction of ``x``.
        max_ratio: Required ``error <= max_ratio * baseline``; an untrained model sits
            at ~1.0, so the margin rules out outputs that only shrink toward the mean.
    """
    datamodule.setup(stage="fit")
    train_x = torch.cat([_features(batch) for batch in datamodule.train_dataloader()])
    test_x = torch.cat([_features(batch) for batch in datamodule.test_dataloader()])
    train_x, test_x = torch.nan_to_num(train_x), torch.nan_to_num(test_x)
    model.eval()
    with torch.no_grad():
        error = F.mse_loss(reconstruct(model, test_x), test_x).item()
    baseline = F.mse_loss(train_x.mean(dim=0, keepdim=True).expand_as(test_x), test_x).item()
    assert error <= max_ratio * baseline, (
        f"reconstruction MSE {error:.4f} not below {max_ratio} x mean baseline {baseline:.4f}"
    )


def supervised_accuracy(
    *, model: pl.LightningModule, datamodule: SyntheticClassificationDataModule, margin: float
) -> None:
    """Assert test accuracy of a supervised module beats chance by ``margin``."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            encoder_inputs, targets = model._batch_adapter(batch)
            predictions = model(*encoder_inputs).argmax(dim=1)
            correct += int((predictions == targets).sum())
            total += len(targets)
    accuracy, chance = correct / total, 1 / datamodule.num_classes
    assert accuracy > chance + margin, (
        f"accuracy {accuracy:.3f} not above chance {chance:.3f} + {margin}"
    )


def _features(batch: torch.Tensor | tuple | list) -> torch.Tensor:
    return batch if isinstance(batch, torch.Tensor) else batch[0]
