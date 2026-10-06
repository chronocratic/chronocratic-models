"""Tests for TS-TCC's collapse detection.

A collapsed encoder pins the loss at chance (every logit equal). The model
tracks the gap between loss and chance and warns once when the
epoch-mean gap is still within ``_COLLAPSE_REL_TOL * chance`` of zero after
``_COLLAPSE_CHECK_MIN_STEPS`` training steps.
"""

import warnings

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models.convolutional.standard.tstcc.model import (
    _chance_loss,
    _COLLAPSE_CHECK_MIN_STEPS,
    _COLLAPSE_REL_TOL,
    TSTCC,
)

_BATCH_SIZE = 16
_CHANCE_AT_16 = _chance_loss(batch_size=_BATCH_SIZE, temporal_weight=1.0, contextual_weight=0.7)


def _small_model() -> TSTCC:
    """Build a tiny TSTCC so Trainer-driven tests stay fast."""
    return TSTCC(
        input_dim=1,
        conv_kernel_size=5,
        representation_dim=16,
        encoder_channels=(8, 8),
        temporal_contrast_hidden_dim=16,
    )


def _end_epoch(*, model: TSTCC, step_count: int, gap: float) -> list[warnings.WarningMessage]:
    """Fill one epoch's gap accumulators, end the epoch, and return its warnings.

    Args:
        model: Model whose collapse guard is exercised.
        step_count: Total training steps the model has seen so far.
        gap: Epoch-mean gap to chance to report.

    Returns:
        Warnings raised by ``on_train_epoch_end``.
    """
    model._train_step_count = step_count
    model._epoch_gap_sum = gap * 10
    model._epoch_chance_sum = _CHANCE_AT_16 * 10
    model._epoch_gap_count = 10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.on_train_epoch_end()
    return [warning for warning in caught if "collapsed" in str(warning.message)]


class TestChanceLoss:
    """_chance_loss matches the flat-logit loss of both terms."""

    def test_default_weights_at_batch_16(self) -> None:
        assert pytest.approx(7.949, abs=1e-3) == _CHANCE_AT_16

    def test_weights_scale_each_term(self) -> None:
        chance = _chance_loss(batch_size=4, temporal_weight=0.0, contextual_weight=1.0)
        assert chance == pytest.approx(torch.log(torch.tensor(7.0)).item())


class TestCollapseWarning:
    """One-shot UserWarning when the loss stays at chance past warm-up."""

    def test_fires_exactly_once_at_chance(self) -> None:
        model = _small_model()
        first = _end_epoch(model=model, step_count=_COLLAPSE_CHECK_MIN_STEPS, gap=0.002)
        second = _end_epoch(model=model, step_count=_COLLAPSE_CHECK_MIN_STEPS + 50, gap=0.0)
        assert len(first) == 1
        assert issubclass(first[0].category, UserWarning)
        assert "instance_normalize=False" in str(first[0].message)
        assert second == []

    def test_silent_below_tolerance(self) -> None:
        model = _small_model()
        gap = -2 * _COLLAPSE_REL_TOL * _CHANCE_AT_16
        assert _end_epoch(model=model, step_count=_COLLAPSE_CHECK_MIN_STEPS, gap=gap) == []

    def test_silent_before_min_steps(self) -> None:
        model = _small_model()
        assert _end_epoch(model=model, step_count=_COLLAPSE_CHECK_MIN_STEPS - 1, gap=0.0) == []

    def test_epoch_accumulators_reset(self) -> None:
        model = _small_model()
        _end_epoch(model=model, step_count=0, gap=0.0)
        assert model._epoch_gap_count == 0
        assert model._epoch_gap_sum == 0.0
        assert model._epoch_chance_sum == 0.0

    def test_guard_not_in_state_dict(self) -> None:
        model = _small_model()
        _end_epoch(model=model, step_count=_COLLAPSE_CHECK_MIN_STEPS, gap=0.0)
        assert model._collapse_warned is True
        assert not any("collapse" in key for key in model.state_dict())


class TestStepCounting:
    """training_step counts steps itself; ``global_step`` stays 0 under raw optimizers."""

    def test_steps_counted_without_logging_gap(self) -> None:
        torch.manual_seed(0)
        model = _small_model()
        dataset = TensorDataset(torch.rand(8, 64, 1), torch.zeros(8))
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, DataLoader(dataset, batch_size=4))

        assert "train_loss_gap_to_chance" not in trainer.callback_metrics
        assert model._train_step_count == 2
