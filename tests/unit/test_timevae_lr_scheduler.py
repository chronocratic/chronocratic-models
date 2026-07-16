# ruff: noqa: D, PLR2004, S101
"""Verify TimeVAE configure_optimizers returns ReduceLROnPlateau scheduler.

Original TF always adds ReduceLROnPlateau(factor=0.5, patience=30).
Without it, the model oscillates at fixed LR and can't converge past
the plateau (~160 loss after 250 epochs).
"""

import torch

from chronocratic.models.generative.timevae import TimeVAE


class TestReduceLROnPlateau:
    """configure_optimizers must return scheduler list, not bare optimizer."""

    def test_returns_list_with_scheduler(self) -> None:
        """Result is a tuple of ([optimizer], [lr_scheduler_config])."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_optimizer_is_adam(self) -> None:
        """Optimizer is Adam with eps=1e-7 (Keras parity)."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        opt = result[0][0]
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["eps"] == 1e-7

    def test_scheduler_is_reduce_lr_on_plateau(self) -> None:
        """LR scheduler is ReduceLROnPlateau with correct params."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        sched_cfg = result[1][0]
        assert isinstance(sched_cfg["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched_cfg["scheduler"].factor == 0.5
        assert sched_cfg["scheduler"].patience == 30
        assert sched_cfg["scheduler"].mode == "min"

    def test_scheduler_monitors_train_loss_epoch(self) -> None:
        """Scheduler monitors the epoch-level aggregated metric."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        assert result[1][0]["monitor"] == "train_loss_epoch"

    def test_lr_changes_on_plateau_step(self) -> None:
        """ReduceLROnPlateau steps can be called without error."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        opt = result[0][0]
        sched_cfg = result[1][0]["scheduler"]
        initial_lr = opt.param_groups[0]["lr"]

        # Simulate no improvement: patience=30, reduction fires at step 31.
        # Run 32 steps to ensure we pass the reduction point.
        for _ in range(32):
            sched_cfg.step(metrics=100.0)

        # LR should have been halved
        assert opt.param_groups[0]["lr"] < initial_lr
