# ruff: noqa: D, PLR2004, S101
"""Verify TimeVAE configure_optimizers returns ReduceLROnPlateau scheduler.

Original TF always adds ReduceLROnPlateau(factor=0.5, patience=30).
Without it, the model oscillates at fixed LR and can't converge past
the plateau (~160 loss after 250 epochs).
"""

import torch

from chronocratic.models.generative.timevae import TimeVAE


class TestReduceLROnPlateau:
    """configure_optimizers must return scheduler dict, not bare optimizer."""

    def test_returns_dict_with_scheduler(self) -> None:
        """Result is a dict containing 'optimizer' and 'lr_scheduler'."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_optimizer_is_adam(self) -> None:
        """Optimizer is Adam with eps=1e-7 (Keras parity)."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        opt = result["optimizer"]
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["eps"] == 1e-7

    def test_scheduler_is_reduce_lr_on_plateau(self) -> None:
        """LR scheduler is ReduceLROnPlateau with correct params."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        sched_cfg = result["lr_scheduler"]["scheduler"]
        assert isinstance(sched_cfg, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched_cfg.factor == 0.5
        assert sched_cfg.patience == 30
        assert sched_cfg.mode == "min"

    def test_scheduler_monitors_train_loss_epoch(self) -> None:
        """Scheduler monitors the epoch-level aggregated metric."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        assert result["lr_scheduler"]["monitor"] == "train_loss_epoch"

    def test_lr_changes_on_plateau_step(self) -> None:
        """ReduceLROnPlateau steps can be called without error."""
        model = TimeVAE(sequence_length=16, input_dim=1)
        result = model.configure_optimizers()
        opt = result["optimizer"]
        sched_cfg = result["lr_scheduler"]["scheduler"]
        initial_lr = opt.param_groups[0]["lr"]

        # Simulate no improvement: patience=30, reduction fires at step 31.
        # Run 32 steps to ensure we pass the reduction point.
        for _ in range(32):
            sched_cfg.step(metrics=100.0)

        # LR should have been halved
        assert opt.param_groups[0]["lr"] < initial_lr
