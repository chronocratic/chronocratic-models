"""Tests for TST fine-tuning via SupervisedModule.

Verifies that the factory produces correct output shapes, logging works,
and freeze/unfreeze behaviour is correct for real TST backbones.
"""

from __future__ import annotations

import torch

from chronocratic.models.supervised import make_tst_supervised
from chronocratic.models.transformer.tst.model import TST


class TestTSTFinetuningModule:
    """Verify TST fine-tuning via SupervisedModule works correctly."""

    def test_classification_output_shape(self) -> None:
        """make_tst_supervised classification produces (B, num_outputs) output."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        x = torch.randn(3, 10, 2)
        padding_masks = torch.ones(3, 10, dtype=torch.bool)
        out = module(x, padding_masks)
        assert out.shape == (3, 5)

    def test_regression_output_shape(self) -> None:
        """make_tst_supervised regression produces (B, num_outputs) output."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=2, task="regression", freeze_backbone=False
        )
        x = torch.randn(3, 10, 2)
        padding_masks = torch.ones(3, 10, dtype=torch.bool)
        out = module(x, padding_masks)
        assert out.shape == (3, 2)

    def test_training_step_returns_scalar(self) -> None:
        """training_step returns a finite scalar loss."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        x = torch.randn(4, 10, 2)
        targets = torch.randint(0, 5, (4,))
        padding_masks = torch.ones(4, 10, dtype=torch.bool)
        ids = torch.arange(4)
        batch = (x, targets, padding_masks, ids)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_freeze_backbone_prevents_grads(self) -> None:
        """freeze_backbone=True: backbone params don't receive gradients."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=True
        )
        x = torch.randn(2, 10, 2)
        targets = torch.randint(0, 5, (2,))
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        ids = torch.arange(2)
        batch = (x, targets, padding_masks, ids)
        loss = module.training_step(batch, 0)
        loss.backward()
        for param in backbone.parameters():
            assert param.grad is None

    def test_unfrozen_backbone_receives_grads(self) -> None:
        """freeze_backbone=False: backbone params receive gradients."""
        backbone = TST(input_dims=2, sequence_length=10, hidden_dims=8, num_heads=2, depth=1)
        module = make_tst_supervised(
            backbone, num_outputs=5, task="classification", freeze_backbone=False
        )
        x = torch.randn(2, 10, 2)
        targets = torch.randint(0, 5, (2,))
        padding_masks = torch.ones(2, 10, dtype=torch.bool)
        ids = torch.arange(2)
        batch = (x, targets, padding_masks, ids)
        loss = module.training_step(batch, 0)
        loss.backward()
        grad_count = sum(1 for p in backbone.parameters() if p.grad is not None)
        assert grad_count > 0
