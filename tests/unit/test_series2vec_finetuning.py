"""Tests for Series2Vec fine-tuning via FineTuningModule.

Verifies that the factory produces correct output shapes, logging works,
and freeze/unfreeze behaviour is correct for real Series2Vec backbones.
"""

from __future__ import annotations

import torch

from tscollection.models._finetuning import make_series2vec_finetuner
from tscollection.models.convolutional.standard.series2vec.model import Series2Vec


class TestSeries2VecFinetuningModule:
    """Verify Series2Vec fine-tuning via FineTuningModule works correctly."""

    def test_classification_output_shape(self) -> None:
        """make_series2vec_finetuner classification produces (B, num_outputs)."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_finetuner(
            backbone, num_outputs=5, task='classification', freeze_backbone=False
        )
        x = torch.randn(3, 20, 2)
        out = module(x)
        assert out.shape == (3, 5)

    def test_regression_output_shape(self) -> None:
        """make_series2vec_finetuner regression produces (B, num_outputs)."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_finetuner(
            backbone, num_outputs=2, task='regression', freeze_backbone=False
        )
        x = torch.randn(3, 20, 2)
        out = module(x)
        assert out.shape == (3, 2)

    def test_training_step_returns_scalar(self) -> None:
        """training_step returns a finite scalar loss."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_finetuner(
            backbone, num_outputs=5, task='classification', freeze_backbone=False
        )
        x = torch.randn(4, 20, 2)
        targets = torch.randint(0, 5, (4,))
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_freeze_backbone_prevents_grads(self) -> None:
        """freeze_backbone=True: backbone params don't receive gradients."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        module = make_series2vec_finetuner(
            backbone, num_outputs=5, task='classification', freeze_backbone=True
        )
        x = torch.randn(2, 20, 2)
        targets = torch.randint(0, 5, (2,))
        batch = (x, targets)
        loss = module.training_step(batch, 0)
        loss.backward()
        for param in backbone.parameters():
            assert param.grad is None
