"""Tests for Series2Vec downstream head migration to FineTuningModule.

Verifies that the old head class is removed and the new factory-based
approach works correctly with real Series2Vec backbones.
"""

from __future__ import annotations

import pytest
import torch

from tscollection.models._finetuning import make_series2vec_finetuner
from tscollection.models.convolutional.standard.series2vec.model import Series2Vec


class TestSeries2VecMigrateToFinetuningModule:
    """Verify Series2Vec fine-tuning via FineTuningModule works correctly."""

    def test_classification_shape(self) -> None:
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

    def test_regression_shape(self) -> None:
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

    def test_training_step_logs(self) -> None:
        """training_step returns a scalar loss."""
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

    def test_network_has_representation_dims_attr(self) -> None:
        """Series2VecNetwork stores _representation_dims (per D-02)."""
        backbone = Series2Vec(
            input_dims=2,
            embedding_dims=8,
            num_heads=2,
            feedforward_dims=16,
            representation_dims=4,
            dropout_rate=0.1,
        )
        assert hasattr(backbone.network, '_representation_dims')
        assert backbone.network._representation_dims == 4  # noqa: SLF001


class TestOldHeadRemoved:
    """Verify old head class is no longer importable."""

    def test_classification_head_removed(self) -> None:
        """Series2VecClassificationHead is no longer importable."""
        with pytest.raises(ImportError):
            from tscollection.models.convolutional.standard.series2vec import (  # noqa: F401, PLC0415
                Series2VecClassificationHead,
            )
