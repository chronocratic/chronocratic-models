"""Batch_size=1 training backward crash test for Phase 11 models.

Verifies that loss computation + backward() works at batch_size=1 without
gradient-disconnect crashes. Phase 11 norm fixes ensure encode() gradient
flow, but loss functions may return disconnected scalars (new_tensor) that
crash backward().

We test _calculate_loss / _compute_loss / _step directly since training_step
requires a Lightning Trainer.
"""

import pytest
import torch

from chronocratic.models.convolutional.standard.mcl.model import MCL
from chronocratic.models.convolutional.standard.series2vec.model import Series2Vec
from chronocratic.models.convolutional.standard.tstcc.model import TSTCC


class TestSeries2VecLossBatchSize1:
    """Series2Vec pretraining_loss returns new_tensor(0.0) when no pairs exist."""

    @pytest.fixture
    def model(self) -> Series2Vec:
        return Series2Vec(
            input_dims=3,
            embedding_dims=8,
            representation_dims=16,
            encoder_kernel_size=4,
            num_heads=2,
            feedforward_dims=32,
        )

    def test_backward_does_not_crash(self, model: Series2Vec) -> None:
        """_calculate_loss + backward at batch_size=1 should not raise."""
        model.train()
        x = torch.randn(1, 32, 3, requires_grad=True)
        loss, _, _ = model._calculate_loss(x)
        # Without fix: "element 0 of tensors does not require grad and does
        # not have a grad_fn" from pretraining_loss returning new_tensor(0.0).
        loss.backward()
        assert x.grad is not None

    def test_loss_is_finite(self, model: Series2Vec) -> None:
        """_calculate_loss at batch_size=1 should produce a finite value."""
        model.train()
        x = torch.randn(1, 32, 3, requires_grad=True)
        loss, _, _ = model._calculate_loss(x)
        assert torch.isfinite(loss), f"Loss is {loss}"


class TestTSTCCLossBatchSize1:
    """TSTCC uses TemporalContrast (nce accumulator) + NTXentLoss."""

    @pytest.fixture
    def model(self) -> TSTCC:
        return TSTCC(
            input_dims=3,
            conv_kernel_size=5,
            stride=2,
            output_dims=16,
            encoder_channels=(8, 16),
            encoder_inner_kernels=(5, 5),
            temporal_contrast_hidden_dim=32,
            temporal_contrast_timesteps=2,
        )

    def test_backward_does_not_crash(self, model: TSTCC) -> None:
        """_compute_loss + backward at batch_size=1 should not raise."""
        model.train()
        x = torch.randn(1, 64, 3, requires_grad=True)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        loss.backward()
        assert x.grad is not None

    def test_loss_is_finite(self, model: TSTCC) -> None:
        """_compute_loss at batch_size=1 should produce finite loss."""
        model.train()
        x = torch.randn(1, 64, 3, requires_grad=True)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        assert torch.isfinite(loss), f"Loss is {loss}"


class TestMCLLossBatchSize1:
    """MCL uses MixUpLoss — z_1==z_2==z_aug at B=1, gradient flows."""

    @pytest.fixture
    def model(self) -> MCL:
        return MCL(input_dims=3, output_dims=16)

    def test_backward_does_not_crash(self, model: MCL) -> None:
        """_step + backward at batch_size=1 should not raise."""
        model.train()
        x = torch.randn(1, 50, 3, requires_grad=True)
        loss = model._step(x)
        loss.backward()
        assert x.grad is not None

    def test_loss_is_finite(self, model: MCL) -> None:
        """_step at batch_size=1 should produce finite loss."""
        model.train()
        x = torch.randn(1, 50, 3, requires_grad=True)
        loss = model._step(x)
        assert torch.isfinite(loss), f"Loss is {loss}"
