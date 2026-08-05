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
from chronocratic.models.convolutional.standard.tstcc.temporal_contrast import TemporalContrast


class TestSeries2VecLossBatchSize1:
    """Series2Vec pretraining_loss returns new_tensor(0.0) when no pairs exist."""

    @pytest.fixture
    def model(self) -> Series2Vec:
        return Series2Vec(
            input_dim=3,
            embedding_dim=8,
            representation_dim=16,
            temporal_kernel_size=4,
            num_heads=2,
            feedforward_dim=32,
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

    def test_split_path_produces_encoder_gradient(self, model: Series2Vec) -> None:
        """Long singleton is windowed into pairs, yielding a real encoder gradient."""
        model.train()
        # L=256, K=2 -> window_len 128 >= kernel: the split path runs, not the dummy.
        x = torch.randn(1, 256, 3)
        loss, _, _ = model._calculate_loss(x)
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
        grads = [p.grad for p in model.network.parameters() if p.grad is not None]
        assert grads, "no encoder parameter received a gradient"
        assert any(torch.any(g != 0) for g in grads), "encoder gradient is all zeros"

    def test_unsplittable_series_falls_back_without_crash(self) -> None:
        """When windows would be shorter than the kernel, split is skipped safely."""
        # K=16 on L=32 -> window_len 2 < kernel 4: split skipped, x stays (1, 32, 3),
        # pretraining_loss finds no pairs -> connected zero-dummy fallback, no crash.
        model = Series2Vec(
            input_dim=3,
            embedding_dim=8,
            representation_dim=16,
            temporal_kernel_size=4,
            num_heads=2,
            feedforward_dim=32,
            singleton_split_count=16,
        )
        model.train()
        x = torch.randn(1, 32, 3, requires_grad=True)
        loss, _, _ = model._calculate_loss(x)
        assert torch.isfinite(loss)
        loss.backward()  # must not raise
        assert x.grad is not None


class TestTSTCCLossBatchSize1:
    """TSTCC uses TemporalContrast (nce accumulator) + NTXentLoss."""

    @pytest.fixture
    def model(self) -> TSTCC:
        return TSTCC(
            input_dim=3,
            conv_kernel_size=5,
            stride=2,
            representation_dim=16,
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

    def test_batch_size_1_produces_nonzero_loss(self, model: TSTCC) -> None:
        """B=1 should produce a nonzero loss after the singleton-split fix."""
        model.train()
        x = torch.randn(1, 300, 3)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        assert loss.item() != 0.0, "Loss is exactly 0.0 — contrastive objectives are degenerate"
        assert torch.isfinite(loss)

    def test_batch_size_1_produces_encoder_gradient(self, model: TSTCC) -> None:
        """Regression test: B=1 should produce nonzero encoder gradients."""
        model.train()
        x = torch.randn(1, 300, 3)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        loss.backward()
        grads = [p.grad for p in model._encoder.parameters() if p.grad is not None]
        assert grads, "no encoder parameter received a gradient"
        assert any(torch.any(g != 0) for g in grads), "encoder gradient is all zeros"

    def test_temporal_contrast_receives_gradient(self, model: TSTCC) -> None:
        """TemporalContrast sub-net should receive gradients at B=1."""
        model.train()
        x = torch.randn(1, 300, 3)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        loss.backward()
        grads = [p.grad for p in model._tc_model.parameters() if p.grad is not None]
        assert grads, "no tc_model parameter received a gradient"
        assert any(torch.any(g != 0) for g in grads), "tc_model gradient is all zeros"

    def test_short_series_does_not_crash(self, model: TSTCC) -> None:
        """Very short input should not crash — TemporalContrast clamps at runtime."""
        model.train()
        x = torch.randn(1, 12, 3)
        batch = (x, torch.zeros(1, dtype=torch.long))
        loss = model._compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()

    def test_tc_model_identity_is_stable_across_batches(self, model: TSTCC) -> None:
        """_tc_model must never be rebuilt mid-training (optimizer safety)."""
        model.train()
        tc_ref = model._tc_model

        # First batch
        loss1 = model._compute_loss((torch.randn(1, 300, 3), torch.zeros(1, dtype=torch.long)))
        loss1.backward()
        model.zero_grad()

        # Second batch with different length
        loss2 = model._compute_loss((torch.randn(2, 150, 3), torch.zeros(2, dtype=torch.long)))
        loss2.backward()

        assert model._tc_model is tc_ref, "_tc_model was rebuilt — optimizer is orphaned"

    def test_batch_size_gt_1_is_unchanged(self, model: TSTCC) -> None:
        """B>1 should work as before (no-op path)."""
        model.train()
        x = torch.randn(4, 100, 3)
        batch = (x, torch.zeros(4, dtype=torch.long))
        loss = model._compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model._encoder.parameters() if p.grad is not None]
        assert grads, "no encoder parameter received a gradient"
        assert any(torch.any(g != 0) for g in grads), "encoder gradient is all zeros"


class TestMCLLossBatchSize1:
    """MCL uses MixUpLoss — z_1==z_2==z_aug at B=1, gradient flows."""

    @pytest.fixture
    def model(self) -> MCL:
        return MCL(input_dim=3, representation_dim=16)

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


class TestTemporalContrastTimestepClamping:
    """TemporalContrast should clamp effective horizon at runtime."""

    @pytest.fixture
    def tc(self) -> TemporalContrast:
        return TemporalContrast(num_channels=16, hidden_dim=32, timesteps=6)

    def test_short_seq_len_degrades_gracefully(self, tc: TemporalContrast) -> None:
        """seq_len=3 < timesteps=6 should clamp and return finite loss."""
        f1 = torch.randn(2, 16, 3)
        f2 = torch.randn(2, 16, 3)
        nce, proj = tc(f1, f2)
        assert torch.isfinite(nce), f"Loss is {nce}"
        assert proj.shape == (2, 4)

    def test_seq_len_1_raises(self, tc: TemporalContrast) -> None:
        """seq_len=1 should raise ValueError — cannot predict future."""
        with pytest.raises(ValueError, match="seq_len"):
            tc(torch.randn(1, 16, 1), torch.randn(1, 16, 1))

    def test_long_seq_len_unchanged(self, tc: TemporalContrast) -> None:
        """seq_len >> timesteps should behave as before (no regression)."""
        torch.manual_seed(42)
        f1 = torch.randn(2, 16, 25)
        f2 = torch.randn(2, 16, 25)
        nce, _ = tc(f1, f2)
        assert torch.isfinite(nce)
        # Loss should be nonzero with random features at B=2
        assert nce.item() != 0.0
