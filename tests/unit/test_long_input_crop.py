"""Tests for long-input cropping in TST and TimeVAE.

Verifies:
1. FixedPositionalEncoding numerical equivalence (forward vs old buffer).
2. FixedPositionalEncoding accepts input longer than sequence_length.
3. LearnablePositionalEncoding raises on overlong input.
4. TST trains on long batches when max_train_length is set.
5. TimeVAE trains on long batches when max_train_length is set.
6. Crop is a no-op when input already fits.
"""

import math

import pytest
import torch

from chronocratic.models import TimeVAE
from chronocratic.models.transformer.tst.model import TST
from chronocratic.models.transformer.tst.ts_transformer import (
    FixedPositionalEncoding,
    LearnablePositionalEncoding,
)

# ---------------------------------------------------------------------------
# Task 3A: FixedPositionalEncoding numerical equivalence
# ---------------------------------------------------------------------------


def _build_old_reference_pe(
    sequence_length: int, hidden_dim: int, scale_factor: float = 1.0
) -> torch.Tensor:
    """Reproduce the old buffer-based encoding for comparison."""
    pe = torch.zeros(sequence_length, hidden_dim)
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
    return pe


class TestFixedPositionalEncodingNumericalEquivalence:
    """New forward-time encoding must match the old buffer for T <= sequence_length."""

    def test_equivalence_at_sequence_length(self) -> None:
        hidden_dim = 16
        sequence_length = 32
        batch_size = 2
        pe_buffer = _build_old_reference_pe(sequence_length, hidden_dim)

        module = FixedPositionalEncoding(
            hidden_dim=hidden_dim, dropout_rate=0.0, sequence_length=sequence_length
        )
        module.eval()
        x = torch.zeros(sequence_length, batch_size, hidden_dim)

        with torch.no_grad():
            new_output = module(x)
            ref_output = x + pe_buffer[:sequence_length, :]

        assert torch.allclose(new_output, ref_output, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 3B: FixedPositionalEncoding accepts overlong input
# ---------------------------------------------------------------------------


class TestFixedPositionalEncodingOverlong:
    """FixedPositionalEncoding now accepts inputs longer than sequence_length."""

    def test_overlong_input(self) -> None:
        module = FixedPositionalEncoding(hidden_dim=16, sequence_length=8)
        x = torch.zeros(100, 2, 16)
        out = module(x)
        assert out.shape == (100, 2, 16)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Task 3C: LearnablePositionalEncoding raises on overlong input
# ---------------------------------------------------------------------------


class TestLearnablePositionalEncodingOverlong:
    """LearnablePositionalEncoding raises ValueError on overlong input."""

    def test_raises_on_overlong(self) -> None:
        module = LearnablePositionalEncoding(hidden_dim=16, sequence_length=8)
        x = torch.zeros(100, 2, 16)
        with pytest.raises(ValueError, match="cannot be extrapolated"):
            module(x)


# ---------------------------------------------------------------------------
# Task 3D: TST trains with max_train_length
# ---------------------------------------------------------------------------


class TestTSTMaxTrainLength:
    """TST trains on long batches when max_train_length is set."""

    def test_training_step_long_batch(self) -> None:
        model = TST(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, max_train_length=16
        )
        model.train()
        batch = torch.randn(2, 512, 3)
        loss = model.training_step(batch, 0)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_negative_control_no_crop(self) -> None:
        """Without max_train_length, long batches do NOT raise.

        After the FixedPositionalEncoding fix, the positional encoding accepts
        any input length. The transformer encoder and output_layer (Linear)
        operate on the last dimension only, so they also accept arbitrary T.
        The batch completes without error — self-attention succeeds, though
        O(T^2) memory makes this impractical for very large T.
        """
        model = TST(
            input_dim=3,
            sequence_length=16,
            hidden_dim=8,
            num_heads=2,
            depth=1,
            max_train_length=None,
        )
        model.train()
        # Use a modest length to avoid OOM during testing.
        batch = torch.randn(2, 128, 3)
        loss = model.training_step(batch, 0)
        assert loss.shape == ()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Task 3E: TimeVAE trains with max_train_length
# ---------------------------------------------------------------------------


class TestTimeVAEMaxTrainLength:
    """TimeVAE trains on long batches when max_train_length is set."""

    def test_training_step_long_batch(self) -> None:
        model = TimeVAE(
            sequence_length=16,
            input_dim=3,
            latent_dim=4,
            hidden_layer_sizes=(4, 8),
            max_train_length=16,
        )
        model.train()
        batch = torch.randn(2, 512, 3)
        loss = model.training_step(batch, 0)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_negative_control_no_crop(self) -> None:
        """Without max_train_length, long batches raise RuntimeError."""
        model = TimeVAE(
            sequence_length=16,
            input_dim=3,
            latent_dim=4,
            hidden_layer_sizes=(4, 8),
            max_train_length=None,
        )
        model.train()
        batch = torch.randn(2, 512, 3)
        with pytest.raises(RuntimeError):
            model.training_step(batch, 0)


# ---------------------------------------------------------------------------
# Task 3F: Crop is no-op when input fits
# ---------------------------------------------------------------------------


class TestCropNoOp:
    """process_sample_length is a no-op when input already fits."""

    def test_tst_loss_unchanged_when_fitting(self) -> None:
        torch.manual_seed(42)
        batch = torch.randn(2, 16, 3)
        batch = batch.clone()  # isolate seed state

        torch.manual_seed(42)
        model_with_crop = TST(
            input_dim=3, sequence_length=16, hidden_dim=8, num_heads=2, depth=1, max_train_length=16
        )
        model_with_crop.train()
        loss_with = model_with_crop.training_step(batch, 0)

        torch.manual_seed(42)
        model_no_crop = TST(
            input_dim=3,
            sequence_length=16,
            hidden_dim=8,
            num_heads=2,
            depth=1,
            max_train_length=None,
        )
        model_no_crop.train()
        loss_without = model_no_crop.training_step(batch, 0)

        assert torch.allclose(loss_with, loss_without)

    def test_timevae_loss_unchanged_when_fitting(self) -> None:
        torch.manual_seed(42)
        batch = torch.randn(2, 16, 3)
        batch = batch.clone()  # isolate seed state

        torch.manual_seed(42)
        model_with_crop = TimeVAE(
            sequence_length=16,
            input_dim=3,
            latent_dim=4,
            hidden_layer_sizes=(4, 8),
            max_train_length=16,
        )
        model_with_crop.train()
        loss_with = model_with_crop.training_step(batch, 0)

        torch.manual_seed(42)
        model_no_crop = TimeVAE(
            sequence_length=16,
            input_dim=3,
            latent_dim=4,
            hidden_layer_sizes=(4, 8),
            max_train_length=None,
        )
        model_no_crop.train()
        loss_without = model_no_crop.training_step(batch, 0)

        assert torch.allclose(loss_with, loss_without)
