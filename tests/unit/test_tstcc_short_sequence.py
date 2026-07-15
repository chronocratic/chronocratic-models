"""TSTCC short-sequence encoder tests.

Verifies that TSTCC correctly computes its encoder output sequence length,
auto-clamps temporal contrast timesteps when the encoder shrinks the sequence
below the default, and handles NaN-padded inputs without producing infinite
losses.

Covers:
- Encoder output length formula: pool(pool(pool(T))) with pool(L) = L//2+1
- Auto-clamp of temporal_contrast_timesteps at init and at runtime
- Forward pass on InsectWingbeat shape (seq_len=22, channels=200)
- TemporalContrast.forward with clamped timesteps
- Gradient flow on short sequences
- NaN defense in _compute_loss
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from chronocratic.models import TSTCC


# ---------------------------------------------------------------------------
# Reference formula (verified against empirical MaxPool1d(k=2,s=2,pad=1))
# ---------------------------------------------------------------------------


def _ref_encoder_output_length(seq_len: int) -> int:
    """Compute encoder output length using the pool compose formula.

    TCCEncoder has 3 MaxPool1d(kernel_size=2, stride=2, padding=1) stages.
    Each stage: output = L // 2 + 1 (verified empirically against PyTorch).
    """

    def pool(L: int) -> int:
        return L // 2 + 1

    return pool(pool(pool(seq_len)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tstcc_insect(**overrides: object) -> TSTCC:
    """Create a TSTCC model configured for InsectWingbeat parameters.

    Default params: input_dim=200, conv_kernel_size=8,
    temporal_contrast_timesteps=6 (will be clamped at init).
    """
    kwargs: dict[str, object] = {
        "input_dim": 200,
        "conv_kernel_size": 8,
        "temporal_contrast_timesteps": 6,
        **overrides,
    }
    return TSTCC(**kwargs)


def _make_nan_padded_batch(
    seq_len: int, channels: int, batch_size: int, pad_timesteps: int
) -> tuple[torch.Tensor, torch.LongTensor]:
    """Create a batch with trailing-NaN timesteps simulating variable-length padding."""
    data = torch.randn(batch_size, seq_len, channels)
    # Fill last `pad_timesteps` timesteps with NaN (all channels)
    if pad_timesteps > 0:
        data[:, -pad_timesteps:, :] = float("nan")
    labels = torch.zeros(batch_size, dtype=torch.long)
    return (data, labels)


# ---------------------------------------------------------------------------
# Test 1: Encoder output length formula
# ---------------------------------------------------------------------------


class TestEncoderOutputLengthFormula:
    """Verify the pool(L) = L//2+1 three-stage compose matches known values."""

    @pytest.mark.parametrize(
        "seq_len,expected",
        [
            (22, 4),
            (32, 5),
            (8, 2),
            (50, 8),
        ],
    )
    def test_pool_compose_known_values(self, seq_len: int, expected: int) -> None:
        """pool(pool(pool(T))) matches verified empirical values."""
        assert _ref_encoder_output_length(seq_len) == expected, (
            f"Encoder output length for T={seq_len} should be {expected}, "
            f"got {_ref_encoder_output_length(seq_len)}"
        )


# ---------------------------------------------------------------------------
# Test 2: TSTCC auto-clamp timesteps at init
# ---------------------------------------------------------------------------


class TestTSTCCAutoClampTimesteps:
    """Creating TSTCC with InsectWingbeat params should clamp timesteps and warn."""

    def test_tstcc_clamps_timesteps_at_init(self) -> None:
        """temporal_contrast_timesteps clamped from 6 to 3 for seq_len=22."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = _make_tstcc_insect(sequence_length=22)

        assert hasattr(model, "temporal_contrast_timesteps")
        assert model.temporal_contrast_timesteps == 3, (
            f"Expected timesteps clamped to 3 for seq_len=22, "
            f"got {model.temporal_contrast_timesteps}"
        )
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "timesteps" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Test 3: TSTCC forward on InsectWingbeat shape
# ---------------------------------------------------------------------------


class TestTSTCCForwardOnInsectWingbeat:
    """Forward pass on InsectWingbeat shape (batch=2, seq=22, channels=200)."""

    def test_forward_no_crash(self) -> None:
        """forward((2, 22, 200)) succeeds without ValueError from TemporalContrast."""
        model = _make_tstcc_insect(sequence_length=22)
        model.eval()
        x = torch.randn(2, 22, 200)
        with torch.no_grad():
            output = model.encode_batch(x)
        assert output.shape[0] == 2
        assert output.shape[-1] == model.representation_dim


# ---------------------------------------------------------------------------
# Test 4: TemporalContrast no crash with clamped timesteps
# ---------------------------------------------------------------------------


class TestTSTCCTemporalContrastNoCrash:
    """TemporalContrast.forward with clamped timesteps should not trigger assertion."""

    def test_temporal_contrast_forward_succeeds(self) -> None:
        """Clamped timesteps (3) < encoder output (4), so no ValueError."""
        model = _make_tstcc_insect(sequence_length=22)
        model.eval()
        x = torch.randn(2, 22, 200)
        # Encoder output should be 4, timesteps clamped to 3 -> 4 > 3, safe
        with torch.no_grad():
            features = model._encoder(x.transpose(1, 2))  # (B, C, T) -> (B, rep, L')
        actual_seq_len = features.shape[-1]
        assert actual_seq_len > model.temporal_contrast_timesteps, (
            f"Encoder output seq_len ({actual_seq_len}) must be > "
            f"clamped timesteps ({model.temporal_contrast_timesteps})"
        )


# ---------------------------------------------------------------------------
# Test 5: Gradient flow on short sequences
# ---------------------------------------------------------------------------


class TestTSTCCGradientsOnShortSeq:
    """Backward through _compute_loss on short sequences produces valid gradients."""

    def test_gradients_nonzero_finite(self) -> None:
        """backward produces non-zero finite gradients for seq_len=22."""
        model = _make_tstcc_insect(sequence_length=22)
        model.train()
        x = torch.randn(2, 22, 200, requires_grad=True)
        labels = torch.zeros(2, dtype=torch.long)
        batch = (x, labels)

        loss = model._compute_loss(batch)
        loss.backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"


# ---------------------------------------------------------------------------
# Test 6: NaN defense in _compute_loss
# ---------------------------------------------------------------------------


class TestTSTCCNaNDefense:
    """NaN-padded batches should produce finite loss after zero-fill."""

    def test_nan_padded_batch_finite_loss(self) -> None:
        """training_step on NaN-padded batch produces finite loss."""
        model = _make_tstcc_insect(sequence_length=22)
        model.train()
        batch = _make_nan_padded_batch(seq_len=22, channels=200, batch_size=2, pad_timesteps=5)

        loss = model._compute_loss(batch)
        assert torch.isfinite(loss), (
            f"Loss is not finite for NaN-padded batch: {loss.item()}"
        )
        assert math.isfinite(loss.item()), (
            f"Loss.item() is not finite: {loss.item()}"
        )
